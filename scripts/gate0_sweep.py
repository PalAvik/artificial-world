#!/usr/bin/env python3
"""Gate 0: find the cheapest render config the model can still read.

This is a constrained minimisation, not a pass/fail check (docs/GATES.md):

    minimise   visual tokens per span
    subject to read-back accuracy >= 95% (1 word) and >= 88% (3 words),
               at the LOWER BOUND of the 95% CI on training fonts, and above a
               lower floor on held-out fonts

The objective and the constraint pull against each other. Smaller strips and a
lower `min_pixels` keep the V_T/V_I cardinality gap near zero and every later
sweep cheap; larger ones make the glyphs legible.

Two refinements the first real sweep forced:

- **Thresholds are checked at the CI lower bound.** At n=128 a measured 0.953
  has a 95% interval of [0.90, 0.98] and does not establish 0.95. Freezing on
  that is freezing on noise, so the default n is 512.
- **Held-out fonts get a floor, not the train threshold.** They need to be
  legible enough that OCR failure stays a minority contributor to MSG. Holding
  them to the train threshold would be requiring the held-out set not to be
  held out. See select_winner.

Usage:
    python scripts/gate0_sweep.py --model Qwen/Qwen3.5-2B
    python scripts/gate0_sweep.py --model Qwen/Qwen3.5-2B --n 64 --quick
"""
from __future__ import annotations

import sys
from pathlib import Path

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root, so
# `freeflow` would not import. Works whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import itertools
import json
import os
import random
import time

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

# Canonical implementations live in the package, so the read-back that chose the
# render config and the read-back that conditions the Phase 1 metrics cannot
# drift apart.
from freeflow.metrics.cycle import levenshtein, normalise, read_back

# Word classes chosen to preview H2 (does the gap track abstractness and
# function-word status rather than length?) while keeping Gate 0 self-contained:
# no dataset download blocks week 1.
WORD_CLASSES: dict[str, list[str]] = {
    "function": ["the", "from", "of", "and", "but", "which", "into", "upon",
                 "nor", "yet", "than", "whom", "amid", "per", "via", "onto"],
    "concrete": ["dog", "table", "mountain", "bicycle", "window", "spoon",
                 "harbour", "lantern", "sparrow", "kettle", "bridge", "orchard",
                 "anvil", "quilt", "ferry", "walnut"],
    "abstract": ["justice", "theory", "freedom", "irony", "purpose", "doubt",
                 "custom", "merit", "hazard", "essence", "notion", "rigour",
                 "candour", "premise", "tenet", "whimsy"],
    "rare_long": ["quixotic", "obfuscate", "perspicacity", "antediluvian",
                  "sesquipedalian", "logorrhoea", "zeugma", "hapax",
                  "syzygy", "brobdingnagian", "eleemosynary", "pulchritude",
                  "crepuscular", "obstreperous", "vicissitude", "peripatetic"],
}


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the 95% Wilson score interval.

    The gate is checked against this rather than the point estimate. At n=128 a
    measured 0.953 carries a CI of [0.90, 0.98] — indistinguishable from 0.93,
    and freezing a config on that basis would be freezing on noise. Wilson
    rather than normal-approximation because accuracies here sit near 1.0,
    where the normal interval misbehaves.
    """
    if n == 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (centre - margin) / denom


def make_spans(n: int, n_words: int, rng: random.Random,
               pool: str = "gate0") -> list[tuple[str, str]]:
    """Return (text, class) pairs, balanced across word classes.

    `pool="spans"` draws from the Phase 1 vocabulary instead of this script's
    own held-out list. That is a *verification* mode, not a selection mode: the
    frozen render config must never be re-chosen on the words it will be
    measured on. Use it to answer "is the Phase 1 pool legible at the frozen
    config?", which matters because that pool averages 9.4 characters against
    the ~7 this config was frozen on, and Gate 0 established that read-back
    tracks length.
    """
    if pool == "spans":
        from freeflow.data import vocab
        picks = vocab.sample(n * n_words, rng, balanced=False)
        out = []
        for i in range(n):
            chunk = picks[i * n_words:(i + 1) * n_words]
            if not chunk:
                break
            out.append((" ".join(w for w, _ in chunk), chunk[0][1]))
        return out

    classes = list(WORD_CLASSES)
    out = []
    for i in range(n):
        cls = classes[i % len(classes)]
        words = rng.sample(WORD_CLASSES[cls], k=min(n_words, len(WORD_CLASSES[cls])))
        out.append((" ".join(words), cls))
    return out


def render(text: str, font_path: str, height: int, pad: int = 4) -> Image.Image:
    font = ImageFont.truetype(font_path, max(6, height - 2 * pad))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = int(probe.textlength(text, font=font)) + 2 * pad
    img = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(img).text((pad, pad), text, fill="black", font=font)
    return img


def visual_token_cost(processor, img: Image.Image) -> int:
    """Tokens the image adds, measured rather than derived from grid metadata,
    whose field names vary across processor versions."""
    probe = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    with_img = [{"role": "user", "content": [
        {"type": "text", "text": "x"}, {"type": "image", "image": img}]}]
    lens = []
    for msgs, images in ((probe, None), (with_img, [img])):
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        lens.append(processor(text=[text], images=images,
                              return_tensors="pt")["input_ids"].shape[1])
    return lens[1] - lens[0]


def evaluate(model, processor, fonts: dict[str, str], height: int,
             spans_1: list[tuple[str, str]], spans_3: list[tuple[str, str]],
             batch: int) -> dict:
    """Read-back accuracy for one render config over one font set."""
    names = sorted(fonts)
    result = {}
    for label, spans in (("w1", spans_1), ("w3", spans_3)):
        # Cycle fonts across spans so every config sees the whole font set.
        images = [render(t, fonts[names[i % len(names)]], height)
                  for i, (t, _) in enumerate(spans)]
        preds = read_back(model, processor, images, batch)
        # zip() would silently truncate and quietly compute accuracy over fewer
        # items than were tested — a wrong number rather than an error.
        assert len(preds) == len(spans), (
            f"read_back returned {len(preds)} predictions for {len(spans)} spans")
        hits, cer_num, cer_den, by_class = 0, 0, 0, {}
        for (truth, cls), pred in zip(spans, preds):
            ok = normalise(pred) == normalise(truth)
            hits += ok
            cer_num += levenshtein(normalise(pred), normalise(truth))
            cer_den += max(1, len(normalise(truth)))
            c = by_class.setdefault(cls, [0, 0])
            c[0] += ok
            c[1] += 1
        result[label] = {
            "acc": hits / max(1, len(spans)),
            "hits": hits,
            "n": len(spans),
            "acc_lo": wilson_lower(hits, len(spans)),
            "cer": cer_num / max(1, cer_den),
            "by_class": {k: v[0] / v[1] for k, v in sorted(by_class.items())},
        }
    return result


def select_winner(rows: list[dict]) -> dict | None:
    """The constrained minimisation: cheapest config meeting both constraints.

    Two constraints, calibrated differently on purpose:

    1. **Train legibility at the CI lower bound**, against the full thresholds.
       A point estimate that grazes a threshold is not evidence of clearing it.
    2. **Held-out-font legibility against a lower floor** (`--held-margin`
       below the train thresholds, 5 points by default).

    Why the floor is lower rather than equal. Gate 2(a) evaluates the trained
    model on held-out fonts, so those fonts must be legible enough that OCR
    failure is a minority contributor to MSG — not so legible that they match
    training typefaces. At 0.94 read-back, 94% of spans are read correctly and
    their MSG is clean signal; the residual 6% is handled properly by reporting
    MSG both unconditionally and conditioned on correct read-back, which the
    metric suite requires anyway (PLAN.md §5). Demanding the *train* threshold
    on held-out fonts would be requiring the held-out set not to be held out.

    This constraint does look at the held-out set. It checks a validity
    precondition of the measuring instrument rather than tuning behaviour toward
    the split — closer to confirming the test set isn't corrupted than to
    fitting on it — but it is a judgement call, so it is stated rather than
    buried.

    Ties on visual-token count break toward higher accuracy. Returns None when
    nothing qualifies — a gate outcome, not an error (docs/GATES.md).
    """
    qualifying = [r for r in rows if r["passes"] and r.get("held_ok", True)]
    if not qualifying:
        return None
    return min(qualifying, key=lambda r: (r["visual_tokens"],
                                          -r["train"]["w1"]["acc_lo"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fonts", default="configs/fonts.yaml")
    ap.add_argument("--out", default="configs/render.yaml")
    ap.add_argument("--log", default="results/gate0_sweep.json")
    ap.add_argument("--n", type=int, default=512, help="spans per config, per length. 512 is the smallest n whose "
                         "CI can resolve a 0.97 point estimate against a "
                         "0.95 threshold; 128 cannot.")
    ap.add_argument("--batch", type=int, default=32,
                    help="32 sits past the throughput knee (docs/COMPUTE.md)")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--heights", type=int, nargs="+", default=[16, 24, 32, 48])
    ap.add_argument("--min-pixels", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--quick", action="store_true",
                    help="two configs and n=32, to shake out the pipeline")
    ap.add_argument("--acc1", type=float, default=0.95, help="1-word threshold")
    ap.add_argument("--acc3", type=float, default=0.88, help="3-word threshold")
    ap.add_argument("--held-margin", type=float, default=0.05,
                    help="how far below the train thresholds held-out fonts may "
                         "fall and still qualify. A floor that keeps OCR failure "
                         "a minority contributor to MSG, not a second copy of the "
                         "train criterion.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", choices=["gate0", "spans"], default="gate0",
                    help="'gate0' uses this script's own held-out words and is "
                         "the selection mode. 'spans' draws from the Phase 1 "
                         "vocabulary to verify that pool is legible at the "
                         "already-frozen config — verification only; never "
                         "re-choose the config on the words it will measure")
    args = ap.parse_args()

    if args.quick:
        args.heights, args.min_pixels, args.n = [24, 48], [1024], 32

    with open(args.fonts) as fh:
        font_cfg = yaml.safe_load(fh)
    train_fonts, held_fonts = font_cfg["train"], font_cfg.get("held_out", {})
    print(f"fonts: {len(train_fonts)} train (selection), "
          f"{len(held_fonts)} held out (reported only)")

    rng = random.Random(args.seed)
    spans_1 = make_spans(args.n, 1, rng, args.pool)
    spans_3 = make_spans(args.n, 3, rng, args.pool)

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    grid = list(itertools.product(args.heights, args.min_pixels))
    print(f"sweeping {len(grid)} configs x {args.n} spans x 2 lengths\n")

    # Only the processor depends on the swept parameters, so the model loads once.
    model = AutoVLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                    attn_implementation=args.attn,
                                    device_map="cuda:0").eval()

    rows = []
    for height, min_px in grid:
        processor = AutoProcessor.from_pretrained(args.model, min_pixels=min_px)
        # Batched generation requires left padding; with right padding the
        # shorter sequences in a batch decode as garbage.
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"

        cost = visual_token_cost(processor, render("the", next(iter(
            train_fonts.values())), height))
        t0 = time.perf_counter()
        train = evaluate(model, processor, train_fonts, height,
                         spans_1, spans_3, args.batch)
        held = (evaluate(model, processor, held_fonts, height,
                         spans_1, spans_3, args.batch) if held_fonts else None)
        dt = time.perf_counter() - t0

        # Constraint is checked at the CI lower bound, so a config cannot be
        # frozen on a point estimate that merely grazes the threshold.
        passes = (train["w1"]["acc_lo"] >= args.acc1 and
                  train["w3"]["acc_lo"] >= args.acc3)
        # Held-out fonts get a floor, not the train threshold — see
        # select_winner for why the two are calibrated differently.
        held_ok = (held is None or
                   (held["w1"]["acc_lo"] >= args.acc1 - args.held_margin and
                    held["w3"]["acc_lo"] >= args.acc3 - args.held_margin))
        rows.append({"height": height, "min_pixels": min_px, "visual_tokens": cost,
                     "train": train, "held_out": held, "passes": passes,
                     "held_ok": held_ok, "seconds": round(dt, 1)})

        mark = "PASS " if passes else "     "
        if passes and not held_ok:
            mark = "train"          # clears on train fonts, fails held-out
        held_note = (f"  held {held['w1']['acc']:.2f}/{held['w3']['acc']:.2f}"
                     if held else "")
        print(f"  h={height:<3} min_px={min_px:<6} {cost:>3} tok  "
              f"1w {train['w1']['acc']:.3f} [{train['w1']['acc_lo']:.3f}]  "
              f"3w {train['w3']['acc']:.3f} [{train['w3']['acc_lo']:.3f}]  "
              f"cer {train['w1']['cer']:.3f}  {mark}{held_note}  [{dt:.0f}s]")

        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "w") as fh:
        json.dump({"args": vars(args), "rows": rows}, fh, indent=2)
    print(f"\nfull results -> {args.log}")

    win = select_winner(rows)
    if win is None:
        train_only = [r for r in rows if r["passes"] and not r["held_ok"]]
        if train_only:
            print("\nNO CONFIG CLEARS BOTH CONSTRAINTS.")
            print(f"  {len(train_only)} config(s) clear on training fonts but fall")
            print(f"  below the held-out floor (train thresholds minus "
                  f"{args.held_margin:.2f}).")
            print("  -> Widen the sweep upward (taller strips, larger min_pixels).")
            print("  -> If the best held-out miss is small, check whether the floor")
            print("     itself is right before widening: held-out fonts only need to")
            print("     be legible enough that OCR failure stays a minority")
            print("     contributor to MSG, and the metric suite already reports")
            print("     MSG conditioned on correct read-back (PLAN.md §5).")
            for r in sorted(train_only, key=lambda r: r["visual_tokens"]):
                print(f"     h={r['height']} min_px={r['min_pixels']} "
                      f"{r['visual_tokens']} tok  held-out 1w "
                      f"{r['held_out']['w1']['acc']:.3f}")
            return 1
        best = max(rows, key=lambda r: r["train"]["w1"]["acc"])
        print("\nNO CONFIG CLEARS THE THRESHOLDS.")
        print(f"  best 1-word accuracy {best['train']['w1']['acc']:.2f} at "
              f"h={best['height']}, min_pixels={best['min_pixels']}")
        if best["train"]["w1"]["acc"] >= 0.80:
            print("  -> CONDITIONAL (docs/GATES.md): one wider sweep — contrast,")
            print("     padding, anti-aliasing, larger heights — then re-test once.")
        else:
            print("  -> Below 80%. Tier B cannot lead; re-plan around Tier A, or DROP.")
        print("  Record the call in results/DECISIONS.md today, with these numbers.")
        return 1

    print(f"\nWINNER: height={win['height']}, min_pixels={win['min_pixels']}, "
          f"{win['visual_tokens']} visual tokens")
    print(f"  train    1w {win['train']['w1']['acc']:.3f} "
          f"[CI lower {win['train']['w1']['acc_lo']:.3f}]  "
          f"3w {win['train']['w3']['acc']:.3f} "
          f"[{win['train']['w3']['acc_lo']:.3f}]")
    if win["held_out"]:
        print(f"  held-out 1w {win['held_out']['w1']['acc']:.3f} "
              f"[{win['held_out']['w1']['acc_lo']:.3f}]  "
              f"3w {win['held_out']['w3']['acc']:.3f} "
              f"[{win['held_out']['w3']['acc_lo']:.3f}]")
    print("  by word class (1 word, train):", {k: round(v, 2) for k, v in
                                               win["train"]["w1"]["by_class"].items()})

    with open(args.out, "w") as fh:
        yaml.safe_dump({
            "_comment": "Frozen at Gate 0 by scripts/gate0_sweep.py. Do not hand-edit. "
                        "Changing any of this invalidates every prior measurement.",
            "height": win["height"],
            "pad": 4,
            "min_pixels": win["min_pixels"],
            "visual_tokens_per_span": win["visual_tokens"],
            "readback_train_1word": round(win["train"]["w1"]["acc"], 4),
            "readback_train_1word_ci_lower": round(win["train"]["w1"]["acc_lo"], 4),
            "readback_train_3word": round(win["train"]["w3"]["acc"], 4),
            "readback_train_3word_ci_lower": round(win["train"]["w3"]["acc_lo"], 4),
            "readback_heldout_1word": (round(win["held_out"]["w1"]["acc"], 4)
                                       if win["held_out"] else None),
            "readback_heldout_3word": (round(win["held_out"]["w3"]["acc"], 4)
                                       if win["held_out"] else None),
            "n_per_cell": args.n,
            "seed": args.seed,
        }, fh, sort_keys=False)
    print(f"\nfrozen -> {args.out}")
    print("Record the Gate 0 decision in results/DECISIONS.md today, and the")
    print("visual-token count in results/RESULTS.md — it enters every later estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
