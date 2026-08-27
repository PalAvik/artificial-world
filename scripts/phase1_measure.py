#!/usr/bin/env python3
"""Phase 1: measure the modality substitution gap. The Gate 1 study.

Wires corpus -> runner -> metrics -> report, and prints the Gate 1 branch the
numbers imply. That branch is computed mechanically from thresholds fixed in
docs/GATES.md before any of these numbers existed, which is the whole point of
writing them down in advance.

What it reports, and why each piece is there:

  normalized MSG      the headline (PLAN.md §5.3). Reported raw *and*
                      offset-free: if most of the gap is a constant
                      per-modality translation, a linear readout could undo it
                      and the problem is far more tractable than the raw number
                      suggests. That difference is H3.
  by word class       H2. Read with the read-back conditioning below, since
                      Gate 0 showed read-back tracks word length.
  conditioned MSG     PLAN.md §5.3a. Every Tier B number appears twice, over
                      all spans and over correctly-read spans, with the rate.
  JSD                 distributional agreement over the shared continuation.
  probe               the Phase 2 baseline. Meaningless alone; it is the
                      denominator of the collapse check at Gate 2.
  per-layer geometry  where in the stack the gap lives, and the offset norm.

Usage:
    python scripts/phase1_measure.py --model Qwen/Qwen3.5-2B --n 2000
    python scripts/phase1_measure.py --model Qwen/Qwen3.5-2B --n 128 --quick
"""
from __future__ import annotations

import sys
from pathlib import Path

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root, so
# `freeflow` would not import. Works whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from freeflow.data import tier_b, tier_c, vocab
from freeflow.data.render import FontSet, RenderConfig
from freeflow.metrics import (aggregate, cycle, functional, geometry, msg,
                              probe, runner)

ROOT = Path(__file__).resolve().parents[1]


def load_model(path: str, attn: str, device: str):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    cfg = RenderConfig.load()
    processor = AutoProcessor.from_pretrained(path, min_pixels=cfg.min_pixels)
    model = AutoVLM.from_pretrained(path, dtype=torch.bfloat16,
                                    attn_implementation=attn,
                                    device_map=device).eval()
    return model, processor


def build_corpus(tier: str, n: int, fonts: FontSet, cfg: RenderConfig,
                 seed: int, held_out_fonts: bool):
    if tier == "B":
        return tier_b.build(n, fonts, cfg, seed=seed, held_out_fonts=held_out_fonts)
    if tier == "C":
        return tier_c.build(n, seed=seed)
    raise ValueError(
        f"tier {tier!r} not wired here. Tier A needs Flickr30k or Visual Genome "
        "on disk; load it with freeflow.data.tier_a and pass the items in.")


def distances(cap: dict, layer_key: str, offset_free: bool) -> dict:
    """Cross-modal and within-modality distances at one captured layer."""
    h_t = cap["text"].hidden[layer_key]
    h_i = cap["image"].hidden[layer_key]
    h_tc = cap["text_control"].hidden[layer_key]
    h_ic = cap["image_control"].hidden[layer_key]
    d = geometry.offset_free_distance if offset_free else geometry.cosine_distance
    return {"cross": d(h_t, h_i),
            "within_text": d(h_t, h_tc),
            "within_image": d(h_i, h_ic)}


def measure_tier(model, processor, items, batch: int, layers, device: str,
                 functional_n: int = 256, seed: int = 0) -> dict:
    """Validity check, capture, then every metric for one tier."""
    t0 = time.perf_counter()
    tier = items[0].tier

    # Read-back only where the image *contains text*. Asking a relation diagram
    # to be transcribed is a category error, and the first Phase 1 run duly
    # reported 0.000 on Tier C. Other tiers get their validity from the
    # forced-choice check below instead.
    if tier == "B":
        rb = cycle.mark_read_ok(model, processor, items, batch)
        print(f"    {rb}")
        readback = {"accuracy": rb.accuracy, "cer": rb.cer, "applicable": True}
    else:
        for it in items:
            it.read_ok = True
        readback = {"accuracy": None, "cer": None, "applicable": False,
                    "note": "read-back is text-transcription; not defined for "
                            f"tier {tier} images"}
        print(f"    read-back not applicable to tier {tier}")

    # Forced choice: the validity check for every tier, and Gate 1's functional
    # delta. Capped because it runs one item at a time.
    subset = items[:functional_n]
    fc = functional.forced_choice(model, processor, subset, seed=seed,
                                  device=device)
    print(f"    forced choice: {fc}")
    warning = fc.validity()
    if warning:
        print(f"    ! {warning}")

    cap = runner.capture(model, processor, items, batch=batch, layers=layers,
                         device=device)
    captured = cap["text"].hidden["layers"]
    final = str(len(captured) - 1)
    groups = cap["text"].groups
    read_ok = torch.tensor(cap["text"].read_ok)

    out: dict = {
        "n": len(items),
        "tier": tier,
        "readback": readback,
        "functional": {
            "text_accuracy": fc.text.accuracy,
            "image_accuracy": fc.image.accuracy,
            "delta": fc.delta,
            "ablated_accuracy": fc.ablated.accuracy if fc.ablated else None,
            "image_above_floor": fc.image_above_floor,
            "n": fc.text.n,
            "chance": fc.image.chance,
            "validity_warning": warning,
        },
        "layers_captured": captured,
        "jsd": cap["text"].jsd.summary(),
        "seconds": 0.0,
    }

    # Headline MSG, raw and offset-free, at the final layer.
    for name, off in (("msg_raw", False), ("msg_offset_free", True)):
        d = distances(cap, final, off)
        report = aggregate.conditioned_msg(
            d["cross"], d["within_text"], d["within_image"], groups, read_ok)
        out[name] = {
            "overall": _msg_dict(report.unconditional.overall),
            "by_group": {k: _msg_dict(v)
                         for k, v in report.unconditional.groups.items()},
            "conditioned": (_msg_dict(report.read_correctly.overall)
                            if report.read_correctly else None),
            "conditioned_by_group": (
                {k: _msg_dict(v) for k, v in report.read_correctly.groups.items()}
                if report.read_correctly else None),
            "readback_rate": report.readback_rate,
            # The two halves of the denominator, separately: if one dominates,
            # the normalisation is really being set by that control alone.
            "within_text_mean": float(d["within_text"].mean()),
            "within_image_mean": float(d["within_image"].mean()),
            "denominator_warning": _denominator_warning(d),
        }
        if out[name]["denominator_warning"]:
            print(f"    ! {name}: {out[name]['denominator_warning']}")

    # Where in the stack the gap lives, and how much of it is a translation.
    per_layer = []
    for i, layer in enumerate(captured):
        d_raw = distances(cap, str(i), False)
        d_off = distances(cap, str(i), True)
        stats = geometry.offset_stats(cap["text"].hidden[str(i)],
                                      cap["image"].hidden[str(i)])
        per_layer.append({
            "layer": layer,
            "cross_raw": float(d_raw["cross"].mean()),
            "cross_offset_free": float(d_off["cross"].mean()),
            "offset_norm": stats.norm,
            "cka": geometry.linear_cka(cap["text"].hidden[str(i)],
                                       cap["image"].hidden[str(i)]),
        })
    out["per_layer"] = per_layer

    # Probe baseline, on each modality's own representation.
    for key in ("text", "image"):
        try:
            out[f"probe_{key}"] = asdict(
                probe.fit_probe(cap[key].hidden[final], cap[key].span_ids))
        except ValueError as exc:
            out[f"probe_{key}"] = {"error": str(exc)}

    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def _denominator_warning(d: dict, ratio: float = 20.0) -> str | None:
    """Flag a within-modality control that contributed almost nothing.

    MSG divides by the *average* of the two controls, so a half that collapses
    toward zero silently doubles the reported gap. This cannot raise — a small
    control distance may be a real finding — but it must never pass unnoticed.
    """
    t, i = float(d["within_text"].mean()), float(d["within_image"].mean())
    if t <= 0 or i <= 0:
        which = "text" if t <= 0 else "image"
        return (f"the {which} control distance is zero; the denominator has "
                "lost half its mass and MSG is inflated by ~2x")
    if max(t, i) / min(t, i) > ratio:
        big, small = ("text", "image") if t > i else ("image", "text")
        return (f"{big} control is {max(t, i) / min(t, i):.0f}x the {small} "
                f"control ({t:.4f} vs {i:.4f}); the normalisation is effectively "
                f"set by the {big} control alone")
    return None


def _tier_header(res: dict) -> str:
    rb = res["readback"]
    parts = [f"n={res['n']}"]
    if rb.get("applicable"):
        parts.append(f"read-back {rb['accuracy']:.3f} (CER {rb['cer']:.4f})")
    else:
        parts.append("read-back n/a")
    f = res["functional"]
    fc = (f"forced choice: text {f['text_accuracy']:.3f} / "
          f"image {f['image_accuracy']:.3f}, delta {f['delta']:+.3f}")
    if f.get("ablated_accuracy") is not None:
        fc += f", span-free floor {f['ablated_accuracy']:.3f}"
    parts.append(fc)
    parts.append(f"{res['seconds']}s")
    line = " · ".join(parts)
    if f.get("validity_warning"):
        line += f"\n\n> **Validity:** {f['validity_warning']}."
    return line


def _msg_dict(r: msg.MSGResult) -> dict:
    return {"msg": r.ratio_of_means, "mean_of_ratios": r.mean_of_ratios,
            "ci": list(r.ci) if r.ci else None, "n": r.n,
            "cross_mean": r.numerator_mean, "within_mean": r.denominator_mean}


def render_report(results: dict, args) -> str:
    """The human-readable half. The JSON is the record; this is what gets read."""
    L = ["# Phase 1 — modality substitution gap", "",
         f"model `{args.model}` · {args.n} items/tier · batch {args.batch} · "
         f"attn {args.attn}", ""]

    L += ["## Gate 1", "",
          "Thresholds from `docs/GATES.md`, fixed before these numbers existed.", ""]
    for tier, res in results.items():
        verdict = res.get("gate1", "—")
        L.append(f"- **Tier {tier}** — {verdict}")
    L += ["", "Gate 1 passes on MSG >= 1.5 with CI lower bound > 1.25 on at least "
          "two of three tiers, together with a functional accuracy delta >= 5 pts "
          "measured separately.", ""]

    for tier, res in results.items():
        raw, off = res["msg_raw"], res["msg_offset_free"]
        L += [f"## Tier {tier}", "",
              _tier_header(res), "",
              "| quantity | MSG | 95% CI | cross | within |",
              "|---|---|---|---|---|"]
        for label, block in (("raw", raw), ("offset-free", off)):
            o = block["overall"]
            ci = f"[{o['ci'][0]:.3f}, {o['ci'][1]:.3f}]" if o["ci"] else "—"
            L.append(f"| {label} | **{o['msg']:.3f}** | {ci} | "
                     f"{o['cross_mean']:.4f} | {o['within_mean']:.4f} |")
        if raw["conditioned"]:
            c = raw["conditioned"]
            ci = f"[{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]" if c["ci"] else "—"
            L.append(f"| raw, read correctly | **{c['msg']:.3f}** | {ci} | "
                     f"{c['cross_mean']:.4f} | {c['within_mean']:.4f} |")
        L.append("")

        # H3 is about the cross-modal *distance*, not the ratio, and removing a
        # shared mean can move a cosine distance either way — so report the two
        # distances and the offset norm rather than a percentage that can go
        # negative and read as nonsense.
        c_raw = raw["overall"]["cross_mean"]
        c_off = off["overall"]["cross_mean"]
        shrink = c_off / c_raw if c_raw else float("nan")
        h3 = ("most of the gap is a translation a linear readout could undo"
              if shrink < 0.5 else
              "the gap is not explained by a per-modality translation")
        L += [f"**H3:** cross-modal distance {c_raw:.4f} -> {c_off:.4f} with the "
              f"per-modality mean removed ({shrink:.2f}x) — {h3}.", "",
              f"Denominator halves: text control {raw['within_text_mean']:.4f}, "
              f"image control {raw['within_image_mean']:.4f}.", ""]
        if raw.get("denominator_warning"):
            L += [f"> **Warning:** {raw['denominator_warning']}.", ""]

        if raw["by_group"]:
            L += ["| group | MSG (all) | MSG (read correctly) | n |", "|---|---|---|---|"]
            cond = raw["conditioned_by_group"] or {}
            for g, v in sorted(raw["by_group"].items()):
                c = cond.get(g)
                L.append(f"| {g} | {v['msg']:.3f} | "
                         f"{c['msg']:.3f} | {v['n']} |" if c else
                         f"| {g} | {v['msg']:.3f} | — | {v['n']} |")
            L.append("")

        j = res["jsd"]
        if j.get("n"):
            L += [f"JSD over the shared continuation: mean {j['mean']:.4f} bits, "
                  f"median {j['median']:.4f}, p90 {j['p90']:.4f}, "
                  f"p99 {j['p99']:.4f}.", ""]

        for key in ("text", "image"):
            p = res.get(f"probe_{key}", {})
            if "accuracy" in p:
                note = f" **{p['warning']}**" if p.get("warning") else ""
                L.append(f"Probe ({key}): {p['accuracy']:.3f} vs chance "
                         f"{p['chance']:.3f}, {p['n_classes']} classes, "
                         f"n_test={p['n_test']}. Phase 2 baseline.{note}")
        L += ["", "| layer | cross raw | cross offset-free | offset norm | CKA |",
              "|---|---|---|---|---|"]
        for r in res["per_layer"]:
            L.append(f"| {r['layer']} | {r['cross_raw']:.4f} | "
                     f"{r['cross_offset_free']:.4f} | {r['offset_norm']:.3f} | "
                     f"{r['cka']:.3f} |")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tiers", nargs="+", default=["B", "C"])
    ap.add_argument("--n", type=int, default=2000,
                    help="items per tier. Gate 1 specifies >=2000, and the probe "
                         "needs roughly 10x the span vocabulary (~150 spans) to "
                         "have several items per class — below that its test "
                         "split degenerates and it reports 1.0 against chance 1.0")
    ap.add_argument("--batch", type=int, default=32,
                    help="32 sits past the throughput knee (docs/COMPUTE.md)")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--functional-n", type=int, default=256,
                    help="items for the forced-choice check. Runs one item at a "
                         "time, so it is the slow path; 256 gives a +/-6pt CI on "
                         "the accuracy delta, enough for Gate 1's 5pt threshold")
    ap.add_argument("--held-out-fonts", action="store_true",
                    help="Gate 2 evaluation set. Not for Gate 1.")
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="n=128, smoke run only")
    args = ap.parse_args()

    if args.quick:
        args.n = 128

    cfg, fonts = RenderConfig.load(), FontSet.load()
    print(f"render config: height {cfg.height}, min_pixels {cfg.min_pixels}, "
          f"{cfg.visual_tokens_per_span} visual tokens/span")
    print(f"fonts: {len(fonts.train)} train, {len(fonts.held_out)} held out")
    print(f"word lengths by class: "
          f"{ {k: round(v, 1) for k, v in vocab.length_summary().items()} }")

    print(f"\nloading {args.model} ...")
    model, processor = load_model(args.model, args.attn, args.device)

    results: dict[str, dict] = {}
    for tier in args.tiers:
        print(f"\n=== Tier {tier} ===")
        items = build_corpus(tier, args.n, fonts, cfg, args.seed,
                             args.held_out_fonts)
        print(f"    {len(items)} items, {len({i.span_id for i in items})} unique spans")
        res = measure_tier(model, processor, items, args.batch, args.layers,
                           args.device, args.functional_n, args.seed)
        r = msg.MSGResult(**{
            "ratio_of_means": res["msg_raw"]["overall"]["msg"],
            "mean_of_ratios": res["msg_raw"]["overall"]["mean_of_ratios"],
            "numerator_mean": res["msg_raw"]["overall"]["cross_mean"],
            "denominator_mean": res["msg_raw"]["overall"]["within_mean"],
            "n": res["msg_raw"]["overall"]["n"],
            "ci": tuple(res["msg_raw"]["overall"]["ci"])
            if res["msg_raw"]["overall"]["ci"] else None})
        verdict = msg.gate1_verdict(r)
        f = res["functional"]
        if f["validity_warning"]:
            verdict = (f"INVALID: {f['validity_warning']}. MSG {r.ratio_of_means:.2f} "
                       "is not interpretable for this tier.")
        elif abs(f["delta"]) < 0.05 and verdict.startswith("PASS"):
            verdict = (f"MSG PASS but functional delta {f['delta']:+.3f} < 0.05 — "
                       "Gate 1 needs both (docs/GATES.md)")
        res["gate1"] = verdict
        results[tier] = res
        print(f"    MSG {r}")
        print(f"    {res['gate1']}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"args": vars(args), "results": results}, indent=2))
    report = render_report(results, args)
    (out_dir / "report.md").write_text(report)

    print(f"\nwrote {out_dir/'results.json'} and {out_dir/'report.md'}")
    print("\nRecord the Gate 1 decision in results/DECISIONS.md today, with these")
    print("numbers and the call — including if the call is to drop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
