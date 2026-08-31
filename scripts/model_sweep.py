"""Run the Phase 1 measurement across several VLMs, unattended.

One model per subprocess, so a crash or an OOM costs that model and not the
sweep, and so each model's weights are freed before the next loads. Resumable:
a model whose output directory already holds `results.json` is skipped, so an
interrupted sweep continues where it stopped.

**What is comparable across models, and what is not.** This matters more than
the mechanics, because the obvious comparison is the wrong one.

*Not comparable:* raw MSG. Its denominator depends on how the model's tokeniser
splits a capitalisation variant, its numerator on how many visual tokens the
processor emits for a strip, and both on the hidden dimension. Three models can
differ in raw MSG without differing in anything we care about. The single-model
run already showed the headline moves 10x on the choice of control alone.

*Comparable:* the **share of the gap each null removes** — offset, rotation,
linear map. That is a ratio of ratios computed inside one model with one
convention throughout, so the per-model scale cancels. It is the sweep's
headline, and the question it answers is the one worth asking: *is the modality
gap orientation in every decoder VLM, or only in this one?*

**Two per-model preconditions the sweep checks and reports rather than assumes.**

1. *Read-back.* A model that cannot read the rendered strip produces a confident
   MSG about an unreadable stimulus. Its row is marked invalid.
2. *Distinct spans against hidden size.* A `[D, D]` map needs roughly `2D`
   distinct spans. Our pool holds 8000, which covers `D <= 4000` and does **not**
   cover a 4096-wide model. Those rows report the map nulls as untestable
   instead of printing a number.

    python scripts/model_sweep.py --preflight          # check ids resolve
    python scripts/model_sweep.py --n 16000            # the sweep itself
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freeflow.metrics import msg  # noqa: E402

DEFAULT_LIST = ROOT / "configs" / "sweep_models.txt"


def read_models(path: Path, override: list[str] | None) -> list[str]:
    if override:
        return override
    if not path.exists():
        raise SystemExit(f"no model list at {path}; pass --models explicitly")
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def preflight(models: list[str]) -> int:
    """Resolve every id before spending GPU-hours on the first one.

    Worth its own mode: the sweep is meant to run unattended, and discovering a
    typo in the sixth model id after five hours is the avoidable version of that.
    """
    from transformers import AutoConfig
    bad = []
    for m in models:
        try:
            cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
            dim = (getattr(cfg, "hidden_size", None)
                   or getattr(getattr(cfg, "text_config", None), "hidden_size", None))
            need = 2 * dim if dim else None
            note = ""
            if need:
                note = (f"hidden {dim}, map nulls need ~{need} distinct spans"
                        f"{'  ! pool holds 8000' if need > 8000 else ''}")
            print(f"  OK   {m:<44} {note}")
        except Exception as exc:                       # noqa: BLE001
            bad.append(m)
            print(f"  FAIL {m:<44} {type(exc).__name__}: {exc}")
    if bad:
        print(f"\n{len(bad)} of {len(models)} ids did not resolve. Fix the list "
              "before running the sweep.")
    return 1 if bad else 0


def run_one(model: str, out_dir: Path, args) -> tuple[str, float]:
    """Measure one model in its own process. Returns (status, seconds)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "phase1_measure.py"),
           "--model", model, "--tiers", *args.tiers,
           "--n", str(args.n), "--batch", str(args.batch),
           "--functional-n", str(args.functional_n),
           "--max-distinct-spans", "--out", str(out_dir)]
    if args.device:
        cmd += ["--device", args.device]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT)
    return ("ok" if proc.returncode == 0 else f"failed ({proc.returncode})",
            time.perf_counter() - t0)


def _reduction(raw_cross: float | None, cross: float | None) -> float | None:
    """Fraction by which a null shrinks the cross-modal distance.

    Replaces the earlier share-of-MSG measure, which divided by a denominator
    whose definition moves the answer ~10x: on Tier B the same null "removes"
    69% or 132% of the gap depending on whether the within-modality control is a
    font change or a capitalisation flip. The cross distance is the same
    measurement under both, so this is comparable across models *and* across
    control designs.
    """
    if not raw_cross or cross is None:
        return None
    return 1.0 - cross / raw_cross


def summarise(rows: list[dict]) -> str:
    """The sweep's headline: share of the gap each null removes, per model."""
    out = ["# Multi-model sweep — is the modality gap orientation everywhere?",
           "",
           "**Percentage by which each null shrinks the cross-modal distance at "
           "the merge position.**",
           "Reported this way rather than as a share of MSG because MSG's "
           "denominator is a choice: on Tier B a",
           "capitalisation-flip control and a font-change control disagree by "
           "7–9x, which moves the same null",
           "between \"removes 69%\" and \"removes 132%\". The cross distance is "
           "the same measurement under either,",
           "so these columns are comparable across models *and* across control "
           "designs.",
           "",
           "A **negative** figure means that null *increases* the distance. "
           "That is expected for the offset",
           "column: subtracting each modality's mean removes a large shared "
           "component that was making every",
           "state look alike, so the cosine distance grows.",
           "",
           "Raw MSG and the raw cross distance are listed for reference. "
           "Neither is comparable across rows —",
           "both depend on the tokeniser, the processor's visual-token count "
           "and the hidden size.",
           "",
           "| model | dim | read-back | raw MSG | raw cross | offset | isometry "
           "| linear | valid |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            out.append(f"| `{r['model']}` | — | — | — | — | — | — | — | "
                       f"**{r['error']}** |")
            continue
        f = lambda x: "—" if x is None else f"{x:.0%}"      # noqa: E731
        rb, rc = r.get("readback"), r.get("raw_cross")
        out.append(
            f"| `{r['model']}` | {r.get('dim', '—')} | "
            f"{'—' if rb is None else f'{rb:.3f}'} | {r['raw']:.2f} | "
            f"{'—' if rc is None else f'{rc:.4f}'} | "
            f"{f(r['offset'])} | {f(r['isometry'])} | {f(r['linear'])} | "
            f"{r['valid']} |")
    out += ["", "A row is invalid when the model cannot read the rendered span, "
            "or when the map nulls could not be",
            "or its forced choice failed. A **—** in one null's column means "
            "that null alone could not",
            "conclude — the fit collapsed or had too little distinct content — "
            "and the other columns in that",
            "row still stand.",
            "",
            "Reference, Qwen3.5-2B at 8000 distinct spans: an isometry cuts the "
            "cross distance 63–64% and a",
            "linear map 75–76%, measured under two different text controls."]
    return "\n".join(out)


def collect(model: str, out_dir: Path) -> dict:
    path = out_dir / "results.json"
    if not path.exists():
        return {"model": model, "error": "no results.json"}
    doc = json.loads(path.read_text())
    res = (doc.get("results") or {}).get("B")
    if not res or "msg_raw" not in res:
        return {"model": model, "error": "no Tier B geometry"}

    raw_cross = msg.cross_from_record(res["msg_raw"])
    row = {"model": model, "raw": res["msg_raw"]["overall"]["msg"],
           "raw_cross": raw_cross, "valid": "yes"}
    for key, name in (("msg_offset_free", "offset"),
                      ("msg_procrustes", "isometry"), ("msg_linear", "linear")):
        block = res.get(key) or {}
        fit = block.get("fit") or {}
        # A null that could not be fitted reports nothing rather than a number.
        if fit.get("underdetermined") or fit.get("collapsed") or not fit.get("n_groups", 1):
            row[name] = None
        else:
            row[name] = _reduction(raw_cross, msg.cross_from_record(block))
        if fit.get("dim"):
            row["dim"] = fit["dim"]
    rb = (res.get("readback") or {})
    if rb.get("applicable"):
        row["readback"] = rb["accuracy"]
        if rb["accuracy"] < 0.95:
            row["valid"] = "read-back below 0.95"
    if (res.get("functional") or {}).get("validity_warning"):
        row["valid"] = "forced choice invalid"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--model-list", default=str(DEFAULT_LIST))
    ap.add_argument("--tiers", nargs="+", default=["B"])
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--functional-n", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/sweep")
    ap.add_argument("--preflight", action="store_true",
                    help="resolve every model id and report the distinct-span "
                         "requirement each hidden size implies, then stop")
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild the summary from results already on disk "
                         "without measuring anything. Use this after a sweep "
                         "that ran under older analysis code — the numbers come "
                         "from each run's results.json, so the summary can be "
                         "regenerated without re-running any model")
    ap.add_argument("--force", action="store_true",
                    help="re-measure models that already have results")
    args = ap.parse_args()

    models = read_models(Path(args.model_list), args.models)
    print(f"{len(models)} models\n")
    if args.preflight:
        return preflight(models)

    root = ROOT / args.out
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, model in enumerate(models, 1):
        out_dir = root / model.replace("/", "__")
        if args.summarise_only:
            pass
        elif (out_dir / "results.json").exists() and not args.force:
            print(f"[{i}/{len(models)}] {model}: already measured, skipping")
        else:
            print(f"\n[{i}/{len(models)}] {model}")
            status, secs = run_one(model, out_dir, args)
            print(f"    {status} in {secs / 60:.1f} min")
        rows.append(collect(model, out_dir))
        # Written after every model, so an interrupted sweep still leaves a
        # readable summary of what completed.
        (root / "summary.json").write_text(json.dumps(rows, indent=2))
        (root / "summary.md").write_text(summarise(rows))

    print("\n" + summarise(rows))
    print(f"\nwrote {root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
