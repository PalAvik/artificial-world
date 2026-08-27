"""Is the modality gap a removable translation?

The decisive number for the Gate 1 extension, and it was computed by the Phase 1
run and never inspected. MSG measures how far apart the two views sit at the
merge position. Offset-free MSG measures the same thing after subtracting each
modality's mean, so it asks whether the gap has any structure beyond a constant
displacement.

The distinction is the whole project:

  offset-free MSG collapses toward 1  ->  the gap is a translation. Whatever
      Phase 2 would learn, a fixed vector already does. The honest output is a
      short note, not a training program.

  offset-free MSG stays well above 1  ->  the gap is structural, the views
      differ in more than position, and there is something to train against.

Reads results/phase1/results.json. Prints, and does not decide anything on its
own — the drop rule it feeds is written down in results/DECISIONS.md.

    python scripts/show_offset.py [--results results/phase1/results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DROP_CI_UPPER = 1.25          # docs/GATES.md, Gate 1 DROP row


def _fmt(block: dict) -> str:
    ci = block.get("ci")
    ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "no CI"
    return f"{block['msg']:.3f}  95% CI {ci_s}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/phase1/results.json")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"no results at {path} — run scripts/phase1_measure.py first")
        return 1
    doc = json.loads(path.read_text())
    # The driver writes {"args": ..., "results": {tier: ...}}; tolerate a bare
    # {tier: ...} too, so an older file still reads.
    tiers = doc.get("results", doc)

    print(f"{path}\n")
    for tier, res in tiers.items():
        if "msg_raw" not in res:
            continue
        raw = res["msg_raw"]["overall"]
        off = res["msg_offset_free"]["overall"]
        print(f"=== Tier {tier} ===")
        print(f"    raw MSG          {_fmt(raw)}")
        print(f"    offset-free MSG  {_fmt(off)}")

        share = 1.0 - (off["msg"] - 1.0) / (raw["msg"] - 1.0) \
            if raw["msg"] > 1.0 else float("nan")
        print(f"    share of the gap explained by a constant offset: {share:.1%}")

        off_ci = off.get("ci")
        upper = off_ci[1] if off_ci else off["msg"]
        if upper < DROP_CI_UPPER:
            print(f"    -> TRANSLATION. Offset-free upper bound {upper:.3f} < "
                  f"{DROP_CI_UPPER}: nothing survives removing the mean.")
        else:
            print(f"    -> STRUCTURAL. Offset-free upper bound {upper:.3f} >= "
                  f"{DROP_CI_UPPER}: the gap is more than a displacement.")

        # Where it lives. The offset norm rising with depth while the offset-free
        # distance stays flat is the signature of a pure translation.
        per_layer = res.get("per_layer") or []
        if per_layer:
            print(f"\n    {'layer':<10} {'cross raw':>10} {'offset-free':>12} "
                  f"{'offset norm':>12} {'CKA':>7}")
            for r in per_layer:
                print(f"    {str(r['layer']):<10} {r['cross_raw']:>10.4f} "
                      f"{r['cross_offset_free']:>12.4f} "
                      f"{r['offset_norm']:>12.3f} {r['cka']:>7.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
