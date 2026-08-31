"""Is the modality gap removable, and if so how cheaply?

The decisive number for the Gate 1 extension, and it was computed by the Phase 1
run and never inspected. MSG measures how far apart the two views sit at the
merge position. Offset-free MSG measures the same thing after subtracting each
modality's mean, so it asks whether the gap has any structure beyond a constant
displacement.

The distinction is the whole project:

Four nulls, each more generous than the last:

  raw            nothing removed.
  offset-free    a constant per-modality translation removed.
  rotation-free  an orthogonal map removed, fitted out-of-fold.
  linear-free    any linear change of basis removed, fitted out-of-fold.

The level at which the gap dies is the result. A gap that dies under a *linear*
map is one a standard projector already closes -- that is what a VLM's adapter
is -- and it would make the training program redundant rather than novel. A gap
that survives all four is a difference in information, which is the only version
of this project worth running.

The maps are fitted out-of-fold, without exception: fitted and scored on the
same items, enough dimensions drive any distance to zero and prove nothing.

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

# Increasingly generous accounts of what the gap could be. A gap that dies at
# one of these is a re-expression; the level at which it dies says how cheaply
# it could be undone. Linear is the one that matters most: a projector between
# modalities is exactly a linear map, so a gap a fitted linear map removes is a
# gap the standard architecture already knows how to close.
NULLS = [("msg_offset_free", "offset-free MSG"),
         ("msg_procrustes", "rotation-free MSG"),
         ("msg_linear", "linear-map-free MSG")]


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
        print(f"=== Tier {tier} ===")
        print(f"    {'raw MSG':<24} {_fmt(raw)}")

        survives = None
        blocked: list[str] = []
        for key, label in NULLS:
            # `null` is the whole per-null block; `block` is only its headline
            # numbers. An earlier version read `fit` and the denominator means
            # off `block`, where they do not live, so both guards silently
            # never fired and a clean-looking report was no evidence at all.
            null = res.get(key) or {}
            block = null.get("overall")
            if not block:
                print(f"    {label:<24} not computed — re-run phase1_measure.py")
                continue
            print(f"    {label:<24} {_fmt(block)}", end="")
            if raw["msg"] > 1.0:
                share = 1.0 - (block["msg"] - 1.0) / (raw["msg"] - 1.0)
                print(f"   ({share:.0%} of the gap)")
            else:
                print()

            fit = null.get("fit") or {}
            if fit:
                # Positive evidence, printed every time. Absence of a warning is
                # not evidence that a check ran.
                held = (f"{fit.get('n_groups')} distinct spans"
                        if fit.get("n_groups") else
                        f"{fit.get('train_n')} rows (NOT grouped)")
                print(f"    {'':<24}   fit: {fit.get('kind')}, "
                      f"{fit.get('folds')} folds, held out by {held}, "
                      f"{fit.get('rows_per_dim', 0):.1f} rows/dim, "
                      f"ridge {fit.get('ridge')}")

            warn = fit.get("underdetermined") if fit else None
            if fit and not fit.get("n_groups"):
                warn = warn or ("folds split rows rather than content — a map "
                                "fitted this way memorises repeated spans")
            if warn:
                print(f"    {'':<24} ! {warn}")
                blocked.append(label)
                continue

            # MSG below 1 says the mapped cross-modal distance is smaller than
            # the within-modality controls. That can be real -- a fitted map
            # optimises for proximity where a paraphrase does not -- but it is
            # also what a map that collapsed the image side would produce, since
            # only the image half of the denominator passes through the map.
            wt, wi = null.get("within_text_mean"), null.get("within_image_mean")
            if block["msg"] < 1.0 and wt and wi:
                ratio = wi / wt if wt else float("inf")
                note = ("IMAGE CONTROL COLLAPSED under the map — the low ratio "
                        "is the denominator, not the gap"
                        if ratio < 0.25 else
                        "both halves of the denominator survive")
                print(f"    {'':<24}   within-text {wt:.4f} / within-image "
                      f"{wi:.4f} ({ratio:.2f}x) — {note}")
                if ratio < 0.25:
                    blocked.append(label)
                    continue
            if null.get("denominator_warning"):
                print(f"    {'':<24} ! {null['denominator_warning']}")

            ci = block.get("ci")
            upper = ci[1] if ci else block["msg"]
            if upper < DROP_CI_UPPER and survives is None:
                survives = label

        print()
        if blocked and not survives:
            print(f"    -> NO VERDICT. {', '.join(blocked)} could not be fitted "
                  "with enough data to be believed. Raise --n and re-run; do "
                  "not read a surviving gap as irreducible.")
            continue
        if survives:
            print(f"    -> REMOVABLE. The gap dies once {survives.lower()} is "
                  f"taken out (upper bound < {DROP_CI_UPPER}): the two views "
                  "hold the same information in a different frame.")
        else:
            print("    -> IRREDUCIBLE at every null tested. No translation, "
                  "rotation or linear change of basis accounts for the gap, so "
                  "the views differ in information and not only in frame.")

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
