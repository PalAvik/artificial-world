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


def _cross(null: dict) -> float | None:
    """Reconstruct the mean cross-modal distance from MSG and its denominator.

    Older result files predate `cross_mean` being recorded directly; the ratio
    and both halves of the denominator are enough to recover it exactly.
    """
    wt, wi = null.get("within_text_mean"), null.get("within_image_mean")
    overall = null.get("overall") or {}
    if not (wt and wi and overall.get("msg")):
        return None
    return overall["msg"] * 0.5 * (wt + wi)


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

        # Validity first. Every number below is contingent on these, and this
        # reporter previously printed the geometry without them -- which is how
        # an 8000-span run was read for its nulls while its read-back and
        # forced-choice numbers sat unexamined in the same file.
        rb = res.get("readback") or {}
        if rb.get("applicable"):
            ok = rb["accuracy"] >= 0.95
            print(f"    {'read-back':<24} {rb['accuracy']:.3f} "
                  f"(CER {rb.get('cer', float('nan')):.4f})"
                  f"{'' if ok else '   ! below 0.95 — misread spans contaminate MSG'}")
        elif rb:
            print(f"    {'read-back':<24} not applicable to tier {tier}")
        fc = res.get("functional") or {}
        if fc:
            floor = fc.get("ablated_accuracy")
            floor_s = f"{floor:.3f}" if floor is not None else "not measured"
            print(f"    {'forced choice':<24} text {fc['text_accuracy']:.3f} / "
                  f"image {fc['image_accuracy']:.3f}, span-free floor {floor_s}")
            if fc.get("validity_warning"):
                print(f"    {'':<24} ! {fc['validity_warning']}")
                print(f"\n    -> TIER INVALID. The geometry below is not "
                      f"interpretable.\n")
                continue
            if fc.get("delta_warning"):
                print(f"    {'':<24} ~ {fc['delta_warning']}")
        elif "msg_raw" in res:
            print(f"    {'validity':<24} ! no read-back or forced-choice record "
                  "in this file — the geometry below is unvalidated")

        print(f"    {'raw MSG':<24} {_fmt(raw)}")

        raw_cross = (res["msg_raw"].get("cross_mean")
                     or _cross(res["msg_raw"]))
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
            cross = null.get("cross_mean") or _cross(null)
            print(f"    {label:<24} {_fmt(block)}", end="")
            if raw["msg"] > 1.0:
                share = 1.0 - (block["msg"] - 1.0) / (raw["msg"] - 1.0)
                print(f"   ({share:.0%} of the gap)")
            else:
                print()

            fit = null.get("fit") or {}
            if cross and raw_cross:
                # Convention-free. MSG's denominator moves the headline ~10x
                # depending on which within-modality control is chosen, but the
                # numerator is the same measurement under every choice, so the
                # reduction it shows is the one number that does not need the
                # convention stated alongside it.
                # Direction stated in words: removing the per-modality mean
                # *increases* the cosine distance (the shared component was
                # making everything look alike), so a signed percentage here
                # reads backwards half the time.
                delta = cross / raw_cross - 1.0
                way = "below" if delta < 0 else "above"
                print(f"    {'':<24}   cross distance {cross:.4f} "
                      f"({abs(delta):.0%} {way} raw) — independent of the "
                      "control")
            if fit:
                # Positive evidence, printed every time. Absence of a warning is
                # not evidence that a check ran.
                held = (f"{fit.get('n_groups')} distinct spans"
                        if fit.get("n_groups") else
                        f"{fit.get('train_n')} rows (NOT grouped)")
                print(f"    {'':<24}   fit: {fit.get('kind')}, "
                      f"{fit.get('folds')} folds, held out by {held}, "
                      f"{fit.get('rows_per_dim', 0):.1f} rows/dim, "
                      f"ridge {fit.get('ridge')}, "
                      f"control retained {fit.get('control_retention', 1.0):.0%}")

            warn = (fit.get("underdetermined") or fit.get("collapsed")) \
                if fit else None
            if fit and not fit.get("n_groups"):
                warn = warn or ("folds split rows rather than content — a map "
                                "fitted this way memorises repeated spans")
            if warn:
                print(f"    {'':<24} ! {warn}")
                blocked.append(label)
                continue

            # The two halves of the denominator, printed for every null. These
            # are NOT interchangeable and comparing them is not a collapse test:
            # the text control is a capitalisation flip, which re-tokenises the
            # span, and the image control is a font change, which does not. They
            # differ ~9x on Tier B *before* any map is fitted. Collapse is
            # mapped-vs-unmapped image control, which is `control_retention`
            # above; conflating the two produced a false NO VERDICT here.
            wt, wi = null.get("within_text_mean"), null.get("within_image_mean")
            if wt and wi:
                # MSG averages these two, and on Tier B they differ by ~9x: a
                # capitalisation flip re-tokenises the span, a font change does
                # not. An arithmetic mean of two such distances is dominated by
                # the larger, so the headline MSG is an artefact of the
                # convention as much as of the model. Report the ratio against
                # each control so the reader can see the bracket.
                cross = block["msg"] * 0.5 * (wt + wi)
                print(f"    {'':<24}   denominator halves: within-text {wt:.4f}"
                      f" / within-image {wi:.4f} ({wt / wi:.1f}x)")
                print(f"    {'':<24}   MSG vs text control {cross / wt:.2f}"
                      f" · vs image control {cross / wi:.2f}"
                      f" · geometric mean {cross / (wt * wi) ** 0.5:.2f}")
            if null.get("denominator_warning"):
                print(f"    {'':<24} ~ {null['denominator_warning']}")

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
