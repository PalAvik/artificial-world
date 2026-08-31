"""The reporter's guards must actually fire.

Written because they did not. `fit` and the denominator means live on the
per-null block, and the reporter read them off that block's `overall` sub-dict,
so the leakage and under-determination checks were dead code -- a report with no
warnings was no evidence that anything had been checked. These tests drive the
script over synthetic results and assert on what it prints, which is the only
level at which a silently-inert guard is visible.

    python -m pytest tests/test_show_offset.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _null(msg, lo, hi, *, fit=None, wt=None, wi=None):
    block = {"overall": {"msg": msg, "mean_of_ratios": msg, "ci": [lo, hi],
                         "n": 8000}}
    if fit is not None:
        block["fit"] = fit
    if wt is not None:
        block["within_text_mean"] = wt
        block["within_image_mean"] = wi
    return block


def _fit(**kw):
    base = {"kind": "linear", "folds": 5, "train_n": 6400, "dim": 2048,
            "ridge": 0.01, "rows_per_dim": 3.1, "n_groups": 8000,
            "underdetermined": None}
    base.update(kw)
    return base


def _run(doc, tmp_path) -> str:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc))
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "show_offset.py"),
         "--results", str(path)], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _doc(**nulls):
    res = {"msg_raw": _null(3.485, 3.460, 3.510)}
    res.update(nulls)
    return {"args": {}, "results": {"B": res}}


def test_the_fit_is_always_described(tmp_path):
    """Positive evidence, not absence of an alarm. The reason the earlier bug
    survived is that a clean report and a dead guard look identical."""
    out = _run(_doc(msg_linear=_null(0.858, 0.851, 0.865, fit=_fit(),
                                     wt=0.049, wi=0.041)), tmp_path)
    assert "8000 distinct spans" in out
    assert "linear, 5 folds" in out
    assert "ridge 0.01" in out


def test_a_rowwise_fit_is_refused(tmp_path):
    out = _run(_doc(msg_linear=_null(0.006, 0.006, 0.007,
                                     fit=_fit(n_groups=0))), tmp_path)
    assert "folds split rows rather than content" in out
    assert "NO VERDICT" in out
    assert "REMOVABLE" not in out


def test_an_underdetermined_fit_is_refused(tmp_path):
    out = _run(_doc(msg_linear=_null(2.5, 2.4, 2.6, fit=_fit(
        n_groups=150, underdetermined="150 distinct spans for 2048 dimensions"))),
        tmp_path)
    assert "150 distinct spans for 2048" in out
    assert "NO VERDICT" in out


def test_a_collapsed_map_is_refused(tmp_path):
    """Collapse is mapped-vs-unmapped image control, reported by the fit."""
    out = _run(_doc(msg_linear=_null(0.20, 0.19, 0.21, wt=0.049, wi=0.004,
        fit=_fit(control_retention=0.08,
                 collapsed="the map retains 8% of the within-modality control"))),
        tmp_path)
    assert "retains 8%" in out
    assert "NO VERDICT" in out and "REMOVABLE" not in out


def test_asymmetric_denominator_halves_are_not_a_collapse(tmp_path):
    """The false NO VERDICT of 2026-08-31. A capitalisation flip re-tokenises
    the span and a font change does not, so the two halves differ ~9x before
    any map exists. That is a caveat on the ratio, not a broken fit."""
    out = _run(_doc(msg_linear=_null(0.858, 0.851, 0.865, wt=0.0817, wi=0.0069,
                                     fit=_fit(control_retention=0.77))), tmp_path)
    assert "denominator halves" in out and "11.8x" in out
    assert "NO VERDICT" not in out
    assert "REMOVABLE" in out


def test_a_sound_sub_one_result_is_reported_as_removable(tmp_path):
    out = _run(_doc(msg_linear=_null(0.858, 0.851, 0.865, fit=_fit(),
                                     wt=0.049, wi=0.041)), tmp_path)
    assert "denominator halves" in out
    assert "control retained 100%" in out
    assert "REMOVABLE" in out


def test_a_surviving_gap_is_reported_as_irreducible(tmp_path):
    out = _run(_doc(msg_linear=_null(2.9, 2.85, 2.95, fit=_fit())), tmp_path)
    assert "IRREDUCIBLE" in out


def test_validity_is_printed_before_the_geometry(tmp_path):
    """The geometry is contingent on these, and reading it without them is how
    an 8000-span run was interpreted while its own validity numbers sat
    unexamined in the same file."""
    doc = _doc(msg_linear=_null(0.858, 0.851, 0.865, fit=_fit()))
    doc["results"]["B"]["readback"] = {"applicable": True, "accuracy": 0.991,
                                       "cer": 0.0013}
    doc["results"]["B"]["functional"] = {
        "text_accuracy": 0.941, "image_accuracy": 1.0,
        "ablated_accuracy": 0.449, "validity_warning": None,
        "delta_warning": "image view is at ceiling"}
    out = _run(doc, tmp_path)
    assert out.index("read-back") < out.index("raw MSG")
    assert "0.991" in out and "span-free floor 0.449" in out


def test_an_invalid_tier_suppresses_the_geometry(tmp_path):
    doc = _doc(msg_linear=_null(0.858, 0.851, 0.865, fit=_fit()))
    doc["results"]["B"]["functional"] = {
        "text_accuracy": 0.99, "image_accuracy": 0.48, "ablated_accuracy": 0.52,
        "validity_warning": "the model cannot recover the span from the image"}
    out = _run(doc, tmp_path)
    assert "TIER INVALID" in out
    assert "REMOVABLE" not in out and "IRREDUCIBLE" not in out


def test_a_file_with_no_validity_record_says_so(tmp_path):
    out = _run(_doc(msg_linear=_null(0.858, 0.851, 0.865, fit=_fit())), tmp_path)
    assert "unvalidated" in out


def test_low_readback_is_flagged(tmp_path):
    doc = _doc(msg_linear=_null(0.858, 0.851, 0.865, fit=_fit()))
    doc["results"]["B"]["readback"] = {"applicable": True, "accuracy": 0.83,
                                       "cer": 0.06}
    doc["results"]["B"]["functional"] = {
        "text_accuracy": 0.94, "image_accuracy": 0.99, "ablated_accuracy": 0.45,
        "validity_warning": None, "delta_warning": None}
    out = _run(doc, tmp_path)
    assert "below 0.95" in out and "contaminate MSG" in out
