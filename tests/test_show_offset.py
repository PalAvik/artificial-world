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


def test_a_collapsed_image_control_is_refused(tmp_path):
    """Only the image half of the denominator passes through the map, so a map
    that collapsed it would look exactly like a vanished gap."""
    out = _run(_doc(msg_linear=_null(0.20, 0.19, 0.21, fit=_fit(),
                                     wt=0.049, wi=0.004)), tmp_path)
    assert "IMAGE CONTROL COLLAPSED" in out
    assert "REMOVABLE" not in out


def test_a_sound_sub_one_result_is_reported_as_removable(tmp_path):
    out = _run(_doc(msg_linear=_null(0.858, 0.851, 0.865, fit=_fit(),
                                     wt=0.049, wi=0.041)), tmp_path)
    assert "both halves of the denominator survive" in out
    assert "REMOVABLE" in out


def test_a_surviving_gap_is_reported_as_irreducible(tmp_path):
    out = _run(_doc(msg_linear=_null(2.9, 2.85, 2.95, fit=_fit())), tmp_path)
    assert "IRREDUCIBLE" in out
