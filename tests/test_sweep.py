"""The sweep's summary logic, which decides what the multi-model claim says.

Driving real models needs a GPU; the part that can go silently wrong on CPU is
the collection — whether an invalid row is reported as invalid rather than
folded into the comparison as a number.

    python -m pytest tests/test_sweep.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "model_sweep", ROOT / "scripts" / "model_sweep.py")
sweep = importlib.util.module_from_spec(spec)
sys.modules["model_sweep"] = sweep
spec.loader.exec_module(sweep)


def _fit(**kw):
    base = {"kind": "linear", "dim": 2048, "n_groups": 8000,
            "underdetermined": None, "collapsed": None}
    base.update(kw)
    return base


def _results(tmp_path, **over):
    res = {
        "readback": {"applicable": True, "accuracy": 0.977, "cer": 0.004},
        "functional": {"text_accuracy": 0.996, "image_accuracy": 1.0,
                       "ablated_accuracy": 0.523, "validity_warning": None},
        "msg_raw": {"overall": {"msg": 3.485}},
        "msg_offset_free": {"overall": {"msg": 3.331}, "fit": None},
        "msg_procrustes": {"overall": {"msg": 1.262}, "fit": _fit(kind="orthogonal")},
        "msg_linear": {"overall": {"msg": 0.858}, "fit": _fit()},
    }
    res.update(over)
    d = tmp_path / "m"
    d.mkdir(exist_ok=True)
    (d / "results.json").write_text(json.dumps({"results": {"B": res}}))
    return d


def test_shares_are_computed_within_model(tmp_path):
    """The one quantity comparable across models: raw MSG depends on the
    tokeniser, the visual-token count and the hidden size, the shares do not."""
    row = sweep.collect("m/x", _results(tmp_path))
    assert row["valid"] == "yes"
    assert abs(row["rotation"] - 0.897) < 0.01      # 1-(1.262-1)/(3.485-1)
    assert row["linear"] > 1.0                      # takes MSG below raw
    assert row["dim"] == 2048


def test_a_model_that_cannot_read_the_span_is_marked_invalid(tmp_path):
    row = sweep.collect("m/x", _results(
        tmp_path, readback={"applicable": True, "accuracy": 0.62, "cer": 0.3}))
    assert row["valid"] == "read-back below 0.95"


def test_untestable_map_nulls_report_nothing(tmp_path):
    """A 4096-wide model needs ~8192 distinct spans and our pool holds 8000, so
    its map nulls must abstain rather than contribute a number to the table."""
    row = sweep.collect("m/x", _results(tmp_path, msg_linear={
        "overall": {"msg": 2.9},
        "fit": _fit(dim=4096, underdetermined="8000 distinct spans for 4096")}))
    assert row["linear"] is None
    assert row["valid"] == "map nulls untestable"


def test_a_failed_forced_choice_invalidates_the_row(tmp_path):
    row = sweep.collect("m/x", _results(tmp_path, functional={
        "text_accuracy": 0.99, "image_accuracy": 0.48, "ablated_accuracy": 0.52,
        "validity_warning": "cannot recover the span from the image"}))
    assert row["valid"] == "forced choice invalid"


def test_a_missing_run_does_not_break_the_summary(tmp_path):
    row = sweep.collect("m/x", tmp_path / "absent")
    assert row["error"] == "no results.json"
    assert "no results.json" in sweep.summarise([row])


def test_the_summary_warns_that_raw_msg_is_not_comparable(tmp_path):
    text = sweep.summarise([sweep.collect("m/x", _results(tmp_path))])
    assert "not** comparable across rows" in text
