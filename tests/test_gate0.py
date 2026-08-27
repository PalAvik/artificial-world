"""CPU-only tests for the Gate 0 sweep.

The parts worth testing are the ones that decide the gate — string metrics,
span construction, and the constrained minimisation — none of which need a GPU.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import importlib.util
import random
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Import a script without a real torch, which the CPU tests don't need."""
    if "torch" not in sys.modules:
        stub = types.ModuleType("torch")
        stub.no_grad = lambda: (lambda f: f)
        sys.modules["torch"] = stub
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g0 = _load("gate0_sweep")


def _row(tokens: int, acc1: float, acc3: float, passes: bool,
         held_ok: bool = True) -> dict:
    return {"visual_tokens": tokens, "passes": passes, "held_ok": held_ok,
            "train": {"w1": {"acc": acc1, "acc_lo": acc1},
                      "w3": {"acc": acc3, "acc_lo": acc3}}}


class TestStringMetrics:
    def test_levenshtein(self):
        assert g0.levenshtein("kitten", "sitting") == 3
        assert g0.levenshtein("", "abc") == 3
        assert g0.levenshtein("same", "same") == 0

    def test_normalise_ignores_case_padding_and_wrapping_punctuation(self):
        assert g0.normalise('  "The Dog." ') == "the dog"
        assert g0.normalise("the  dog") == g0.normalise("The Dog")

    def test_normalise_keeps_internal_content(self):
        # Only wrapping punctuation is stripped; a wrong transcription must
        # still count as wrong.
        assert g0.normalise("the cat") != g0.normalise("the dog")


class TestSpans:
    def test_classes_are_balanced(self):
        spans = g0.make_spans(64, 1, random.Random(0))
        counts = Counter(cls for _, cls in spans)
        assert set(counts) == set(g0.WORD_CLASSES)
        assert len(set(counts.values())) == 1

    def test_multiword_spans_are_class_pure_without_repeats(self):
        for text, cls in g0.make_spans(32, 3, random.Random(0)):
            words = text.split()
            assert len(words) == 3
            assert len(set(words)) == 3, "a repeated word makes the span easier"
            assert all(w in g0.WORD_CLASSES[cls] for w in words)

    def test_deterministic_under_seed(self):
        assert (g0.make_spans(16, 3, random.Random(7)) ==
                g0.make_spans(16, 3, random.Random(7)))


class TestRendering:
    @pytest.mark.parametrize("height", [16, 24, 32, 48])
    def test_renders_non_blank_strip_at_requested_height(self, height):
        import matplotlib
        font = (Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
                / "DejaVuSans.ttf")
        img = g0.render("the quick brown", str(font), height)
        assert img.size[1] == height
        assert len(img.getcolors(65536)) > 1, "rendered blank"

    def test_wider_text_makes_a_wider_strip(self):
        import matplotlib
        font = str(Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
                   / "DejaVuSans.ttf")
        assert (g0.render("a", font, 32).size[0]
                < g0.render("a much longer span", font, 32).size[0])


class _FakeTensor:
    """Just enough tensor to exercise read_back's slicing."""

    def __init__(self, rows: int, cols: int):
        self.shape = (rows, cols)

    def __getitem__(self, idx):
        _, sl = idx
        return _FakeTensor(self.shape[0], self.shape[1] - (sl.start or 0))

    def to(self, _device):
        return self


class _FakeInputs(dict):
    def to(self, _device):
        return self


class _FakeProcessor:
    """Records what it was handed, so batching and ordering can be asserted."""

    def __init__(self):
        self.chunks: list[int] = []
        self.counter = 0

    def apply_chat_template(self, msgs, **_kw):
        # Touching the content is what catches a NameError in the caller's
        # message construction — the bug this class exists to prevent.
        for part in msgs[0]["content"]:
            assert part.get("image") is not None or part.get("text") is not None
        return "PROMPT"

    def __call__(self, text, images, **_kw):
        assert len(text) == len(images), "one prompt per image"
        self.chunks.append(len(images))
        return _FakeInputs(input_ids=_FakeTensor(len(images), 7))

    def batch_decode(self, new, **_kw):
        n = new.shape[0]
        out = [f"pred{self.counter + i}" for i in range(n)]
        self.counter += n
        return out


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids, **_kw):
        return _FakeTensor(input_ids.shape[0], input_ids.shape[1] + 24)


class TestReadBack:
    def test_returns_one_prediction_per_image_in_order(self):
        proc, model = _FakeProcessor(), _FakeModel()
        images = [f"img{i}" for i in range(10)]
        preds = g0.read_back(model, proc, images, batch=4)
        assert preds == [f"pred{i}" for i in range(10)]

    def test_respects_batch_size_including_the_short_final_chunk(self):
        proc, model = _FakeProcessor(), _FakeModel()
        g0.read_back(model, proc, [f"img{i}" for i in range(10)], batch=4)
        assert proc.chunks == [4, 4, 2]

    def test_handles_a_single_image(self):
        proc, model = _FakeProcessor(), _FakeModel()
        assert g0.read_back(model, proc, ["only"], batch=32) == ["pred0"]

    def test_empty_input_makes_no_calls(self):
        proc, model = _FakeProcessor(), _FakeModel()
        assert g0.read_back(model, proc, [], batch=8) == []
        assert proc.chunks == []


class TestSelection:
    def test_picks_cheapest_passing_not_most_accurate(self):
        rows = [_row(1, 0.90, 0.80, False), _row(3, 0.96, 0.90, True),
                _row(6, 0.99, 0.98, True), _row(10, 0.99, 0.99, True)]
        assert g0.select_winner(rows)["visual_tokens"] == 3

    def test_ties_on_cost_break_toward_accuracy(self):
        rows = [_row(3, 0.96, 0.90, True), _row(3, 0.98, 0.93, True)]
        assert g0.select_winner(rows)["train"]["w1"]["acc"] == 0.98

    def test_returns_none_when_nothing_passes(self):
        # A gate outcome, not an error: the caller branches CONDITIONAL vs DROP.
        assert g0.select_winner([_row(1, 0.5, 0.4, False)]) is None

    def test_failing_rows_are_never_selected(self):
        rows = [_row(1, 0.99, 0.99, False), _row(9, 0.95, 0.88, True)]
        assert g0.select_winner(rows)["visual_tokens"] == 9

    def test_held_out_failure_disqualifies_a_train_passing_config(self):
        # The exact shape of the first real sweep: the cheapest config cleared
        # on training fonts but not on held-out ones, which would leave Gate
        # 2(a) floor-limited by OCR.
        rows = [_row(3, 0.95, 0.98, True, held_ok=False),
                _row(8, 0.99, 0.98, True, held_ok=True)]
        assert g0.select_winner(rows)["visual_tokens"] == 8

    def test_returns_none_when_all_passing_configs_fail_held_out(self):
        assert g0.select_winner([_row(3, 0.99, 0.99, True, held_ok=False)]) is None


class TestHeldOutFloor:
    """The floor is deliberately below the train threshold — see select_winner.
    These pin the calibration so it cannot drift back by accident."""

    def test_a_config_just_under_the_train_threshold_on_held_out_still_qualifies(self):
        # h=32/min_px=4096 from the n=512 sweep: held-out lower bound 0.945.
        # Under the train threshold it was rejected; under the floor it is fine.
        rows = [_row(8, 0.975, 0.947, True, held_ok=True)]
        assert g0.select_winner(rows)["visual_tokens"] == 8

    def test_floor_is_five_points_below_by_default(self):
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--held-margin", type=float, default=0.05)
        assert ap.parse_args([]).held_margin == 0.05

    def test_genuinely_illegible_held_out_still_disqualifies(self):
        # h=16 configs: held-out ~0.71 lower bound, far under any sensible floor.
        assert g0.select_winner([_row(4, 0.97, 0.95, True, held_ok=False)]) is None


class TestWilson:
    def test_lower_bound_is_below_the_point_estimate(self):
        lo = g0.wilson_lower(122, 128)
        assert 0.89 < lo < 0.92, lo          # 0.953 point -> ~0.90 lower

    def test_tightens_with_more_samples(self):
        assert g0.wilson_lower(970, 1000) > g0.wilson_lower(97, 100)

    def test_n128_cannot_resolve_a_grazing_point_estimate(self):
        # 0.953 at n=128 does not clear a 0.95 threshold at the lower bound.
        assert g0.wilson_lower(122, 128) < 0.95

    def test_n512_resolves_a_097_point_estimate(self):
        assert g0.wilson_lower(round(0.97 * 512), 512) >= 0.95

    def test_degenerate_n_is_zero_not_a_crash(self):
        assert g0.wilson_lower(0, 0) == 0.0
