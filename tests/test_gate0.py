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


def _row(tokens: int, acc1: float, acc3: float, passes: bool) -> dict:
    return {"visual_tokens": tokens, "passes": passes,
            "train": {"w1": {"acc": acc1}, "w3": {"acc": acc3}}}


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
