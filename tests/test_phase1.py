"""End-to-end test of the Phase 1 driver against a fake VLM.

The driver is mostly wiring, and wiring bugs are exactly the kind that survive
unit tests and then cost a cluster round-trip to discover. So this stands up a
minimal model and processor with the same contract the real ones have —
character-level tokenisation, an expanding image placeholder, hidden states, an
output head, greedy generation — and runs the whole pipeline on it.

What it is really checking: that merge positions land on the right token, that
the four views stay aligned through suffix-grouped batching, that read-back
conditioning selects the items it claims to, and that a report comes out the
far end.

    python -m pytest tests/test_phase1.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from freeflow.data import tier_b, views
from freeflow.data.render import FontSet, RenderConfig
from freeflow.metrics import cycle, functional, runner

ROOT = Path(__file__).resolve().parents[1]
VISION = "<|vision_start|><|image_pad|><|vision_end|>"
IMG_TOKENS = 6                     # matches the frozen Gate 0 config
PAD, IMG_ID, VOCAB, DIM = 0, 1, 128, 24


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "phase1_measure", ROOT / "scripts" / "phase1_measure.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase1_measure"] = mod
    spec.loader.exec_module(mod)
    return mod


driver = _load_driver()


# ----------------------------------------------------------- the fake stack ---

class _Tok:
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": _encode(text)}


def _encode(text: str) -> list[int]:
    """Character-level, so tokenisation is compositional and a suffix really is
    a token-level suffix — the property runner asserts at runtime."""
    out: list[int] = []
    i = 0
    while i < len(text):
        if text.startswith(VISION, i):
            out.extend([IMG_ID] * IMG_TOKENS)
            i += len(VISION)
        else:
            out.append(2 + (ord(text[i]) % (VOCAB - 2)))
            i += 1
    return out


class _Batch(dict):
    """Real processors return a BatchFeature, which has .to(); a plain dict
    does not, and cycle.read_back rightly relies on it."""

    def to(self, _device):
        return self


class FakeProcessor:
    def __init__(self):
        self.tokenizer = _Tok()
        self.last_images: list = []

    def __call__(self, text, images=None, padding=True, return_tensors="pt"):
        self.last_images = list(images or [])
        rows = [_encode(t) for t in text]
        # Image tokens must depend on the pixels, or two fonts of the same span
        # tokenise identically, the image control costs nothing, and the MSG
        # denominator loses half its mass without anything noticing.
        for i, img in enumerate(self.last_images):
            if i >= len(rows):
                break
            tag = 2 + (hash(img.tobytes()) % (VOCAB - 2))
            rows[i] = [tag if t == IMG_ID else t for t in rows[i]]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), PAD, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            # Right padding, matching what the runner requires.
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        return _Batch(input_ids=ids, attention_mask=mask)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        parts = []
        for c in msgs[0]["content"]:
            parts.append(VISION if c.get("type") == "image" else c.get("text", ""))
        return "".join(parts)

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(int(t)) for t in row if int(t) > 31) for row in ids]


class FakeModel(torch.nn.Module):
    """Deterministic, and deliberately *not* modality-blind: image tokens get a
    distinct embedding, so the cross-modal distance is non-degenerate and MSG
    has something to measure."""

    device = "cpu"

    def __init__(self, processor: FakeProcessor, misread_every: int = 0):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, DIM)
        self.l1 = torch.nn.Linear(DIM, DIM)
        self.l2 = torch.nn.Linear(DIM, DIM)
        self.head = torch.nn.Linear(DIM, VOCAB, bias=False)
        self.processor = processor
        self.misread_every = misread_every
        self._calls = 0

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                **_kw):
        h0 = self.embed(input_ids)
        # Causal mixing — a running mean over preceding positions. Without it
        # the merge-position state would depend only on the token sitting there
        # (identical in both views), the cross-modal distance would be ~0, and
        # the test would validate the plumbing while measuring nothing.
        pos = torch.arange(1, h0.shape[1] + 1, device=h0.device).view(1, -1, 1)
        mixed = h0.cumsum(dim=1) / pos
        h1 = mixed + torch.tanh(self.l1(mixed))
        h2 = h1 + torch.tanh(self.l2(h1))
        out = type("Out", (), {})()
        out.hidden_states = (h0, h1, h2) if output_hidden_states else None
        out.logits = self.head(h2)
        return out

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=24, **_kw):
        """Echo each image's attached ground-truth text, mangling every Nth so
        the read-back conditioning path has both outcomes to handle."""
        rows = []
        for img in self.processor.last_images:
            text = getattr(img, "freeflow_text", "")
            self._calls += 1
            if self.misread_every and self._calls % self.misread_every == 0:
                text = text + "zz"
            rows.append([ord(c) for c in text])
        width = max((len(r) for r in rows), default=1) or 1
        out = torch.full((len(rows), input_ids.shape[1] + width), 32,
                         dtype=torch.long)
        out[:, :input_ids.shape[1]] = input_ids
        for i, r in enumerate(rows):
            out[i, input_ids.shape[1]:input_ids.shape[1] + len(r)] = torch.tensor(r)
        return out


@pytest.fixture(scope="module")
def stack():
    proc = FakeProcessor()
    return FakeModel(proc, misread_every=7), proc


@pytest.fixture(scope="module")
def corpus():
    import matplotlib
    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    fonts = FontSet(train={"a": str(ttf / "DejaVuSans.ttf"),
                           "b": str(ttf / "DejaVuSerif.ttf"),
                           "c": str(ttf / "DejaVuSansMono.ttf")},
                    held_out={"h": str(ttf / "cmr10.ttf"),
                              "h2": str(ttf / "STIXGeneral.ttf")})
    cfg = RenderConfig(height=48, pad=4, min_pixels=1024, visual_tokens_per_span=6)
    items = tier_b.build(64, fonts, cfg, seed=0)
    for it in items:                       # let the fake model "read" the strip
        it.span_image.freeflow_text = it.span_text
        it.span_image_alt.freeflow_text = it.span_text
    return items


# ------------------------------------------------------------- merge index ---

class TestMergePosition:
    def test_lands_on_the_first_suffix_token_in_both_views(self, stack, corpus):
        """The index the entire comparison hangs on. A wrong one would shift
        every hidden state by a token and never raise."""
        _, proc = stack
        chunk = [it for it in corpus if it.suffix == corpus[0].suffix][:4]
        for modality in ("text", "image"):
            vb = runner.build_view(proc, chunk, modality, "primary", device="cpu")
            ids, mask = vb.inputs["input_ids"], vb.inputs["attention_mask"]
            for row, item in enumerate(chunk):
                n = int(mask[row].sum())
                start = int(vb.merge_index[row])
                assert ids[row, start:n].tolist() == _encode(item.suffix)

    def test_the_image_view_is_longer_by_the_visual_token_count(self, stack, corpus):
        _, proc = stack
        chunk = [it for it in corpus if it.suffix == corpus[0].suffix][:4]
        t = runner.build_view(proc, chunk, "text", "primary", device="cpu")
        i = runner.build_view(proc, chunk, "image", "primary", device="cpu")
        for row, item in enumerate(chunk):
            delta = int(i.merge_index[row]) - int(t.merge_index[row])
            assert delta == IMG_TOKENS - len(item.span_text)

    def test_a_mixed_suffix_batch_is_refused(self, stack, corpus):
        _, proc = stack
        mixed = [corpus[0], next(c for c in corpus if c.suffix != corpus[0].suffix)]
        with pytest.raises(ValueError, match="share a suffix length"):
            runner.build_view(proc, mixed, "text", "primary", device="cpu")

    def test_rows_disagreeing_on_their_final_tokens_are_caught(self):
        """The real hazard: a token merging across the span/suffix boundary for
        some items but not others, which shifts those rows' merge positions
        while every index still looks plausible.

        Checked against the other rows rather than against the suffix tokenised
        in isolation — a SentencePiece leading space is its own token at
        sequence start and merges into the next one mid-sequence, so the
        standalone form legitimately differs. Comparing against it aborted the
        llava-v1.6 run over one token id at equal length.
        """
        inputs = {"input_ids": torch.tensor([[5, 6, 7, 8], [5, 6, 9, 8]]),
                  "attention_mask": torch.ones(2, 4, dtype=torch.long)}
        with pytest.raises(ValueError, match="do not share their final"):
            runner._assert_clean_boundary(inputs, [], k=2, processor=None)

    def test_agreeing_rows_pass_even_with_unusual_ids(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 7, 8], [3, 4, 7, 8]]),
                  "attention_mask": torch.ones(2, 4, dtype=torch.long)}
        runner._assert_clean_boundary(inputs, [], k=2, processor=None)


# ------------------------------------------------------------------ capture ---

class TestCapture:
    def test_all_four_views_are_captured_and_aligned(self, stack, corpus):
        model, proc = stack
        cap = runner.capture(model, proc, corpus, batch=8, device="cpu")
        assert set(cap) == {"text", "image", "text_control", "image_control"}
        n = len(corpus)
        for key, res in cap.items():
            assert len(res.span_ids) == n, key
            assert res.hidden["0"].shape[0] == n, key
        # Every view must see the same items in the same order, or the
        # numerator and denominator would be computed on different spans.
        ids = [cap[k].span_ids for k in cap]
        assert all(x == ids[0] for x in ids)

    def test_jsd_is_accumulated_once_per_item(self, stack, corpus):
        model, proc = stack
        cap = runner.capture(model, proc, corpus, batch=8, device="cpu")
        assert cap["text"].jsd.per_item.shape == (len(corpus),)
        assert cap["text"].jsd.summary()["n"] == len(corpus)

    def test_suffix_grouping_does_not_drop_or_duplicate_items(self, stack, corpus):
        model, proc = stack
        cap = runner.capture(model, proc, corpus, batch=8, device="cpu")
        assert sorted(cap["text"].span_ids) == sorted(it.span_id for it in corpus)


# ------------------------------------------------------------------ driver ---

class TestDriver:
    def test_measure_tier_produces_every_reported_quantity(self, stack, corpus):
        model, proc = stack
        res = driver.measure_tier(model, proc, corpus, batch=8, layers=None,
                                  device="cpu")
        for key in ("msg_raw", "msg_offset_free", "per_layer", "jsd",
                    "readback", "probe_text", "probe_image"):
            assert key in res, key
        assert res["msg_raw"]["overall"]["msg"] > 0
        assert res["msg_raw"]["overall"]["ci"] is not None
        assert len(res["per_layer"]) == len(res["layers_captured"])

    def test_readback_conditioning_selects_the_correctly_read_items(
            self, stack, corpus):
        model, proc = stack
        rate = driver.measure_tier(model, proc, corpus, batch=8, layers=None,
                                   device="cpu")["msg_raw"]["readback_rate"]
        # The fake mangles every 7th transcription, so the rate is high but
        # emphatically not 1.0 — both branches of the conditioning are exercised.
        assert 0.5 < rate < 1.0

    def test_the_merge_state_actually_depends_on_the_span(self, stack, corpus):
        """Guards the test itself: with a non-causal model the merge position
        carries no span information, the cross-modal distance collapses to ~0,
        and every metric below would be vacuously satisfied."""
        model, proc = stack
        cap = runner.capture(model, proc, corpus, batch=8, device="cpu")
        final = str(len(cap["text"].hidden["layers"]) - 1)
        d = driver.distances(cap, final, "raw")
        assert float(d["cross"].mean()) > 1e-3, "merge state ignores the span"
        assert float(d["within_text"].mean()) > 0

    def test_offset_free_msg_is_computed_and_differs_from_raw(self, stack, corpus):
        model, proc = stack
        res = driver.measure_tier(model, proc, corpus, batch=8, layers=None,
                                  device="cpu")
        assert res["msg_offset_free"]["overall"]["msg"] != pytest.approx(
            res["msg_raw"]["overall"]["msg"], rel=1e-3)

    def test_denominator_halves_are_reported_separately(self, stack, corpus):
        """If one control dominates, the normalisation is really being set by
        that control alone — which the reader has to be able to see."""
        model, proc = stack
        res = driver.measure_tier(model, proc, corpus, batch=8, layers=None,
                                  device="cpu")
        block = res["msg_raw"]
        assert block["within_text_mean"] > 0
        assert block["within_image_mean"] > 0, (
            "the image control cost nothing — half the denominator is missing")

    def test_report_renders_with_the_gate_verdict(self, stack, corpus):
        model, proc = stack
        res = driver.measure_tier(model, proc, corpus, batch=8, layers=None,
                                  device="cpu")
        res["gate1"] = "PASS: MSG 1.90, CI lower 1.60 > 1.25"
        args = type("A", (), {"model": "fake", "n": len(corpus), "batch": 8,
                              "attn": "sdpa"})()
        report = driver.render_report({"B": res}, args)
        assert "# Phase 1" in report and "Gate 1" in report
        assert "PASS" in report and "Tier B" in report
        assert "offset-free" in report

    def test_build_corpus_rejects_tier_a_without_data(self):
        with pytest.raises(ValueError, match="Flickr30k"):
            driver.build_corpus("A", 4, None, None, 0, False)


# ------------------------------------------------- forced choice / validity ---

class TestForcedChoice:
    def test_distractors_come_from_the_same_group(self, corpus):
        """A distractor from another word class would be separable on register
        alone, and the task would stop testing whether the span was encoded."""
        d = functional.make_distractors(corpus, seed=0)
        by_group = {}
        for it in corpus:
            by_group.setdefault(it.group, set()).add(it.span_text)
        for it, dis in zip(corpus, d):
            assert dis != it.span_text
            assert dis in by_group[it.group]

    def test_single_span_group_falls_back_rather_than_self_matching(self):
        from freeflow.data import tier_c
        items = tier_c.build(10, seed=0)
        for it, dis in zip(items, functional.make_distractors(items, seed=0)):
            assert dis != it.span_text

    def test_scores_both_views_and_reports_a_delta(self, stack, corpus):
        model, proc = stack
        r = functional.forced_choice(model, proc, corpus[:8], device="cpu")
        assert 0.0 <= r.text.accuracy <= 1.0
        assert 0.0 <= r.image.accuracy <= 1.0
        assert r.delta == pytest.approx(r.text.accuracy - r.image.accuracy)

    def test_flags_a_negative_delta_below_ceiling(self):
        """A negative delta with headroom left in the image view still means the
        two views are answering different questions."""
        broken = functional.FunctionalResult(
            text=functional.ChoiceResult(0.910, 256),
            image=functional.ChoiceResult(0.975, 256),
            ablated=functional.ChoiceResult(0.480, 256))
        assert "not answering the same question" in broken.delta_verdict()

    def test_a_saturated_image_view_bounds_the_delta_rather_than_breaking(self):
        """Tier B, 2026-08-27: text 0.941, image 1.000, floor 0.449. Reported as
        a broken task, but the image view had simply run out of headroom — at
        Tier B the image is a rendering of the answer, so it can read the span as
        directly as the text view can. Valid measurement, unusable delta."""
        sat = functional.FunctionalResult(
            text=functional.ChoiceResult(0.941, 256),
            image=functional.ChoiceResult(1.000, 256),
            ablated=functional.ChoiceResult(0.449, 256))
        assert sat.validity() is None          # the image view recovers the span
        assert "at ceiling" in sat.delta_verdict()   # but the delta is a bound

    def test_saturation_is_diagnosed_before_a_negative_delta(self):
        """Order matters: saturation explains a negative delta, not the reverse.
        Getting this backwards is what mislabelled Tier B."""
        sat = functional.FunctionalResult(
            text=functional.ChoiceResult(0.90, 256),
            image=functional.ChoiceResult(1.00, 256),
            ablated=functional.ChoiceResult(0.45, 256))
        assert sat.delta < -0.05 and "at ceiling" in sat.delta_verdict()

    def test_flags_a_text_view_that_cannot_read_its_own_answer(self):
        weak = functional.FunctionalResult(
            text=functional.ChoiceResult(0.70, 256),
            image=functional.ChoiceResult(0.60, 256))
        assert "mis-specified" in weak.task_sanity()

    def test_a_healthy_task_passes_sanity(self):
        good = functional.FunctionalResult(
            text=functional.ChoiceResult(0.98, 256),
            image=functional.ChoiceResult(0.88, 256))
        assert good.task_sanity() is None and good.validity() is None

    def test_flags_a_choice_that_is_decidable_without_the_span(self):
        """The control the first three Phase 1 runs did not have. If the blanked
        view scores what the text view scores, both views are riding on the
        context and the delta is not about the substitution at all."""
        hollow = functional.FunctionalResult(
            text=functional.ChoiceResult(0.90, 256),
            image=functional.ChoiceResult(0.98, 256),
            ablated=functional.ChoiceResult(0.88, 256))
        assert "decidable from the context alone" in hollow.task_sanity()

    def test_the_floor_replaces_chance_once_it_is_measured(self):
        """An image view well above chance can still be at the floor: same-group
        distractors are not indistinguishable without the span, so chance
        understates what the context gives away."""
        r = functional.FunctionalResult(
            text=functional.ChoiceResult(0.95, 256),
            image=functional.ChoiceResult(0.80, 256),
            ablated=functional.ChoiceResult(0.75, 256))
        assert r.image_above_chance > 0.15 and r.image_above_floor < 0.15
        assert "span-free floor" in r.validity()

    def test_flags_an_image_view_stuck_at_chance(self):
        """The Tier C failure the first Phase 1 run could not see: a diagram the
        model cannot decode still yields a confident MSG."""
        blind = functional.FunctionalResult(
            text=functional.ChoiceResult(0.95, 100),
            image=functional.ChoiceResult(0.52, 100))
        assert "cannot recover the span" in blind.validity()

    def test_no_warning_when_the_image_view_is_informative(self):
        ok = functional.FunctionalResult(
            text=functional.ChoiceResult(0.95, 100),
            image=functional.ChoiceResult(0.84, 100))
        assert ok.validity() is None

    def test_the_ablated_view_blanks_the_span_in_both_modalities(self, stack,
                                                                 corpus):
        """The floor has to remove the span, not move it: scored through the
        text path with a blank, so no image carries it either."""
        model, proc = stack
        r = functional.forced_choice(model, proc, corpus[:6], device="cpu")
        assert r.ablated is not None and r.ablated.n == 6
        assert functional.BLANK not in corpus[0].span_text

    def test_option_length_is_measured_in_context_not_standalone(self, stack,
                                                                 corpus):
        """Tokenising an option on its own can disagree with how it tokenises
        glued to the preceding text, drifting the scored positions off it."""
        model, proc = stack
        scores = functional._option_scores(
            model, proc, corpus[:4], "text",
            [it.span_text for it in corpus[:4]], device="cpu", neutral_cache={})
        assert scores.shape == (4,) and torch.isfinite(scores).all()

    def test_the_neutral_baseline_is_subtracted_from_every_score(self, stack,
                                                                 corpus):
        """PMI, not raw likelihood. Without the null-context term an option's
        unconditional frequency competes with the in-context evidence, which is
        how a rare true span loses to a common distractor with the answer
        written out in front of the model."""
        model, proc = stack
        items, opts = corpus[:4], [it.span_text for it in corpus[:4]]
        cache = {}
        pmi = functional._option_scores(model, proc, items, "text", opts,
                                        device="cpu", neutral_cache=cache)
        neutral = torch.tensor([cache[o] for o in opts])
        raw = functional._option_scores(
            model, proc, items, "text", opts, device="cpu",
            neutral_cache={o: 0.0 for o in opts})
        assert torch.allclose(pmi, raw - neutral, atol=1e-5)

    def test_the_neutral_score_is_computed_once_per_option_string(self, stack,
                                                                  corpus):
        """Cached, and the same cache is handed to both views — a calibration
        term that differed by modality would put the delta back where it was."""
        model, proc = stack
        items = corpus[:4]
        opts = [items[0].span_text] * 4
        cache = {}
        calls = []
        real = functional._logprob_of_tail

        def counting(m, inputs, n_full, k):
            calls.append(k)
            return real(m, inputs, n_full, k)

        functional._logprob_of_tail = counting
        try:
            functional._option_scores(model, proc, items, "text", opts,
                                      device="cpu", neutral_cache=cache)
        finally:
            functional._logprob_of_tail = real
        # Four conditioned scores, one neutral score for the single option.
        assert len(cache) == 1 and len(calls) == 5


class TestTierAwareValidity:
    def test_read_back_is_skipped_for_a_tier_without_text_in_its_images(
            self, stack):
        """Asking a relation diagram to be transcribed is a category error;
        the first Phase 1 run reported read-back 0.000 on Tier C for that
        reason alone."""
        from freeflow.data import tier_c
        model, proc = stack
        items = tier_c.build(20, seed=0)
        res = driver.measure_tier(model, proc, items, batch=8, layers=None,
                                  device="cpu", functional_n=8)
        assert res["readback"]["applicable"] is False
        assert res["readback"]["accuracy"] is None
        assert all(it.read_ok for it in items)

    def test_read_back_still_runs_for_tier_b(self, stack, corpus):
        model, proc = stack
        res = driver.measure_tier(model, proc, corpus[:24], batch=8, layers=None,
                                  device="cpu", functional_n=8)
        assert res["readback"]["applicable"] is True
        assert res["readback"]["accuracy"] is not None


# ------------------------------------------------------------- read-back ---

class TestReadBackScoring:
    def test_scores_exact_matches_and_character_errors(self):
        r = cycle.score(["dog", "cat"], ["dog", "cot"])
        assert r.accuracy == 0.5 and r.correct == [True, False]
        assert r.cer == pytest.approx(1 / 6)

    def test_normalisation_forgives_case_and_wrapping_punctuation(self):
        assert cycle.score(["dog"], ['  "Dog." ']).accuracy == 1.0

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="truths for"):
            cycle.score(["a", "b"], ["a"])

    def test_mark_read_ok_sets_the_flag_in_place(self, stack, corpus):
        model, proc = stack
        items = [
            views.SpanItem(prefix="p ", span_text=it.span_text, suffix=" s.",
                           span_image=it.span_image, span_paraphrase="X",
                           span_image_alt=it.span_image_alt, span_id=it.span_id)
            for it in corpus[:16]]
        result = cycle.mark_read_ok(model, proc, items, batch=8)
        assert [it.read_ok for it in items] == result.correct
        assert 0.0 < result.accuracy <= 1.0


class TestNullHierarchy:
    """The gap measured against progressively more generous accounts of it."""

    def test_every_mode_runs_end_to_end(self, stack, corpus):
        model, proc = stack
        cap = runner.capture(model, proc, corpus, batch=8, device="cpu")
        final = str(len(cap["text"].hidden["layers"]) - 1)
        for mode in ("raw", "offset_free", "procrustes", "linear"):
            d = driver.distances(cap, final, mode, seed=0)
            assert set(d) == {"cross", "within_text", "within_image", "fit"}
            assert all(torch.isfinite(d[k]).all()
                       for k in ("cross", "within_text", "within_image"))
            # Only the mapped modes carry a fit, and it reports its own
            # sample-size adequacy.
            assert (d["fit"] is None) == (mode in ("raw", "offset_free"))

    def test_a_rotated_view_survives_raw_and_dies_under_procrustes(self, stack,
                                                                   corpus):
        """The discriminating case. If the image view is a rotation of the text
        view, the raw distance is large and the mapped one is not — which is
        exactly the claim the Phase 1 numbers need tested."""
        model, proc = stack
        cap = runner.capture(model, proc, corpus, batch=8, device="cpu")
        final = str(len(cap["text"].hidden["layers"]) - 1)
        h_t = cap["text"].hidden[final].float()
        torch.manual_seed(0)
        q, _ = torch.linalg.qr(torch.randn(h_t.shape[1], h_t.shape[1]))
        cap["image"].hidden[final] = h_t @ q
        cap["image_control"].hidden[final] = cap["text_control"].hidden[final].float() @ q
        raw = driver.distances(cap, final, "raw")["cross"].mean()
        mapped = driver.distances(cap, final, "procrustes")["cross"].mean()
        assert raw > 0.2 and mapped < raw / 2


class TestBanner:
    """The run banner touches vocab and config summaries before any GPU work.

    It broke on a return-type change to `vocab.length_summary()` that every
    other test missed, because nothing exercised the printed path — and it broke
    after the model had loaded on the cluster, which is the expensive place to
    find a formatting bug.
    """

    def test_the_length_summary_renders(self):
        from freeflow.data import vocab
        lines = []
        for name, st in sorted(vocab.length_summary().items()):
            lines.append(f"{name:20} n={st['n']:6}  {st['mean']:5.2f} "
                         f"+/- {st['sd']:.2f}")
        assert len(lines) == len(vocab.CLASSES) + len(vocab.MATCHED) + 1
        assert all(ch in "".join(lines) for ch in ("function", "concrete"))

    def test_every_summary_entry_has_the_keys_the_banner_reads(self):
        from freeflow.data import vocab
        for name, st in vocab.length_summary().items():
            assert {"n", "mean", "sd"} <= set(st), name
            assert isinstance(st["n"], int) and isinstance(st["mean"], float)
