"""CPU tests for the metric suite.

The math is separated from the model plumbing precisely so it can be tested
without a GPU. What matters most here is not that the functions run, but that
they have the *properties the plan relies on*: offset-free distance really is
blind to a translation, JSD really is bounded and symmetric, and MSG really
equals 1 when crossing modalities costs exactly what rephrasing costs.

    python -m pytest tests/test_metrics.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from freeflow.metrics import aggregate, distribution, geometry, msg, probe, runner

torch.manual_seed(0)


# --------------------------------------------------------------- geometry ---

class TestCosineDistance:
    def test_identical_vectors_are_zero(self):
        a = torch.randn(16, 32)
        assert torch.allclose(geometry.cosine_distance(a, a),
                              torch.zeros(16), atol=1e-5)

    def test_orthogonal_is_one_and_opposite_is_two(self):
        a = torch.tensor([[1.0, 0.0]])
        assert geometry.cosine_distance(a, torch.tensor([[0.0, 1.0]])).item() \
            == pytest.approx(1.0, abs=1e-6)
        assert geometry.cosine_distance(a, -a).item() == pytest.approx(2.0, abs=1e-6)

    def test_bf16_input_does_not_lose_the_near_zero_regime(self):
        a = torch.randn(8, 64)
        b = a + 1e-3 * torch.randn(8, 64)
        d = geometry.cosine_distance(a.bfloat16(), b.bfloat16())
        assert torch.all(d >= 0) and torch.all(d < 0.1)


class TestOffsetFree:
    def test_a_pure_translation_has_zero_offset_free_distance(self):
        """The property the whole H3 test rests on.

        If the image cloud is the text cloud shifted by a constant, a linear
        readout could undo it entirely — so the offset-free distance must see
        nothing, while the raw distance sees the shift.
        """
        a = torch.randn(64, 32)
        b = a + torch.tensor([5.0] + [0.0] * 31)
        assert geometry.offset_free_distance(a, b).mean() < 1e-4
        assert geometry.cosine_distance(a, b).mean() > 0.05

    def test_offset_norm_matches_the_translation(self):
        a = torch.randn(128, 16)
        shift = torch.full((16,), 0.25)
        stats = geometry.offset_stats(a + shift, a)
        assert float(stats.norm) == pytest.approx(float(shift.norm()), rel=1e-4)

    def test_genuine_divergence_survives_offset_removal(self):
        a = torch.randn(64, 32)
        b = torch.randn(64, 32)
        assert geometry.offset_free_distance(a, b).mean() > 0.5


class TestCKA:
    def test_identical_representations_give_one(self):
        a = torch.randn(64, 16)
        assert geometry.linear_cka(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_invariant_to_isotropic_scaling_and_rotation(self):
        a = torch.randn(64, 16)
        q, _ = torch.linalg.qr(torch.randn(16, 16))
        assert geometry.linear_cka(a, 3.7 * a @ q) == pytest.approx(1.0, abs=1e-4)

    def test_unrelated_representations_score_low(self):
        assert geometry.linear_cka(torch.randn(256, 16), torch.randn(256, 16)) < 0.25


def test_per_layer_distances_rejects_mismatched_layer_counts():
    with pytest.raises(ValueError, match="layer count"):
        geometry.per_layer_distances([torch.randn(4, 8)] * 3, [torch.randn(4, 8)] * 2)


# ----------------------------------------------------------- distribution ---

class TestJSD:
    def test_identical_distributions_are_zero(self):
        z = torch.randn(8, 100)
        assert torch.allclose(distribution.jensen_shannon(z, z),
                              torch.zeros(8), atol=1e-6)

    def test_never_negative_at_exact_equality(self):
        # Floating point can land the analytic zero at -1e-9; clamping is why.
        z = torch.randn(64, 500)
        assert torch.all(distribution.jensen_shannon(z, z) >= 0.0)

    def test_disjoint_support_saturates_at_one_bit(self):
        big = 40.0
        a = torch.tensor([[big, -big]])
        b = torch.tensor([[-big, big]])
        assert distribution.jensen_shannon(a, b).item() == pytest.approx(1.0, abs=1e-3)

    def test_is_symmetric(self):
        a, b = torch.randn(16, 200), torch.randn(16, 200)
        assert torch.allclose(distribution.jensen_shannon(a, b),
                              distribution.jensen_shannon(b, a), atol=1e-6)

    def test_stays_within_bounds_over_random_inputs(self):
        a, b = torch.randn(64, 300) * 5, torch.randn(64, 300) * 5
        d = distribution.jensen_shannon(a, b)
        assert torch.all(d >= 0) and torch.all(d <= 1.0 + 1e-5)

    def test_nats_option_is_smaller_by_log2(self):
        a, b = torch.randn(8, 50), torch.randn(8, 50)
        bits = distribution.jensen_shannon(a, b, base2=True)
        nats = distribution.jensen_shannon(a, b, base2=False)
        assert torch.allclose(nats, bits * distribution.LOG2, atol=1e-6)


class TestStreamingJSD:
    def test_accumulates_one_value_per_item_across_batches(self):
        s = distribution.StreamingJSD()
        for _ in range(3):
            s.update(torch.randn(4, 5, 50), torch.randn(4, 5, 50))
        assert s.per_item.shape == (12,)
        assert s.summary()["n"] == 12

    def test_mask_excludes_padded_positions(self):
        a, b = torch.randn(2, 4, 30), torch.randn(2, 4, 30)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        masked = s_masked = distribution.StreamingJSD().update(a, b, mask)
        unmasked = distribution.StreamingJSD().update(a, b)
        # Row 1 is fully attended, so masking cannot change it.
        assert masked[1].item() == pytest.approx(unmasked[1].item(), abs=1e-6)
        # Row 0 drops two positions, so it generally does change.
        assert s_masked[0].item() != pytest.approx(unmasked[0].item(), abs=1e-9)

    def test_empty_summary_does_not_crash(self):
        assert distribution.StreamingJSD().summary() == {"n": 0}

    def test_summary_reports_the_tail_not_only_the_mean(self):
        s = distribution.StreamingJSD()
        s.update(torch.randn(64, 3, 40), torch.randn(64, 3, 40))
        keys = s.summary()
        assert {"mean", "median", "p90", "p99", "max"} <= set(keys)


# -------------------------------------------------------------------- MSG ---

class TestNormalizedMSG:
    def test_equals_one_when_crossing_costs_what_rephrasing_costs(self):
        """The calibration of the headline number: MSG ~ 1 is the target."""
        d = torch.rand(200) + 0.1
        r = msg.normalized_msg(d, d, d, n_boot=200)
        assert r.ratio_of_means == pytest.approx(1.0, abs=1e-5)

    def test_scales_linearly_with_the_cross_modal_distance(self):
        d = torch.rand(200) + 0.1
        r = msg.normalized_msg(2.0 * d, d, d, n_boot=0)
        assert r.ratio_of_means == pytest.approx(2.0, abs=1e-5)

    def test_denominator_averages_the_two_controls(self):
        cross = torch.full((50,), 1.0)
        r = msg.normalized_msg(cross, torch.full((50,), 1.0),
                               torch.full((50,), 3.0), n_boot=0)
        assert r.denominator_mean == pytest.approx(2.0, abs=1e-6)

    def test_ratio_of_means_resists_a_near_zero_denominator(self):
        """Why the headline is ratio-of-means and not mean-of-ratios."""
        cross = torch.full((100,), 0.5)
        within = torch.full((100,), 0.5)
        within[0] = 1e-6                      # one paraphrase the model sees as identical
        r = msg.normalized_msg(cross, within, within, n_boot=0)
        assert r.ratio_of_means == pytest.approx(1.0, rel=0.05)
        assert r.mean_of_ratios > 100          # the naive average is destroyed

    def test_ci_brackets_the_point_estimate(self):
        cross = torch.rand(300) * 2
        within = torch.rand(300) + 0.5
        r = msg.normalized_msg(cross, within, within, n_boot=500, seed=1)
        assert r.ci is not None
        assert r.ci[0] <= r.ratio_of_means <= r.ci[1]

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            msg.normalized_msg(torch.rand(10), torch.rand(9), torch.rand(10))

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="no items"):
            msg.normalized_msg(torch.empty(0), torch.empty(0), torch.empty(0))


class TestGate1Verdict:
    def _res(self, point, lo, hi):
        return msg.MSGResult(point, point, 0.0, 0.0, 100, (lo, hi))

    def test_pass_needs_both_the_point_and_the_ci_lower_bound(self):
        assert msg.gate1_verdict(self._res(1.8, 1.5, 2.1)).startswith("PASS")

    def test_a_high_point_with_a_low_ci_is_only_marginal(self):
        assert msg.gate1_verdict(self._res(1.6, 1.1, 2.4)).startswith("MARGINAL")

    def test_ci_entirely_below_the_floor_is_a_drop_candidate(self):
        assert msg.gate1_verdict(self._res(1.05, 0.95, 1.15)).startswith("DROP")

    def test_missing_ci_is_inconclusive_not_a_pass(self):
        r = msg.MSGResult(2.0, 2.0, 0.0, 0.0, 10, None)
        assert msg.gate1_verdict(r).startswith("INCONCLUSIVE")


# ------------------------------------------------------------------ probe ---

class TestProbe:
    def test_recovers_span_identity_from_separable_representations(self):
        labels = np.repeat(np.arange(5), 40)
        centres = torch.eye(5)[labels] * 8.0
        h = centres + torch.randn(200, 5) * 0.2
        assert probe.fit_probe(h, labels, seed=0).accuracy > 0.9

    def test_scores_near_chance_when_the_representation_is_uninformative(self):
        """The collapse signature: invariant but content-free."""
        labels = np.repeat(np.arange(5), 40)
        r = probe.fit_probe(torch.randn(200, 5) * 0.01, labels, seed=0)
        assert r.accuracy < 0.45

    def test_chance_uses_the_majority_class_not_one_over_k(self):
        labels = np.array([0] * 90 + [1] * 10)
        r = probe.fit_probe(torch.randn(100, 4), labels, seed=0)
        assert r.chance > 0.6            # majority baseline, not 0.5

    def test_flags_a_degenerate_split_instead_of_reporting_1_point_0(self):
        """Seen for real: a 120-item corpus with 119 unique spans left a test
        split of one item, where accuracy and chance are both 1.0 and the
        number is meaningless."""
        labels = np.arange(60)                 # every class has exactly one member
        labels = np.concatenate([labels, [0]])  # ...except one
        r = probe.fit_probe(torch.randn(61, 6), labels, seed=0)
        assert r.warning is not None and "degenerate" in r.warning

    def test_no_warning_when_every_class_is_well_populated(self):
        labels = np.repeat(np.arange(6), 30)
        assert probe.fit_probe(torch.randn(180, 6), labels, seed=0).warning is None

    def test_rejects_mismatched_labels(self):
        with pytest.raises(ValueError, match="hidden states for"):
            probe.fit_probe(torch.randn(10, 4), np.arange(9))


class TestCollapseCheck:
    def test_invariance_with_retained_probe_passes(self):
        r = probe.collapse_check(0.9, 2.0, 0.80, 0.82)
        assert r["verdict"].startswith("PASS")

    def test_invariance_with_a_falling_probe_is_named_collapse(self):
        r = probe.collapse_check(0.9, 2.0, 0.50, 0.82)
        assert r["verdict"].startswith("COLLAPSE")

    def test_a_small_msg_reduction_is_insufficient_not_collapse(self):
        r = probe.collapse_check(1.9, 2.0, 0.81, 0.82)
        assert r["verdict"].startswith("INSUFFICIENT")


# -------------------------------------------------------------- aggregate ---

class TestAggregate:
    def _data(self, n=200):
        return torch.rand(n) + 0.5, torch.rand(n) + 0.5, torch.rand(n) + 0.5

    def test_small_groups_are_dropped_rather_than_reported_noisily(self):
        cross, wt, wi = self._data(100)
        groups = ["big"] * 95 + ["tiny"] * 5
        b = aggregate.msg_by_group(cross, wt, wi, groups, n_boot=50, min_group=30)
        assert "big" in b.groups and "tiny" not in b.groups

    def test_group_label_count_must_match_items(self):
        cross, wt, wi = self._data(10)
        with pytest.raises(ValueError, match="group labels"):
            aggregate.msg_by_group(cross, wt, wi, ["a"] * 9, n_boot=0)

    def test_conditioning_reports_the_readback_rate_and_both_views(self):
        cross, wt, wi = self._data(200)
        read_ok = torch.zeros(200, dtype=torch.bool)
        read_ok[:180] = True
        rep = aggregate.conditioned_msg(cross, wt, wi, ["all"] * 200, read_ok,
                                        n_boot=50, min_group=30)
        assert rep.readback_rate == pytest.approx(0.9)
        assert rep.n_correct == 180
        assert rep.read_correctly is not None
        assert rep.read_correctly.overall.n == 180
        assert rep.unconditional.overall.n == 200

    def test_conditioning_is_skipped_when_too_few_spans_read_correctly(self):
        cross, wt, wi = self._data(100)
        read_ok = torch.zeros(100, dtype=torch.bool)
        read_ok[:5] = True
        rep = aggregate.conditioned_msg(cross, wt, wi, ["all"] * 100, read_ok,
                                        n_boot=20, min_group=30)
        assert rep.read_correctly is None

    def test_bootstrap_ci_brackets_the_mean(self):
        x = torch.randn(500) + 3.0
        lo, hi = aggregate.bootstrap_ci(x, n_boot=400, seed=0)
        assert lo < float(x.mean()) < hi


# ----------------------------------------------------------------- runner ---

class TestLayerSelection:
    def test_always_includes_the_final_layer(self):
        for n in (5, 13, 25, 41):
            assert runner.default_layers(n)[-1] == n - 1

    def test_returns_every_layer_when_there_are_few(self):
        assert runner.default_layers(6, k=8) == list(range(6))

    def test_spans_the_stack_without_duplicates(self):
        idx = runner.default_layers(25, k=8)
        assert idx[0] == 0 and len(idx) == len(set(idx)) and idx == sorted(idx)


class TestMergePositionGather:
    def test_pulls_the_state_at_each_row_s_own_merge_index(self):
        n, length, d = 4, 7, 3
        # Encode position in the values so a wrong index is visible.
        hidden = torch.arange(n * length * d, dtype=torch.float32).reshape(n, length, d)
        merge = torch.tensor([0, 2, 4, 6])
        got = runner.gather_merge_hidden([hidden], merge, [0])[0]
        for row in range(n):
            assert torch.equal(got[row], hidden[row, merge[row]])

    def test_handles_layers_of_differing_width(self):
        hs = [torch.randn(3, 5, 8), torch.randn(3, 5, 16)]
        out = runner.gather_merge_hidden(hs, torch.tensor([1, 2, 3]), [0, 1])
        assert out[0].shape == (3, 8) and out[1].shape == (3, 16)


class TestContinuationLogits:
    class _Head(torch.nn.Module):
        def __init__(self, d, v):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(v, d))

        def forward(self, x):
            return x @ self.weight.T

    class _Model:
        def __init__(self, head):
            self._head = head

        def get_output_embeddings(self):
            return self._head

    def test_slices_the_positions_that_predict_the_suffix(self):
        n, length, d, v, k = 2, 10, 4, 7, 3
        head = self._Head(d, v)
        model = self._Model(head)
        hidden = torch.randn(n, length, d)
        merge = torch.tensor([4, 6])
        out = runner.continuation_logits(model, hidden, merge, k)
        assert out.shape == (n, k, v)
        # Position i predicts token i+1, so the slice starts at merge-1.
        for row, m in enumerate(merge.tolist()):
            expected = head(hidden[row, m - 1: m - 1 + k])
            assert torch.allclose(out[row], expected, atol=1e-5)

    def test_merge_index_zero_does_not_index_backwards(self):
        head = self._Head(4, 5)
        out = runner.continuation_logits(self._Model(head), torch.randn(1, 6, 4),
                                         torch.tensor([0]), 2)
        assert out.shape == (1, 2, 5)

    def test_missing_output_head_raises_rather_than_returning_nonsense(self):
        class _NoHead:
            def get_output_embeddings(self):
                return None
        with pytest.raises(ValueError, match="output embedding"):
            runner.continuation_logits(_NoHead(), torch.randn(1, 4, 3),
                                       torch.tensor([1]), 2)


class TestCrossValidatedMap:
    """The null one level above the offset: is the gap a change of basis?"""

    def test_a_rotation_is_undone_and_a_real_difference_is_not(self):
        torch.manual_seed(0)
        n, d = 400, 16
        text = torch.randn(n, d)
        q, _ = torch.linalg.qr(torch.randn(d, d))       # a true rotation
        rotated = text @ q
        mapped, _, fit = geometry.cross_validated_map(rotated, text,
                                                      kind="orthogonal")
        assert fit.kind == "orthogonal"
        assert geometry.cosine_distance(mapped, text).mean() < 0.05
        # Independent noise is not a rotation of anything, and must survive.
        noise = torch.randn(n, d)
        mapped_noise, _, _ = geometry.cross_validated_map(noise, text,
                                                          kind="orthogonal")
        assert geometry.cosine_distance(mapped_noise, text).mean() > 0.5

    def test_every_row_is_predicted_out_of_fold(self):
        """A map fitted and scored on the same rows drives any distance to zero
        given enough dimensions, which would prove nothing. Here d > n per fold,
        so an in-fold fit would fit perfectly and the test would catch it."""
        torch.manual_seed(0)
        n, d = 40, 64
        source, target = torch.randn(n, d), torch.randn(n, d)
        mapped, _, _ = geometry.cross_validated_map(source, target,
                                                    kind="linear", folds=5)
        # Unrelated data: held out, the map cannot help.
        assert geometry.cosine_distance(mapped, target).mean() > 0.5

    def test_the_control_rides_the_same_fold_map(self):
        """Mapping the numerator while leaving the denominator alone would
        manufacture a collapse in the ratio."""
        torch.manual_seed(0)
        n, d = 200, 16
        text = torch.randn(n, d)
        q, _ = torch.linalg.qr(torch.randn(d, d))
        image = text @ q
        image_ctl = (text + 0.05 * torch.randn(n, d)) @ q
        mapped, extra, _ = geometry.cross_validated_map(
            image, text, also=[image_ctl], kind="orthogonal")
        # The within-image control stays close to the mapped image view, so the
        # MSG denominator survives the mapping intact.
        assert geometry.cosine_distance(mapped, extra[0]).mean() < 0.15

    def test_an_underdetermined_fit_reports_itself(self):
        """The failure mode that would favour continuing the project: fitted
        from fewer rows than dimensions, a map fails out-of-fold whatever the
        truth is, and the failure is indistinguishable from an irreducible
        gap."""
        torch.manual_seed(0)
        thin = geometry.cross_validated_map(torch.randn(40, 64),
                                            torch.randn(40, 64),
                                            kind="linear")[2]
        assert thin.per_dim < 1
        assert "neither verdict is available" in thin.underdetermined()

        fat = geometry.cross_validated_map(torch.randn(400, 16),
                                           torch.randn(400, 16),
                                           kind="linear")[2]
        assert fat.per_dim >= 2 and fat.underdetermined() is None

    def test_repeated_content_does_not_count_as_sample_size(self):
        """Thirteen renderings of one span are thirteen rows and one
        constraint. Counting rows is what let an n=8000 run clear the guard
        while resting on 150 distinct spans."""
        torch.manual_seed(0)
        d, n_spans, reps = 32, 10, 40          # 400 rows, 10 distinct spans
        groups = [f"s{i}" for i in range(n_spans) for _ in range(reps)]
        fit = geometry.cross_validated_map(torch.randn(400, d),
                                           torch.randn(400, d),
                                           kind="linear", groups=groups)[2]
        assert fit.effective_n == n_spans and fit.rows_per_dim > 2
        assert "distinct spans" in fit.underdetermined()
        assert "more renderings of the same words will not help" in \
            fit.underdetermined()

    def test_the_penalty_is_chosen_to_favour_the_map(self):
        """The grid is scored out-of-fold and the best penalty wins, so the null
        gets its best shot. A gap that survives a penalty picked in its own
        favour is the only kind worth reporting."""
        torch.manual_seed(0)
        n, d = 300, 16
        target = torch.randn(n, d)
        source = target + 0.3 * torch.randn(n, d)
        best = geometry.cross_validated_map(source, target, kind="linear")
        fixed = geometry.cross_validated_map(source, target, kind="linear",
                                             ridge=1e4)
        assert (geometry.cosine_distance(best[0], target).mean()
                <= geometry.cosine_distance(fixed[0], target).mean() + 1e-6)
        assert best[2].ridge in geometry.RIDGE_GRID

    def test_a_rowwise_split_memorises_repeated_spans(self):
        """The bug that produced linear-map-free MSG 0.006 on 2026-08-31.

        Each span appears many times with different renderings. Split by row,
        a held-out row has near-twins in training and the map memorises; split
        by span, it must actually generalise. The corpus is built so that no
        linear map relates the two modalities, so *any* low distance here is
        leakage rather than structure.
        """
        torch.manual_seed(0)
        n_spans, reps, d = 40, 13, 64   # d >= n_spans: memorisable
        # One arbitrary text state and one arbitrary image state per span, with
        # no linear relation between them; repetitions differ only by noise.
        text_of = torch.randn(n_spans, d)
        image_of = torch.randn(n_spans, d)
        groups, text, image = [], [], []
        for s_i in range(n_spans):
            for _ in range(reps):
                groups.append(f"span{s_i}")
                text.append(text_of[s_i] + 0.01 * torch.randn(d))
                image.append(image_of[s_i] + 0.01 * torch.randn(d))
        text, image = torch.stack(text), torch.stack(image)

        rowwise = geometry.cross_validated_map(image, text, kind="linear")
        grouped = geometry.cross_validated_map(image, text, kind="linear",
                                               groups=groups)
        d_row = float(geometry.cosine_distance(rowwise[0], text).mean())
        d_grp = float(geometry.cosine_distance(grouped[0], text).mean())
        # Row-wise looks like a near-perfect map; grouped reveals there is none.
        assert d_row < 0.1, d_row
        assert d_grp > 0.5, d_grp
        assert rowwise[2].leakage() is not None
        assert grouped[2].leakage() is None

    def test_refuses_to_fit_when_content_does_not_vary(self):
        with pytest.raises(ValueError, match="cannot fill"):
            geometry.cross_validated_map(torch.randn(50, 8), torch.randn(50, 8),
                                         groups=["a"] * 25 + ["b"] * 25)

    def test_refuses_to_fit_on_too_few_items(self):
        with pytest.raises(ValueError, match="at least"):
            geometry.cross_validated_map(torch.randn(3, 8), torch.randn(3, 8))

    def test_a_collapsing_penalty_is_not_selected(self):
        """The 2026-08-31 failure. Minimising distance to the target alone is
        won by a map predicting its centroid for everything, which destroys the
        structure a change of basis must preserve -- and shrinks the mapped half
        of the MSG denominator along with the numerator."""
        torch.manual_seed(0)
        n, d = 600, 24
        target = torch.randn(n, d)
        source = torch.randn(n, d)                 # no linear relation at all
        control = source + 0.15 * torch.randn(n, d)
        groups = [f"g{i}" for i in range(n)]

        _, _, fit = geometry.cross_validated_map(
            source, target, also=[control], kind="linear", groups=groups)
        assert fit.control_retention >= 0.5
        assert fit.collapsed() is None

        # With the constraint switched off, the heaviest penalty wins and takes
        # the control down with it.
        _, _, loose = geometry.cross_validated_map(
            source, target, also=[control], kind="linear", groups=groups,
            preserve=0.0)
        assert loose.ridge >= fit.ridge

    def test_an_orthogonal_map_cannot_collapse(self):
        """Procrustes preserves angles, so rotation-free MSG is immune to the
        failure that invalidated the linear number."""
        torch.manual_seed(0)
        n, d = 400, 16
        target = torch.randn(n, d)
        q, _ = torch.linalg.qr(torch.randn(d, d))
        source = target @ q
        control = (target + 0.1 * torch.randn(n, d)) @ q
        _, _, fit = geometry.cross_validated_map(
            source, target, also=[control], kind="orthogonal",
            groups=[f"g{i}" for i in range(n)])
        assert abs(fit.control_retention - 1.0) < 0.02
        assert fit.collapsed() is None

    def test_collapse_is_reported_when_no_penalty_survives(self):
        """A grid on which every penalty collapses must return a rejectable
        result, not silently pick the least-bad one."""
        fit = geometry.MapFit(kind="linear", folds=5, train_n=6400, dim=2048,
                              ridge=100.0, n_groups=8000,
                              control_retention=0.08)
        assert "retains 8%" in fit.collapsed()
        assert "discarding structure" in fit.collapsed()


class TestImagePlaceholder:
    """Which string marks an image, per processor.

    Hardcoding Qwen's confined the instrument to one family without saying so:
    the first multi-model sweep returned five Qwen rows and one failure, and the
    failure was llava, whose placeholder is `<image>`.
    """

    def test_uses_the_processors_own_token(self):
        class P:
            image_token = "<image>"

            class tokenizer:
                @staticmethod
                def get_vocab():
                    return {"<image>": 32000}
        text, tid = runner.image_placeholder(P())
        assert text == "<image>" and tid == 32000

    def test_wraps_a_qwen_style_token_in_its_vision_markers(self):
        class P:
            image_token = "<|image_pad|>"

            class tokenizer:
                @staticmethod
                def get_vocab():
                    return {"<|image_pad|>": 5, "<|vision_start|>": 6,
                            "<|vision_end|>": 7}
        text, tid = runner.image_placeholder(P())
        assert text == "<|vision_start|><|image_pad|><|vision_end|>"
        assert tid == 5

    def test_falls_back_when_the_processor_names_no_token(self):
        class P:
            pass
        text, tid = runner.image_placeholder(P())
        assert "<|image_pad|>" in text and tid is None
