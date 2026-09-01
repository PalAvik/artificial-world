"""Span-identity probe — the anti-collapse control (PLAN.md §5.4).

Cross-modal invariance is trivially maximised by mapping everything to a
constant. The probe is what makes that cheat visible: if the merge-position
representation still tells you *which* span was substituted, the invariance was
bought by aligning content rather than by discarding it.

Logged on the same chart as MSG at every eval step, because collapse has to be
visible *during* training rather than diagnosed after it. The two together are
the invariance-informativeness frontier, and a method that moves down-and-left
along the existing frontier has achieved nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ProbeResult:
    accuracy: float
    chance: float
    n_train: int
    n_test: int
    n_classes: int
    warning: str | None = None       # set when the split is too thin to mean anything

    @property
    def above_chance(self) -> float:
        """Headroom actually used, so results stay comparable when the number
        of span classes changes between tiers."""
        if self.chance >= 1.0:
            return 0.0
        return (self.accuracy - self.chance) / (1.0 - self.chance)

    def __str__(self) -> str:
        base = (f"probe {self.accuracy:.3f} (chance {self.chance:.3f}, "
                f"{self.above_chance:+.3f} of headroom, "
                f"{self.n_classes} classes, n={self.n_test})")
        return f"{base}  ! {self.warning}" if self.warning else base


def fit_probe(
    hidden: torch.Tensor,
    labels: np.ndarray | list,
    seed: int = 0,
    test_frac: float = 0.3,
    max_iter: int = 2000,
    C: float = 1.0,
) -> ProbeResult:
    """Fit a linear probe recovering span identity from `[N, D]` hidden states.

    Linear on purpose. A deep probe would report what is *recoverable* with
    enough capacity, which is nearly everything; a linear one reports what is
    *available* to the model's own affine readout, which is the question that
    matters for whether the representation still carries the span.

    Every knob here is fixed rather than tuned, because the probe is compared
    across checkpoints: a probe whose regularisation is chosen per checkpoint
    measures the tuning, not the representation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = hidden.detach().to(torch.float32).cpu().numpy()
    y = np.asarray(labels)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x.shape[0]} hidden states for {y.shape[0]} labels")

    # Stratify by hand so every class appears in both halves; sklearn's
    # stratified split refuses classes with a single member, which is common
    # in the tail of a span vocabulary.
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        members = np.flatnonzero(y == cls)
        rng.shuffle(members)
        n_test = max(1, int(round(len(members) * test_frac)))
        if len(members) <= 1:
            # Nothing to hold out: keep it in train so the class still exists
            # for the classifier, and it simply cannot be scored.
            train_idx.extend(members)
            continue
        test_idx.extend(members[:n_test])
        train_idx.extend(members[n_test:])

    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    if test_idx.size == 0:
        raise ValueError("no held-out items: every class has a single member")

    scaler = StandardScaler().fit(x[train_idx])
    # No multi_class kwarg: removed in recent sklearn, and multinomial
    # is the default for multiclass problems anyway.
    clf = LogisticRegression(max_iter=max_iter, C=C)
    clf.fit(scaler.transform(x[train_idx]), y[train_idx])
    acc = float(clf.score(scaler.transform(x[test_idx]), y[test_idx]))

    # Chance is the majority-class rate in the test split, not 1/n_classes:
    # with an unbalanced span vocabulary the latter flatters the probe.
    _, counts = np.unique(y[test_idx], return_counts=True)
    chance = float(counts.max() / counts.sum())

    # Classes with a single member cannot be held out, so a corpus with almost
    # as many spans as items leaves a test split of one or two — where accuracy
    # and chance are both 1.0 and the number says nothing. Seen for real on a
    # 120-item corpus with 119 unique spans.
    warning = None
    scored_classes = int(np.unique(y[test_idx]).size)
    if test_idx.size < 20 or scored_classes < 2:
        warning = (f"degenerate split: {test_idx.size} test items over "
                   f"{scored_classes} scored class(es). Needs several items per "
                   "span — raise n or shrink the span vocabulary.")
    elif chance > 0.9:
        warning = f"chance is {chance:.2f}; the test split is nearly single-class"

    return ProbeResult(accuracy=acc, chance=chance, n_train=int(train_idx.size),
                       n_test=int(test_idx.size),
                       n_classes=int(np.unique(y).size), warning=warning)


def collapse_check(msg_now: float, msg_base: float,
                   probe_now: float, probe_base: float,
                   msg_drop: float = 0.40, probe_tol: float = 0.05) -> dict:
    """Gate 2 conditions (a) and (b) evaluated together (docs/GATES.md).

    They have to be read together. A 40% MSG reduction is only a result if the
    probe held; the same reduction with the probe falling proportionally is
    collapse, and the two are indistinguishable from the MSG number alone.
    """
    reduction = (msg_base - msg_now) / msg_base if msg_base else 0.0
    probe_delta = probe_now - probe_base
    invariance_ok = reduction >= msg_drop
    informative_ok = probe_delta >= -probe_tol

    if invariance_ok and informative_ok:
        verdict = "PASS: invariance gained, span identity retained"
    elif invariance_ok and not informative_ok:
        verdict = ("COLLAPSE: MSG fell but the probe fell with it — invariance "
                   "bought by discarding content. One redesign (raise lambda_g), "
                   "then drop on a second occurrence.")
    elif not invariance_ok and informative_ok:
        verdict = f"INSUFFICIENT: MSG reduction {reduction:.1%} < {msg_drop:.0%}"
    else:
        verdict = "FAIL: no invariance gained and the probe degraded"

    return {"msg_reduction": reduction, "probe_delta": probe_delta,
            "invariance_ok": invariance_ok, "informative_ok": informative_ok,
            "verdict": verdict}
