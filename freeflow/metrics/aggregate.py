"""Aggregation: breakdowns, read-back conditioning, bootstrap CIs.

Two things here are load-bearing rather than bookkeeping.

**Read-back conditioning (PLAN.md §5.3a).** A misread span produces a large MSG
for a reason that has nothing to do with representational geometry. Every Tier B
metric is therefore reported both unconditionally and over spans the model reads
correctly, with the read-back rate stated. Gate 0 showed read-back tracks word
*length*, not word *class* — so an unconditioned class breakdown would report
OCR difficulty as if it were a representational gap, which is precisely the
confound H2 must avoid.

**Ratio-of-means on every subgroup.** Subgroups are small, and the mean of
per-item ratios is dominated by whichever item had the smallest denominator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .msg import MSGResult, normalized_msg


@dataclass
class Breakdown:
    """One metric sliced by a grouping, plus the overall value."""

    overall: MSGResult
    groups: dict[str, MSGResult]
    label: str = ""

    def __str__(self) -> str:
        lines = [f"{self.label or 'overall'}: {self.overall}"]
        for name, res in sorted(self.groups.items()):
            lines.append(f"    {name:<12} {res}")
        return "\n".join(lines)


def _select(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return t[mask]


def msg_by_group(
    cross: torch.Tensor,
    within_text: torch.Tensor,
    within_image: torch.Tensor,
    groups: list[str],
    n_boot: int = 2000,
    seed: int = 0,
    min_group: int = 30,
    label: str = "",
) -> Breakdown:
    """MSG overall and per group, skipping groups too small to mean anything.

    `min_group` is not cosmetic: a bootstrap CI over 8 items is wide enough to
    be consistent with almost any hypothesis, and reporting it invites reading
    signal into noise. Groups below the floor are dropped and their absence is
    visible in the output.
    """
    if len(groups) != cross.numel():
        raise ValueError(f"{len(groups)} group labels for {cross.numel()} items")

    overall = normalized_msg(cross, within_text, within_image, n_boot, seed)
    names = sorted(set(groups))
    group_tensor = torch.tensor([names.index(g) for g in groups])

    out: dict[str, MSGResult] = {}
    for i, name in enumerate(names):
        mask = group_tensor == i
        if int(mask.sum()) < min_group:
            continue
        out[name] = normalized_msg(_select(cross, mask), _select(within_text, mask),
                                   _select(within_image, mask), n_boot, seed)
    return Breakdown(overall=overall, groups=out, label=label)


@dataclass
class ConditionedReport:
    """A metric reported both ways, with the read-back rate that separates them."""

    unconditional: Breakdown
    read_correctly: Breakdown | None
    readback_rate: float
    n_total: int
    n_correct: int

    def __str__(self) -> str:
        parts = [f"read-back {self.readback_rate:.3f} "
                 f"({self.n_correct}/{self.n_total})",
                 "unconditional:", str(self.unconditional)]
        if self.read_correctly is not None:
            parts += ["conditioned on correct read-back:", str(self.read_correctly)]
        else:
            parts.append("conditioned: too few correctly-read spans to report")
        return "\n".join(parts)


def conditioned_msg(
    cross: torch.Tensor,
    within_text: torch.Tensor,
    within_image: torch.Tensor,
    groups: list[str],
    read_ok: torch.Tensor,
    n_boot: int = 2000,
    seed: int = 0,
    min_group: int = 30,
) -> ConditionedReport:
    """The reporting contract for every Tier B metric (PLAN.md §5.3a).

    `read_ok` is a boolean mask over items: did the model transcribe this span
    correctly at the frozen render config? At the Gate 0 config that is true for
    ~94-99% of spans, so conditioning discards little — but stating the rate is
    what turns a residual OCR error from an unquantified confound into a
    reported number.
    """
    read_ok = read_ok.bool()
    n_total, n_correct = int(read_ok.numel()), int(read_ok.sum())

    uncond = msg_by_group(cross, within_text, within_image, groups,
                          n_boot, seed, min_group, label="all spans")

    cond = None
    if n_correct >= min_group:
        idx = read_ok
        cond = msg_by_group(
            _select(cross, idx), _select(within_text, idx), _select(within_image, idx),
            [g for g, ok in zip(groups, read_ok.tolist()) if ok],
            n_boot, seed, min_group, label="read correctly")

    return ConditionedReport(
        unconditional=uncond, read_correctly=cond,
        readback_rate=n_correct / max(1, n_total),
        n_total=n_total, n_correct=n_correct)


def bootstrap_ci(values: torch.Tensor, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a `[N]` vector.

    For MSG use `normalized_msg`, which bootstraps the ratio rather than the
    mean of ratios. This is for the plain quantities — JSD, read-back rate,
    benchmark scores.
    """
    x = values.to(torch.float32).flatten()
    n = int(x.numel())
    if n < 2:
        return (float("nan"), float("nan"))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = x[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([alpha / 2, 1 - alpha / 2]))
    return (float(lo), float(hi))
