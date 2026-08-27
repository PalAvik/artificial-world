"""Normalized Modality Substitution Gap — the headline number (PLAN.md §5.3).

                    d( h(V_T) , h(V_I) )
    MSG  =  ------------------------------------------
            1/2 [ d(h(V_T), h(V_T')) + d(h(V_I), h(V_I')) ]

A raw cross-modal distance is uninterpretable — is 0.3 large? — so it is
divided by how much the model's own representation moves under a *within*-
modality restatement: a text paraphrase (V_T') and a re-render in a different
font or a different photo (V_I').

    MSG ~ 1   crossing modalities costs no more than rephrasing.
              This is the operational definition of "free-flowing".
    MSG >> 1  modality dominates content.

The denominator is what makes the whole suite credible, which is why building
the within-modality controls is a Phase 0 dependency rather than an afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class MSGResult:
    """Both aggregations, because they answer different questions."""

    ratio_of_means: float
    mean_of_ratios: float
    numerator_mean: float
    denominator_mean: float
    n: int
    ci: tuple[float, float] | None = None
    per_item: torch.Tensor = field(default=None, repr=False)

    def __str__(self) -> str:
        ci = f"  95% CI [{self.ci[0]:.3f}, {self.ci[1]:.3f}]" if self.ci else ""
        return (f"MSG {self.ratio_of_means:.3f}{ci}  "
                f"(cross {self.numerator_mean:.4f} / within "
                f"{self.denominator_mean:.4f}, n={self.n})")


def _finite(x: torch.Tensor) -> torch.Tensor:
    return x[torch.isfinite(x)]


def msg_per_item(
    cross: torch.Tensor,
    within_text: torch.Tensor,
    within_image: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (numerator, denominator) per item, both `[N]`.

    The denominator averages the two within-modality distances rather than
    picking one, so neither modality sets the scale on its own — the same
    symmetry argument that makes the distributional metric Jensen-Shannon
    rather than KL.
    """
    if not (cross.shape == within_text.shape == within_image.shape):
        raise ValueError(
            f"shape mismatch: cross {tuple(cross.shape)}, "
            f"within_text {tuple(within_text.shape)}, "
            f"within_image {tuple(within_image.shape)}")
    denom = 0.5 * (within_text.to(torch.float32) + within_image.to(torch.float32))
    return cross.to(torch.float32), denom.clamp(min=eps)


def normalized_msg(
    cross: torch.Tensor,
    within_text: torch.Tensor,
    within_image: torch.Tensor,
    n_boot: int = 2000,
    seed: int = 0,
    eps: float = 1e-6,
) -> MSGResult:
    """Normalized MSG with a bootstrap CI over items.

    Reports two aggregations deliberately:

    - **ratio_of_means** — the headline. Sums numerator and denominator over
      items before dividing, so an item whose within-modality distance is near
      zero (a paraphrase the model treats as identical) contributes its small
      numerator and its small denominator, rather than a huge ratio.
    - **mean_of_ratios** — the naive per-item average. Reported alongside
      because a large divergence between the two is diagnostic: it means the
      result is being driven by a handful of items with tiny denominators, and
      the headline should not be trusted without looking at them.

    The CI resamples items, so it captures item-level variation. It does not
    capture seed or checkpoint variation — those need separate runs, and Gate 3
    requires three seeds for exactly that reason.
    """
    num, den = msg_per_item(cross, within_text, within_image, eps)
    ratios = _finite(num / den)
    n = int(num.numel())
    if n == 0:
        raise ValueError("no items to aggregate")

    point = float(num.sum() / den.sum())

    ci = None
    if n_boot > 0 and n > 1:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, n, (n_boot, n), generator=g)
        boots = num[idx].sum(dim=1) / den[idx].sum(dim=1)
        boots = _finite(boots)
        if boots.numel():
            lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975]))
            ci = (float(lo), float(hi))

    return MSGResult(
        ratio_of_means=point,
        mean_of_ratios=float(ratios.mean()) if ratios.numel() else float("nan"),
        numerator_mean=float(num.mean()),
        denominator_mean=float(den.mean()),
        n=n,
        ci=ci,
        per_item=ratios,
    )


def gate1_verdict(result: MSGResult, pass_at: float = 1.5,
                  drop_below: float = 1.25) -> str:
    """Translate an MSG into the Gate 1 branch it implies (docs/GATES.md).

    Deliberately mechanical. The point of writing thresholds down before the
    numbers exist is that the call does not get re-argued once they do.
    """
    if result.ci is None:
        return "INCONCLUSIVE: no confidence interval; re-run with n_boot > 0"
    lo, hi = result.ci
    if result.ratio_of_means >= pass_at and lo > drop_below:
        return f"PASS: MSG {result.ratio_of_means:.2f}, CI lower {lo:.2f} > {drop_below}"
    if hi < drop_below:
        return (f"DROP CANDIDATE: CI upper {hi:.2f} < {drop_below} — no gap to "
                "train toward on this tier")
    return (f"MARGINAL: MSG {result.ratio_of_means:.2f}, CI [{lo:.2f}, {hi:.2f}] — "
            "one week of config-sensitivity testing, no second extension")
