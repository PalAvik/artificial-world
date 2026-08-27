"""Distributional agreement between the two views (PLAN.md §5.1).

Jensen-Shannon rather than KL for two reasons that both matter here: it is
symmetric, so neither modality is privileged as the reference — the same reason
the training objective alternates its stop-gradient — and it is bounded by
log 2, so items are comparable and a single pathological position cannot
dominate a mean.

**Logits are never persisted.** The vocabulary is 248,320, so one position's
distribution is ~1 MB in fp32 and a full sequence for a batch of 32 is ~8 GB.
Everything here reduces inside the forward pass and returns scalars.
"""
from __future__ import annotations

import torch

LOG2 = 0.6931471805599453


def jensen_shannon(logits_a: torch.Tensor, logits_b: torch.Tensor,
                   base2: bool = True) -> torch.Tensor:
    """JSD between two sets of logits over the last dimension.

    Shapes `[..., V]` -> `[...]`. In bits by default, so the range is [0, 1]
    and "half the maximum possible disagreement" reads directly off the number.

    Computed in float32 from log-probabilities throughout: the mixture
    `m = (p + q) / 2` has to be formed in probability space, but taking
    `log m` via logsumexp rather than `log(exp(...) + exp(...))` keeps the
    small-probability tail — which is most of a 248k vocabulary — from
    underflowing.
    """
    log_p = torch.log_softmax(logits_a.to(torch.float32), dim=-1)
    log_q = torch.log_softmax(logits_b.to(torch.float32), dim=-1)

    # log m = log((p + q)/2) = logsumexp(log p, log q) - log 2
    log_m = torch.logsumexp(torch.stack([log_p, log_q]), dim=0) - LOG2

    kl_pm = torch.sum(log_p.exp() * (log_p - log_m), dim=-1)
    kl_qm = torch.sum(log_q.exp() * (log_q - log_m), dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm)
    # Clamp only the floating-point undershoot at exact equality, where the
    # analytic value is 0 and the sum can land at -1e-9.
    jsd = jsd.clamp(min=0.0)
    return jsd / LOG2 if base2 else jsd


def logits_at(model, hidden: torch.Tensor) -> torch.Tensor:
    """Project hidden states to vocabulary logits for selected positions only.

    The reason to do this by hand rather than reading `outputs.logits`: the
    model computes logits for *every* position, and at batch 32 x sequence 512
    x 248,320 that is ~8 GB of activation for the handful of continuation
    positions the metric actually reads. Passing a `[N, K, D]` slice here keeps
    the allocation proportional to K.
    """
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("model exposes no output embedding layer")
    return head(hidden.to(head.weight.dtype))


class StreamingJSD:
    """Accumulate JSD over many batches without keeping any of them.

    Tracks per-item means so that aggregation (bootstrap CIs, per-class splits,
    conditioning on correct read-back) can happen downstream on a small vector
    rather than on logits.
    """

    def __init__(self) -> None:
        self._per_item: list[torch.Tensor] = []

    def update(self, logits_a: torch.Tensor, logits_b: torch.Tensor,
               mask: torch.Tensor | None = None) -> torch.Tensor:
        """Add one batch. Shapes `[N, K, V]`; `mask` is `[N, K]` over positions.

        Returns this batch's per-item JSD so a caller can log it live.
        """
        jsd = jensen_shannon(logits_a, logits_b)          # [N, K]
        if mask is None:
            per_item = jsd.mean(dim=-1)
        else:
            mask = mask.to(jsd.dtype)
            counts = mask.sum(dim=-1).clamp(min=1.0)
            per_item = (jsd * mask).sum(dim=-1) / counts
        self._per_item.append(per_item.detach().cpu())
        return per_item

    @property
    def per_item(self) -> torch.Tensor:
        """One JSD per evaluated item, in the order they were added."""
        if not self._per_item:
            return torch.empty(0)
        return torch.cat(self._per_item)

    def summary(self) -> dict:
        """Mean plus tail quantiles.

        The tail is reported because it matters more than the mean: a model
        that agrees on most spans and diverges wildly on a few is not the same
        object as one that disagrees mildly everywhere, and averaging hides
        exactly that difference (PLAN.md §5.1).
        """
        x = self.per_item
        if x.numel() == 0:
            return {"n": 0}
        return {
            "n": int(x.numel()),
            "mean": float(x.mean()),
            "median": float(x.median()),
            "p90": float(x.quantile(0.90)),
            "p99": float(x.quantile(0.99)),
            "max": float(x.max()),
        }
