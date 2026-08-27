"""Geometric distances between merge-position hidden states (PLAN.md §5.2).

Everything here operates on `[N, D]` matrices of hidden states taken at the
merge position — the first position after the substituted span. The span
positions themselves are never compared: they have different cardinality in the
two views (6 visual tokens against 1-2 text tokens after Gate 0) and forcing
them to match is a modeling error, not a metric (PLAN.md §1.1).

The distinction this module exists to make: a constant per-modality translation
is harmless if the downstream readout is affine, so raw distance and
offset-free distance are reported separately. Conflating them reports a benign
offset as representational divergence, which is a recurring flaw in how the
modality gap gets measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OffsetStats:
    """The per-modality mean offset, factored out and reported in its own right.

    `norm` is the size of the translation between the two modality clouds;
    testing H3 is asking how much of the raw gap this single vector explains.
    """

    mean_a: torch.Tensor           # [D]
    mean_b: torch.Tensor           # [D]
    norm: float                    # ||E[a] - E[b]||

    @property
    def offset(self) -> torch.Tensor:
        return self.mean_a - self.mean_b


def _as_float(x: torch.Tensor) -> torch.Tensor:
    """bf16 dot products lose too much precision for distances near zero."""
    return x.to(torch.float32)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-item cosine distance in [0, 2]. Shapes `[N, D]` -> `[N]`."""
    a, b = _as_float(a), _as_float(b)
    return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1)


def offset_stats(a: torch.Tensor, b: torch.Tensor) -> OffsetStats:
    """Mean of each modality cloud, over the whole evaluation set.

    Estimate this on the full set rather than per-minibatch: a mean taken over
    32 items is noisy enough to move the offset-free distances around, and the
    offset is meant to be a property of the model, not of the batching.
    """
    a, b = _as_float(a), _as_float(b)
    mean_a, mean_b = a.mean(0), b.mean(0)
    return OffsetStats(mean_a, mean_b, float(torch.linalg.vector_norm(mean_a - mean_b)))


def offset_free_distance(a: torch.Tensor, b: torch.Tensor,
                         stats: OffsetStats | None = None) -> torch.Tensor:
    """Cosine distance after removing each modality's mean.

    What survives here is divergence a linear readout could not undo. If this
    is much smaller than the raw distance, most of the modality gap is a
    translation — the H3 case, and a far more tractable problem than the raw
    number suggests.
    """
    stats = stats or offset_stats(a, b)
    return cosine_distance(_as_float(a) - stats.mean_a, _as_float(b) - stats.mean_b)


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """Linear CKA between two `[N, D]` sets of representations, in [0, 1].

    Complements the per-item distances: CKA asks whether the two clouds have
    the same *shape* — whether items near each other in one modality are near
    each other in the other — which per-item distance cannot see. Invariant to
    isotropic scaling and orthogonal transforms, so it is insensitive to the
    kind of difference the offset-free distance already factors out.
    """
    a, b = _as_float(a), _as_float(b)
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    # ||b^T a||_F^2 / (||a^T a||_F ||b^T b||_F), computed without forming
    # the N x N Gram matrices.
    cross = torch.linalg.matrix_norm(b.T @ a) ** 2
    norm_a = torch.linalg.matrix_norm(a.T @ a)
    norm_b = torch.linalg.matrix_norm(b.T @ b)
    denom = norm_a * norm_b
    if denom <= 0:
        return 0.0
    return float(cross / denom)


def per_layer_distances(
    hidden_a: list[torch.Tensor],
    hidden_b: list[torch.Tensor],
) -> list[dict]:
    """Raw, offset-free, CKA and offset norm for each layer.

    Takes lists of `[N, D]` matrices, one per captured layer. Which layers get
    captured is a sampling decision made upstream — dumping all 25 for every
    item is 20x the disk for nothing the sparse set doesn't show
    (docs/COMPUTE.md).
    """
    if len(hidden_a) != len(hidden_b):
        raise ValueError(f"layer count mismatch: {len(hidden_a)} vs {len(hidden_b)}")
    out = []
    for layer, (a, b) in enumerate(zip(hidden_a, hidden_b)):
        stats = offset_stats(a, b)
        out.append({
            "layer": layer,
            "raw": cosine_distance(a, b),
            "offset_free": offset_free_distance(a, b, stats),
            "cka": linear_cka(a, b),
            "offset_norm": stats.norm,
        })
    return out
