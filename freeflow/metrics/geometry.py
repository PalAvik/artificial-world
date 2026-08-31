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
from typing import Sequence

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


# Penalties tried when fitting a linear map. Spans six orders of magnitude
# because the right scale depends on the hidden states' norm, which varies by
# layer -- the final layer's is ~50x the mid-stack's.
RIDGE_GRID = (1e-4, 1e-2, 1.0, 1e2, 1e4)


@dataclass(frozen=True)
class MapFit:
    """A fitted map plus the fold structure it was fitted under."""

    kind: str
    folds: int
    train_n: int
    dim: int = 0
    ridge: float = 0.0

    @property
    def rows_per_dim(self) -> float:
        return self.train_n / self.dim if self.dim else 0.0

    def underdetermined(self, need: float = 2.0) -> str | None:
        """None when the fit had enough data to be believed.

        A `[D, D]` map has D^2 free parameters. Fitted from fewer rows than
        dimensions it cannot generalise out-of-fold whatever the truth is, and
        the failure looks exactly like an irreducible gap — the finding that
        happens to favour continuing the project. So it is checked and reported
        rather than left to be read off a number.
        """
        if self.rows_per_dim >= need:
            return None
        return (f"{self.train_n} training rows for {self.dim} dimensions "
                f"({self.rows_per_dim:.2f} per dim, want >= {need:.0f}). The map "
                "is under-determined, so a surviving gap is not evidence of one: "
                "raise --n until this clears")


def _fit_map(source: torch.Tensor, target: torch.Tensor, kind: str,
             ridge: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(A, mean_s, mean_t)` for `x -> (x - mean_s) @ A + mean_t`."""
    mean_s, mean_t = source.mean(0), target.mean(0)
    x, y = source - mean_s, target - mean_t
    if kind == "orthogonal":
        # Orthogonal Procrustes: min ||xA - y|| over A^T A = I, so A = U V^T
        # from the SVD of x^T y. The nearest *rotation*, nothing more.
        u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
        return u @ vh, mean_s, mean_t
    if kind == "linear":
        # Ridge, so a fit on fewer items than dimensions stays defined. The
        # penalty is what keeps this a claim about structure rather than about
        # memorising the training fold.
        d = x.shape[1]
        gram = x.T @ x + ridge * torch.eye(d, dtype=x.dtype, device=x.device)
        return torch.linalg.solve(gram, x.T @ y), mean_s, mean_t
    raise ValueError(f"unknown map kind {kind!r}; expected orthogonal or linear")


def cross_validated_map(source: torch.Tensor, target: torch.Tensor,
                        also: Sequence[torch.Tensor] = (),
                        kind: str = "orthogonal", folds: int = 5,
                        ridge: float | Sequence[float] = RIDGE_GRID,
                        seed: int = 0
                        ) -> tuple[torch.Tensor, list[torch.Tensor], MapFit]:
    """Map `source` toward `target`, every row predicted out-of-fold.

    The generalisation of `offset_free_distance`, one step up a hierarchy of
    nulls: a constant offset is a translation, this is a rotation
    (`orthogonal`) or an arbitrary linear change of basis (`linear`). Each asks
    the same question at a different strength — *is the modality gap a
    reversible re-expression of the same information, or is the information
    itself different?*

    This matters because a high `linear_cka` alongside a large raw distance is
    precisely the signature of a rotation: CKA is invariant to orthogonal
    transforms, so it reports two clouds as identically shaped while per-item
    distance reports them as far apart. Only fitting the map distinguishes
    "differently oriented" from "differently informed".

    **Held out, necessarily.** A map fitted and evaluated on the same items can
    drive any distance to zero given enough dimensions, which would prove
    nothing at all. Every row here is predicted by a map that never saw it.

    `also` carries additional matrices — the within-modality control — through
    the *same* fold's map, so the MSG ratio stays coherent: mapping the
    numerator while leaving the denominator alone would manufacture a collapse.

    **`ridge` may be a grid**, in which case the penalty is chosen by the
    out-of-fold distance it achieves. That is mildly optimistic for the map —
    the same held-out rows pick the penalty and score it — and deliberately so:
    the claim being tested is that the gap *survives* a linear map, so every
    thumb on the scale should favour the map. A gap that survives a penalty
    chosen in its own favour is the only kind worth reporting.
    """
    source, target = _as_float(source), _as_float(target)
    also = [_as_float(x) for x in also]
    n = source.shape[0]
    if n < folds or n < 2:
        raise ValueError(f"need at least {max(folds, 2)} items to fit a "
                         f"cross-validated map; got {n}")

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    folds_idx = []
    for f in range(folds):
        test_idx = perm[f::folds]
        mask = torch.ones(n, dtype=torch.bool)
        mask[test_idx] = False
        folds_idx.append((torch.arange(n)[mask], test_idx))

    # Procrustes has no penalty to tune; a grid would just repeat the same fit.
    grid = ([0.0] if kind == "orthogonal"
            else ([ridge] if isinstance(ridge, (int, float)) else list(ridge)))

    best = None
    for lam in grid:
        out = torch.empty_like(source)
        out_also = [torch.empty_like(x) for x in also]
        for train_idx, test_idx in folds_idx:
            A, mean_s, mean_t = _fit_map(source[train_idx], target[train_idx],
                                         kind, float(lam))
            out[test_idx] = (source[test_idx] - mean_s) @ A + mean_t
            for j, x in enumerate(also):
                out_also[j][test_idx] = (x[test_idx] - mean_s) @ A + mean_t
        score = float(cosine_distance(out, target).mean())
        if best is None or score < best[0]:
            best = (score, out, out_also, float(lam))

    _, out, out_also, lam = best
    train_n = min(int(t.numel()) for t, _ in folds_idx)
    return out, out_also, MapFit(kind=kind, folds=folds, train_n=train_n,
                                 dim=int(source.shape[1]), ridge=lam)


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
