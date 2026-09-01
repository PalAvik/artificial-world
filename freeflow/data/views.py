"""The paired-view contract: one item, one span, four renderings.

    V_T   the span as text tokens
    V_I   the span as image tokens
    V_T'  a within-modality text control
    V_I'  a within-modality image control

The controls are the MSG denominator, so what counts as a "control" sets the
scale of the headline number. That choice is made here, once, and deserves to
be stated plainly rather than buried.

**The image control is the same content in a different font.** Uncontroversial.

**The text control is the same content in a different surface form** — by
default a capitalisation variant. This is the tight analogue of a font change:
same string, different surface. The obvious alternative, a synonym, is *not*
content-identical and is undefined for exactly the class the project cares most
about — there is no synonym for "the". Capitalisation is defined for every span
in the vocabulary.

The trade-off, stated because it is real: capitalisation mid-sentence is
semantically marked, so it may move the representation for reasons beyond
surface form, inflating the denominator and *deflating* MSG. That errs toward
not finding a gap, which is the safe direction, but it means:

    **MSG's absolute value is relative to the control convention.** Only
    comparisons under a fixed control are meaningful, and the control must be
    frozen alongside the render config for the whole project.

`ControlKind.SYNONYM` is available for content words where a synonym exists,
and mixing the two within one comparison is what the freeze exists to prevent.
"""
from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence


class ControlKind(str, enum.Enum):
    """How the within-modality text control is formed."""

    SURFACE = "surface"      # capitalisation variant — defined for every span
    SYNONYM = "synonym"      # content word only, and not content-identical


@dataclass
class SpanItem:
    """One evaluation item: a context whose span is expressed four ways.

    `prefix` and `suffix` are shared verbatim across all four views — only the
    span differs. The suffix must be non-empty: it is where the merge position
    lives and what the JSD is scored over.
    """

    prefix: str
    span_text: str
    suffix: str
    span_image: object                # PIL.Image of the span
    span_paraphrase: str              # V_T'
    span_image_alt: object            # V_I'
    span_id: str                      # probe label — the content, not the surface
    group: str = "all"                # word class, or relation type
    tier: str = "B"
    read_ok: bool = True              # set after the read-back pass
    meta: dict = field(default_factory=dict)


# Short suffixes on purpose. The JSD is scored over the shared continuation,
# and its first position — what the model expects immediately after the span —
# is the one that actually depends on the span. A long continuation dilutes
# that signal with tokens the span barely influences.
CONTEXTS: tuple[tuple[str, str], ...] = (
    ("She wrote ", " on the page."),
    ("He said ", " out loud."),
    ("They read ", " again."),
    ("I noticed ", " immediately."),
)


def surface_variant(span: str) -> str:
    """A capitalisation variant: same content, different surface.

    Falls back to a trailing-space form for spans with no cased characters
    (punctuation, digits), so the control is defined for every span rather
    than silently degenerating to an identical string — which would drive the
    denominator to zero and MSG to infinity.
    """
    flipped = span.upper() if span[:1].islower() else span.lower()
    if flipped != span:
        return flipped
    swapped = span.swapcase()
    return swapped if swapped != span else f" {span}"


def batch_by_suffix(items: Sequence[SpanItem],
                    batch: int) -> Iterable[list[SpanItem]]:
    """Yield batches whose items share a suffix.

    The runner requires this: merge positions are located as
    `len(sequence) - len(suffix_tokens)`, so a batch mixing suffixes of
    different lengths would misalign the JSD slices across views. Grouping here
    is cheaper and less error-prone than padding around it.
    """
    buckets: dict[str, list[SpanItem]] = defaultdict(list)
    for item in items:
        buckets[item.suffix].append(item)
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch):
            yield bucket[start:start + batch]


def validate(items: Sequence[SpanItem]) -> None:
    """Fail loudly on the corpus errors that would otherwise surface as numbers.

    Each of these produces a plausible-looking MSG rather than an exception, so
    checking here is the difference between a wrong result and a stopped run.
    """
    if not items:
        raise ValueError("empty corpus")
    for i, it in enumerate(items):
        if not it.suffix:
            raise ValueError(f"item {i}: empty suffix leaves no merge position")
        if not it.span_text:
            raise ValueError(f"item {i}: empty span")
        if it.span_paraphrase == it.span_text:
            raise ValueError(
                f"item {i}: text control is identical to the span "
                f"({it.span_text!r}). The MSG denominator would collapse toward "
                "zero and the ratio would diverge.")
        if it.span_image is it.span_image_alt:
            raise ValueError(
                f"item {i}: image control is the same object as the span image; "
                "the denominator's image half would be exactly zero")
