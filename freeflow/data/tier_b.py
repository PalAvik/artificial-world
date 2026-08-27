"""Tier B — orthographic substitution (PLAN.md §2).

The centre of the project, and the tier that makes the abstract-word problem
tractable: you do not need a picture *of* "the", you need a picture of the
*word* "the". Every token in the vocabulary is substitutable this way, the data
is synthesised rather than annotated, and the bridge is information-preserving
by construction — the pixels contain the string exactly.

That last property is what makes the tier worth leading with. Any behavioural
gap here cannot be blamed on semantic ambiguity, annotation noise, or grounding
error. It is a statement about where the OCR pathway lands relative to the
embedding table, and nothing else.
"""
from __future__ import annotations

import random
from typing import Sequence

from .render import FontSet, RenderConfig, render_pair
from .views import CONTEXTS, ControlKind, SpanItem, surface_variant, validate

# Only for spans where a genuine near-synonym exists. Not the default: see
# data/views.py for why surface variants are the primary control.
SYNONYMS: dict[str, str] = {
    "dog": "hound", "table": "desk", "mountain": "peak", "window": "pane",
    "harbour": "port", "bridge": "span", "kettle": "pot", "ferry": "boat",
    "justice": "fairness", "theory": "hypothesis", "freedom": "liberty",
    "purpose": "aim", "doubt": "uncertainty", "custom": "tradition",
    "merit": "worth", "hazard": "danger", "notion": "idea", "virtue": "goodness",
    "malice": "spite", "sorrow": "grief", "wisdom": "insight", "regret": "remorse",
}


def _control(span: str, kind: ControlKind) -> tuple[str, str]:
    """The control span, and which kind was *actually* applied.

    The two can differ: a word with no synonym falls back to a surface variant
    rather than returning the span itself, which would collapse the
    denominator. Reporting the fallback matters — a denominator that mixes
    synonyms and capitalisation flips is two different measurements averaged
    together, and the surface half is much the smaller of the two.
    """
    if kind is ControlKind.SYNONYM:
        alt = SYNONYMS.get(span.lower())
        if alt:
            return alt, ControlKind.SYNONYM.value
    return surface_variant(span), ControlKind.SURFACE.value


def build(
    n: int,
    fonts: FontSet,
    cfg: RenderConfig,
    seed: int = 0,
    control: ControlKind = ControlKind.SURFACE,
    held_out_fonts: bool = False,
    words: Sequence[tuple[str, str]] | None = None,
) -> list[SpanItem]:
    """Build `n` Tier B items.

    `held_out_fonts=True` renders from the held-out font pool — that is the
    Gate 2(a) evaluation set, and it must never be used to build training data
    or to choose anything.

    Each item draws its two fonts without replacement, so the image half of the
    MSG denominator is a genuine font-to-font distance rather than zero.
    """
    from . import vocab

    rng = random.Random(seed)
    picks = list(words) if words is not None else vocab.sample(n, rng)
    items: list[SpanItem] = []

    for i, (word, cls) in enumerate(picks):
        prefix, suffix = CONTEXTS[i % len(CONTEXTS)]
        img, img_alt, font_a, font_b = render_pair(word, fonts, cfg, rng,
                                                   held_out_fonts)
        control_text, control_applied = _control(word, control)
        items.append(SpanItem(
            prefix=prefix,
            span_text=word,
            suffix=suffix,
            span_image=img,
            span_paraphrase=control_text,
            span_image_alt=img_alt,
            # The probe must recover *content*, not surface: two items showing
            # the same word in different fonts share a label.
            span_id=word.lower(),
            group=cls,
            tier="B",
            meta={"font": font_a, "font_control": font_b,
                  "control_kind": control.value,
                  "control_applied": control_applied,
                  "held_out_fonts": held_out_fonts,
                  "context": i % len(CONTEXTS)},
        ))

    validate(items)
    return items


def summary(items: Sequence[SpanItem]) -> dict:
    """Corpus composition, for the record alongside any number it produces."""
    from collections import Counter

    groups = Counter(it.group for it in items)
    fonts = Counter(it.meta.get("font") for it in items)
    return {
        "n": len(items),
        "tier": "B",
        "groups": dict(sorted(groups.items())),
        "unique_spans": len({it.span_id for it in items}),
        "fonts_used": len(fonts),
        "contexts": len({it.suffix for it in items}),
        "control_kind": items[0].meta.get("control_kind") if items else None,
        "held_out_fonts": items[0].meta.get("held_out_fonts") if items else None,
    }
