"""Tier C — relational substitution (PLAN.md §2).

The hardest tier and the one carrying the strongest claim. If entities transfer
across modalities but *relations* do not, that asymmetry says multimodal
pretraining fuses nouns and not structure — which is a statement about world
models rather than about representation hygiene.

**The diagram shows the relation, not the proposition.** A picture of a red
circle above a blue square would carry the whole sentence, not just the span,
and the substitution would no longer be like-for-like. So the diagram uses
neutral unlabelled markers: it depicts *"the first thing is above the second"*
as a schema, leaving the sentence to name what the things are. That keeps the
image span carrying the same information the text span carries — which is the
premise of the whole substitution test.

Scenes are generated with programmatic ground truth, so unlike Tier A there is
no annotation noise to mistake for a representational gap.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from PIL import Image, ImageDraw

from .views import ControlKind, SpanItem, surface_variant, validate

CANVAS = 96          # 3x3 tokens at 32px each, plus wrappers
MARKER = 26


@dataclass(frozen=True)
class Relation:
    """A relation, its phrasing, and where the two markers sit."""

    name: str
    phrase: str
    a: tuple[float, float]           # centre of marker A, in [0,1] canvas coords
    b: tuple[float, float]
    nested: bool = False             # B drawn inside A


RELATIONS: tuple[Relation, ...] = (
    Relation("above", "above", (0.5, 0.24), (0.5, 0.76)),
    Relation("below", "below", (0.5, 0.76), (0.5, 0.24)),
    Relation("left_of", "left of", (0.24, 0.5), (0.76, 0.5)),
    Relation("right_of", "right of", (0.76, 0.5), (0.24, 0.5)),
    Relation("inside", "inside", (0.5, 0.5), (0.5, 0.5), nested=True),
)

# Frames whose objects are named in the text, so the diagram never has to.
FRAMES: tuple[tuple[str, str], ...] = (
    ("The circle is ", " the square."),
    ("The marker sits ", " the box."),
    ("The token is ", " the frame."),
)

# The control varies marker *shape*, which is orthogonal to the relation. Fill
# cannot be the control axis: it is what marks the subject (see draw_relation).
STYLES = ("square", "circle")


def draw_relation(rel: Relation, style: str = "square",
                  canvas: int = CANVAS, marker: int = MARKER) -> Image.Image:
    """Render a relation as a neutral two-marker diagram.

    **The subject is always filled and the object always outlined.** Without
    that asymmetry the two markers are interchangeable and "above" renders
    identically to "below" — the arrangement alone cannot say which marker the
    sentence is about, so the image would not carry the relation at all. This
    is why the style control varies shape rather than fill: fill is load-bearing.
    """
    img = Image.new("RGB", (canvas, canvas), "white")
    d = ImageDraw.Draw(img)

    def mark(centre: tuple[float, float], size: float, filled: bool):
        cx, cy = centre[0] * canvas, centre[1] * canvas
        half = size / 2
        xy = [cx - half, cy - half, cx + half, cy + half]
        shape = d.ellipse if style == "circle" else d.rectangle
        shape(xy, fill="black" if filled else None, outline="black", width=3)

    if rel.nested:
        mark(rel.b, marker * 2.4, False)      # object: the container
        mark(rel.a, marker * 0.6, True)       # subject: filled, inside it
    else:
        mark(rel.b, marker, False)            # object: outline
        mark(rel.a, marker, True)             # subject: filled
    return img


def build(n: int, seed: int = 0, control: ControlKind = ControlKind.SURFACE,
          relations: Sequence[Relation] = RELATIONS) -> list[SpanItem]:
    """Build `n` Tier C items, balanced across relations.

    `control` is accepted for signature parity with the other tiers but only
    SURFACE is meaningful here: a relation has no synonym any more than "the"
    does, which is the same reason surface variants are the default everywhere.
    """
    if control is not ControlKind.SURFACE:
        raise ValueError(
            f"Tier C supports only ControlKind.SURFACE, got {control}. "
            "Relations have no synonyms.")
    rng = random.Random(seed)
    items: list[SpanItem] = []

    for i in range(n):
        rel = relations[i % len(relations)]
        prefix, suffix = FRAMES[i % len(FRAMES)]
        style, style_alt = rng.sample(STYLES, 2)
        items.append(SpanItem(
            prefix=prefix,
            span_text=rel.phrase,
            suffix=suffix,
            span_image=draw_relation(rel, style),
            span_paraphrase=surface_variant(rel.phrase),
            span_image_alt=draw_relation(rel, style_alt),
            span_id=rel.name,
            group=rel.name,
            tier="C",
            meta={"style": style, "style_control": style_alt,
                  "nested": rel.nested, "canvas": CANVAS},
        ))

    validate(items)
    return items


def summary(items: Sequence[SpanItem]) -> dict:
    from collections import Counter

    return {
        "n": len(items),
        "tier": "C",
        "relations": dict(sorted(Counter(it.group for it in items).items())),
        "canvas": CANVAS,
        "contexts": len({it.suffix for it in items}),
    }
