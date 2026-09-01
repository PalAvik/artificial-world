"""Tier P --- pictorial substitution: a word replaced by a photograph of it.

Tier B replaces a word with a *picture of the word*; this replaces it with a
picture of the **thing the word denotes**. It is the substitution most people
mean when they ask whether a model's representation of "dog" and its
representation of a dog are the same thing, and it is closer to the question
this project started from. It is also strictly weaker as an instrument, in
three specific ways that have to be stated before any number is read.

**1. The substitution is not content-preserving.** `dog` denotes a category; a
photograph denotes one particular animal, breed, pose, lighting and background.
A nonzero gap is therefore expected under every hypothesis, which is exactly
the confound orthographic substitution was introduced to remove. What keeps the
tier interpretable is the control: `span_image_alt` is **a different photograph
of the same category**, never a re-crop or re-encoding of the same one. The
denominator then carries instance variation too, so the ratio asks the sharper
question --- *does crossing from the word to a photograph cost more than moving
between two photographs of the same thing?* --- rather than the unanswerable
one about a word and a picture being identical.

**2. Only concrete nouns are depictable.** There is no photograph of `although`
or of `justice`. So this tier cannot speak to H2 at all: the word-class
comparison is precisely what Tier B exists to make possible, and no pictorial
corpus can replace it.

**3. Distinct spans are capped by the number of categories.** A span here is an
object category, and extra photographs of one category add rows, not
constraints. At `D = 2048` the map-based nulls need ~4096 distinct spans, which
rules out COCO (80), Open Images (600) and ImageNet-1k (1000). Only a
category-rich source --- ImageNet-21k, iNaturalist --- can support them. With a
1000-class source this tier yields MSG and the forced choice, and the map nulls
must be marked untestable rather than run.

**Token budget is a confound between tiers.** A glyph strip is high-contrast
line art and survives six visual tokens; a photograph at six tokens is unlikely
to be recognisable at all. Comparing Tier B at six tokens against Tier P at
sixty-four would confound the modality gap with the token budget. `target_pixels`
therefore controls the budget explicitly and is recorded per item, and a
cross-tier claim requires the two tiers to be run at the same budget.

Images are read from an ImageFolder layout, which is what nearly every source
already provides:

    <root>/<category>/<any>.jpg
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from .views import CONTEXTS, ControlKind, SpanItem, surface_variant, validate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_PER_CATEGORY = 2      # one for the view, a *different* one for the control


@dataclass(frozen=True)
class PictorialConfig:
    """How a photograph is presented to the model.

    `target_pixels` is the budget knob. It is not the frozen Gate 0 geometry:
    that was chosen for glyph strips, and a photograph is a different kind of
    stimulus. Whatever is chosen here must be recorded and matched across any
    tiers being compared.
    """

    target_pixels: int = 200704      # 448x448, a common ViT working resolution
    square: bool = True              # letterbox rather than crop; see below


def load_categories(root: str | Path,
                    min_images: int = MIN_PER_CATEGORY) -> dict[str, list[Path]]:
    """Read an ImageFolder tree into `{category: [paths]}`.

    Categories with fewer than `min_images` photographs are dropped rather than
    reused: a control that is the *same* photograph as the view would drive the
    image half of the denominator to zero and send MSG to infinity, which is
    the same failure the two-fonts-without-replacement rule prevents in Tier B.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} is not a directory. Tier P expects an ImageFolder tree, "
            "<root>/<category>/<image>.jpg — see docs/ENVIRONMENT.md")
    out: dict[str, list[Path]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        paths = sorted(p for p in child.iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(paths) >= min_images:
            out[_category_word(child.name)] = paths
    if not out:
        raise ValueError(
            f"no category under {root} has {min_images} images; the image "
            "control must be a *different* photograph of the same category")
    return out


def _category_word(name: str) -> str:
    """Directory name to the word that will appear in the text view.

    ImageFolder trees are often keyed by synset id (`n02084071`) or by a
    multi-word label. A span that is not a plain lowercase word cannot be
    compared with Tier B, so those are normalised here and anything still
    unusable is rejected by `build`.
    """
    label = name.strip().lower().replace("_", " ").replace("-", " ")
    return label.split(",")[0].strip()


def _present(path: Path, cfg: PictorialConfig) -> Image.Image:
    """Load and scale one photograph to the configured budget.

    Letterboxed rather than centre-cropped: a crop can remove the object the
    word denotes, which would silently turn some items into a substitution of
    the wrong content — a failure that produces a plausible number rather than
    an error.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = (cfg.target_pixels / float(w * h)) ** 0.5
    new = (max(28, int(w * scale)), max(28, int(h * scale)))
    img = img.resize(new, Image.LANCZOS)
    if cfg.square:
        side = max(img.size)
        canvas = Image.new("RGB", (side, side), "white")
        canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        img = canvas
    return img


def build(n: int, images_root: str | Path, cfg: PictorialConfig | None = None,
          seed: int = 0, control: ControlKind = ControlKind.SURFACE,
          synonyms: dict[str, str] | None = None,
          categories: dict[str, list[Path]] | None = None) -> list[SpanItem]:
    """Build `n` Tier P items.

    Each item substitutes a category word with a photograph of that category;
    the image control is a different photograph of the same category, and the
    text control is a synonym where one is known, else a surface variant.
    """
    cfg = cfg or PictorialConfig()
    rng = random.Random(seed)
    cats = categories if categories is not None else load_categories(images_root)
    usable = {w: p for w, p in cats.items() if w.isalpha() and w.islower()}
    if not usable:
        raise ValueError(
            "no category name is a plain lowercase word; Tier P spans must be "
            "words so the text view is comparable with the other tiers. "
            "Map synset ids to words before building.")

    words = sorted(usable)
    items: list[SpanItem] = []
    for i in range(n):
        word = words[i % len(words)]
        paths = usable[word]
        primary, alt = rng.sample(paths, 2)      # never the same photograph
        prefix, suffix = CONTEXTS[i % len(CONTEXTS)]
        syn = (synonyms or {}).get(word)
        paraphrase = syn if (control is ControlKind.SYNONYM and syn) \
            else surface_variant(word)
        items.append(SpanItem(
            prefix=prefix,
            span_text=word,
            suffix=suffix,
            span_image=_present(primary, cfg),
            span_paraphrase=paraphrase,
            span_image_alt=_present(alt, cfg),
            # Content, not instance: two photographs of a dog share a label, so
            # the probe measures whether the category survives, not whether a
            # particular photograph is memorable.
            span_id=word,
            group="pictorial",
            tier="P",
            meta={"primary_image": str(primary), "control_image": str(alt),
                  "target_pixels": cfg.target_pixels,
                  "control_kind": control.value,
                  "control_applied": ("synonym" if (control is ControlKind.SYNONYM
                                                    and syn) else "surface"),
                  "context": i % len(CONTEXTS)},
        ))
    validate(items)
    return items


def coverage(items: Sequence[SpanItem], dim: int = 2048) -> str | None:
    """None when the corpus can support a `[D, D]` map fit.

    Extra photographs of one category are extra rows and not extra constraints,
    so this counts categories. Stated as a function rather than left to the
    caller because getting it wrong is what produced a meaningless MSG of 0.006
    on 2026-08-31.
    """
    n_spans = len({it.span_id for it in items})
    if n_spans >= 2 * dim:
        return None
    return (f"{n_spans} distinct categories against {dim} dimensions. Tier P "
            "spans are object categories, and more photographs per category add "
            "rows rather than constraints, so the map-based nulls cannot "
            "conclude either way here. MSG and the forced choice are unaffected")
