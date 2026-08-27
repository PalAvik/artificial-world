"""Tier A — referential substitution (PLAN.md §2).

A phrase replaced by a crop of the region it refers to. The bridge is grounding,
and it is the one tier whose alignment is *annotated* rather than constructed —
which is exactly why it is the control and not the contribution. Flickr30k
Entities and Visual Genome give phrase-to-box pairs directly, so nothing is
synthesised here; the cost is annotation noise that Tiers B and C do not have.

**Substitution here is lossy in a way the other tiers are not**, and the plan
says so: the crop specifies a particular dog, the phrase specifies a class. Any
MSG measured here is an upper bound that mixes representational divergence with
genuine information asymmetry, so it is read as a control against Tier B rather
than as a headline.

**The image control is a different instance of the same phrase** — another
region annotated with the same words, ideally from a different image. That is
the referential analogue of a font change: same content, different rendering.
"""
from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .views import CONTEXTS, ControlKind, SpanItem, surface_variant, validate

MIN_BOX = 32          # crops smaller than a visual token carry nothing
MAX_PHRASE_WORDS = 4


@dataclass(frozen=True)
class Region:
    """One annotated phrase and the box it refers to."""

    phrase: str
    image_path: Path
    box: tuple[int, int, int, int]        # x0, y0, x1, y1

    @property
    def area(self) -> int:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


# --------------------------------------------------------- Visual Genome ---

def load_visual_genome(regions_json: str | Path, images_dir: str | Path,
                       limit: int | None = None) -> list[Region]:
    """Read `region_descriptions.json`.

    Format: a list of `{"id": image_id, "regions": [{phrase, x, y, width,
    height, image_id}, ...]}`. Images are looked up as `<image_id>.jpg`, which
    is how the VG_100K dumps are laid out.
    """
    images_dir = Path(images_dir)
    data = json.loads(Path(regions_json).read_text())
    out: list[Region] = []
    for entry in data:
        for r in entry.get("regions", []):
            phrase = _clean(r.get("phrase", ""))
            if not phrase:
                continue
            img = images_dir / f"{r.get('image_id', entry.get('id'))}.jpg"
            x, y = int(r.get("x", 0)), int(r.get("y", 0))
            w, h = int(r.get("width", 0)), int(r.get("height", 0))
            if w < MIN_BOX or h < MIN_BOX:
                continue
            out.append(Region(phrase, img, (x, y, x + w, y + h)))
            if limit and len(out) >= limit:
                return out
    return out


# ---------------------------------------------------- Flickr30k Entities ---

_ENTITY = re.compile(r"\[/EN#(\d+)(?:/[^\s]+)*\s+([^\]]+)\]")


def parse_flickr_sentence(line: str) -> list[tuple[str, str]]:
    """Extract `(entity_id, phrase)` from one Flickr30k Entities sentence.

    Markup looks like `[/EN#283585/people A man] is [/EN#.../bike riding]`.
    """
    out = []
    for m in _ENTITY.finditer(line):
        phrase = _clean(m.group(2))
        if phrase:                    # _clean empties phrases over the word cap
            out.append((m.group(1), phrase))
    return out


def parse_flickr_boxes(xml_path: str | Path) -> dict[str, list[tuple]]:
    """Map entity id -> boxes from an Annotations XML file.

    Entities marked `nobox` are skipped: they are annotated as present but
    unlocalised, and a phrase with no region cannot be substituted.
    """
    root = ET.parse(str(xml_path)).getroot()
    boxes: dict[str, list[tuple]] = defaultdict(list)
    for obj in root.iter("object"):
        names = [n.text for n in obj.findall("name") if n.text]
        bb = obj.find("bndbox")
        if bb is None or not names:
            continue
        try:
            box = tuple(int(bb.find(t).text) for t in
                        ("xmin", "ymin", "xmax", "ymax"))
        except (AttributeError, TypeError, ValueError):
            continue
        for name in names:
            boxes[name].append(box)
    return dict(boxes)


def load_flickr30k(root: str | Path, images_dir: str | Path,
                   limit: int | None = None) -> list[Region]:
    """Read a Flickr30k Entities checkout: `Sentences/` + `Annotations/`."""
    root, images_dir = Path(root), Path(images_dir)
    sentences_dir, annotations_dir = root / "Sentences", root / "Annotations"
    if not sentences_dir.is_dir():
        raise FileNotFoundError(
            f"{sentences_dir} not found — expected a flickr30k_entities "
            "checkout with Sentences/ and Annotations/ (docs/ENVIRONMENT.md §4)")

    out: list[Region] = []
    for sent_file in sorted(sentences_dir.glob("*.txt")):
        stem = sent_file.stem
        xml = annotations_dir / f"{stem}.xml"
        if not xml.exists():
            continue
        boxes = parse_flickr_boxes(xml)
        image = images_dir / f"{stem}.jpg"
        for line in sent_file.read_text(encoding="utf-8").splitlines():
            for entity_id, phrase in parse_flickr_sentence(line):
                for box in boxes.get(entity_id, []):
                    if (box[2] - box[0]) < MIN_BOX or (box[3] - box[1]) < MIN_BOX:
                        continue
                    out.append(Region(phrase, image, box))
                    if limit and len(out) >= limit:
                        return out
    return out


# ------------------------------------------------------------ assembling ---

def _clean(phrase: str) -> str:
    phrase = " ".join(phrase.split()).strip().lower()
    return phrase if 0 < len(phrase.split()) <= MAX_PHRASE_WORDS else ""


def group_by_phrase(regions: Sequence[Region],
                    min_instances: int = 2) -> dict[str, list[Region]]:
    """Group regions by phrase, keeping only phrases with enough instances.

    Two is the floor because the image control is *another instance of the same
    phrase*: a phrase seen once has no control, and including it would leave
    the MSG denominator undefined for that item.
    """
    by_phrase: dict[str, list[Region]] = defaultdict(list)
    for r in regions:
        by_phrase[r.phrase].append(r)
    return {p: rs for p, rs in by_phrase.items() if len(rs) >= min_instances}


def _crop(region: Region, cache: dict) -> object:
    from PIL import Image

    img = cache.get(region.image_path)
    if img is None:
        img = Image.open(region.image_path).convert("RGB")
        cache[region.image_path] = img
    return img.crop(region.box)


def build(regions: Sequence[Region], n: int, seed: int = 0,
          control: ControlKind = ControlKind.SURFACE,
          prefer_cross_image: bool = True) -> list[SpanItem]:
    """Assemble Tier A items from loaded regions.

    `prefer_cross_image` picks the control crop from a *different* photograph
    where possible. Two crops of the same phrase from one image often overlap,
    and a near-duplicate control would shrink the MSG denominator and inflate
    the gap — the same failure as rendering the image control in the same font.
    """
    rng = random.Random(seed)
    grouped = group_by_phrase(regions)
    if not grouped:
        raise ValueError(
            "no phrase has two annotated instances; Tier A needs a second "
            "instance per phrase to form the image control")

    phrases = sorted(grouped)
    rng.shuffle(phrases)
    cache: dict = {}
    items: list[SpanItem] = []

    for i in range(n):
        phrase = phrases[i % len(phrases)]
        instances = grouped[phrase]
        primary = rng.choice(instances)
        alt = _pick_control(primary, instances, rng, prefer_cross_image)
        if alt is None:
            continue
        prefix, suffix = CONTEXTS[i % len(CONTEXTS)]
        items.append(SpanItem(
            prefix=prefix,
            span_text=phrase,
            suffix=suffix,
            span_image=_crop(primary, cache),
            span_paraphrase=surface_variant(phrase),
            span_image_alt=_crop(alt, cache),
            span_id=phrase,
            group="referential",
            tier="A",
            meta={"image": primary.image_path.name,
                  "image_control": alt.image_path.name,
                  "cross_image": primary.image_path != alt.image_path,
                  "box": primary.box, "control_kind": control.value},
        ))

    validate(items)
    return items


def _pick_control(primary: Region, instances: Sequence[Region],
                  rng: random.Random, prefer_cross_image: bool) -> Region | None:
    others = [r for r in instances if r is not primary]
    if not others:
        return None
    if prefer_cross_image:
        cross = [r for r in others if r.image_path != primary.image_path]
        if cross:
            return rng.choice(cross)
    return rng.choice(others)


def summary(items: Sequence[SpanItem]) -> dict:
    cross = sum(1 for it in items if it.meta.get("cross_image"))
    return {
        "n": len(items),
        "tier": "A",
        "unique_phrases": len({it.span_id for it in items}),
        "cross_image_controls": cross,
        "cross_image_rate": cross / max(1, len(items)),
    }


def iter_summary(regions: Sequence[Region]) -> Iterator[str]:
    grouped = group_by_phrase(regions)
    yield f"{len(regions)} regions, {len(grouped)} phrases with >=2 instances"
    top = sorted(grouped.items(), key=lambda kv: -len(kv[1]))[:5]
    for phrase, rs in top:
        yield f"  {phrase!r}: {len(rs)} instances"
