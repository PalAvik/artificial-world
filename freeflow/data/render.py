"""Glyph rendering at the Gate 0 config, plus the font split.

Single source of truth for turning a span into pixels. Everything reads the
frozen `configs/render.yaml` and `configs/fonts.yaml` rather than carrying its
own defaults — changing a render parameter must invalidate every prior
measurement loudly, not silently produce a differently-scaled corpus.
"""
from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RenderConfig:
    """The frozen Gate 0 config. Do not construct by hand outside tests."""

    height: int
    pad: int
    min_pixels: int
    visual_tokens_per_span: int

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RenderConfig":
        path = Path(path or ROOT / "configs" / "render.yaml")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Gate 0 must pass before any corpus is built: "
                "run scripts/gate0_sweep.py, which writes this file.")
        cfg = yaml.safe_load(path.read_text())
        return cls(height=int(cfg["height"]), pad=int(cfg["pad"]),
                   min_pixels=int(cfg["min_pixels"]),
                   visual_tokens_per_span=int(cfg["visual_tokens_per_span"]))


@dataclass(frozen=True)
class FontSet:
    """Train and held-out font paths, kept apart by construction."""

    train: dict[str, str]
    held_out: dict[str, str]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FontSet":
        path = Path(path or ROOT / "configs" / "fonts.yaml")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/setup_fonts.py --download.")
        cfg = yaml.safe_load(path.read_text())
        return cls(train=dict(cfg.get("train") or {}),
                   held_out=dict(cfg.get("held_out") or {}))

    def pool(self, held_out: bool = False) -> list[str]:
        names = sorted(self.held_out if held_out else self.train)
        if len(names) < 2:
            raise ValueError(
                f"need >=2 fonts to form an image control, have {len(names)}")
        return names

    def paths(self, held_out: bool = False) -> dict[str, str]:
        return self.held_out if held_out else self.train


@functools.lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def render_span(text: str, font_path: str, cfg: RenderConfig) -> Image.Image:
    """Render one span as a narrow strip at the frozen geometry."""
    font = _font(font_path, max(6, cfg.height - 2 * cfg.pad))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = int(probe.textlength(text, font=font)) + 2 * cfg.pad
    img = Image.new("RGB", (width, cfg.height), "white")
    ImageDraw.Draw(img).text((cfg.pad, cfg.pad), text, fill="black", font=font)
    return img


def render_pair(text: str, fonts: FontSet, cfg: RenderConfig,
                rng: random.Random, held_out: bool = False
                ) -> tuple[Image.Image, Image.Image, str, str]:
    """Render a span twice in two *different* fonts.

    Returns `(primary, control, primary_font, control_font)`. The two fonts are
    drawn without replacement: rendering the control in the same font would
    make the image half of the MSG denominator exactly zero, which would send
    the ratio to infinity for every item.
    """
    names = fonts.pool(held_out)
    a, b = rng.sample(names, 2)
    paths = fonts.paths(held_out)
    return (render_span(text, paths[a], cfg),
            render_span(text, paths[b], cfg), a, b)
