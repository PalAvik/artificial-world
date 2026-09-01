"""Render the figure's example span with the project's own renderer.

The method figure shows a Tier B item, and the image view of that item should be
*the actual stimulus* at the frozen Gate 0 geometry rather than a drawing of
one. Two consequences worth the small amount of machinery: the figure cannot
drift from the corpus, and a reader can see for themselves how legible a
six-visual-token span really is.

    python scripts/make_figure_assets.py --word wisdom
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freeflow.data.render import FontSet, RenderConfig, render_span  # noqa: E402


def _fallback_fonts() -> FontSet:
    import matplotlib
    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    return FontSet(train={"sans": str(ttf / "DejaVuSans.ttf"),
                          "serif": str(ttf / "DejaVuSerif.ttf")},
                   held_out={})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="wisdom")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()

    try:
        cfg = RenderConfig.load()
    except FileNotFoundError:
        cfg = RenderConfig(height=48, pad=4, min_pixels=1024,
                           visual_tokens_per_span=6)
    try:
        fonts = FontSet.load()
        names = fonts.pool(False)[:2]
        paths = [fonts.paths(False)[n] for n in names]
    except (FileNotFoundError, KeyError, IndexError):
        fonts = _fallback_fonts()
        names = fonts.pool(False)[:2]
        paths = [fonts.paths(False)[n] for n in names]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, path in zip(("primary", "control"), paths):
        img = render_span(args.word, path, cfg)
        dest = out_dir / f"span_{tag}.png"
        img.save(dest)
        print(f"{dest}  {img.width}x{img.height}  font={Path(path).stem}")
    print(f"config: height {cfg.height}, pad {cfg.pad}, "
          f"{cfg.visual_tokens_per_span} visual tokens/span")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
