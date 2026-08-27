#!/usr/bin/env python3
"""Assemble the Tier B font set without root.

Pillow loads TrueType files by absolute path, so no system font installation is
needed — `apt-get install fonts-*` was never actually required. Two sources:

  1. matplotlib's bundled TTFs (guaranteed present, since matplotlib is a
     dependency) — five visually distinct families, enough on their own.
  2. Optional Google Fonts extras downloaded into ~/.local/share/fonts, which
     widen the within-modality control and the render-robustness ablation axes.

Font diversity is not cosmetic here: re-rendering a span in a different family is
V_I', the denominator of the normalized MSG. And Gate 2 requires held-out fonts,
so the split is written now rather than improvised later.

Usage:
    python scripts/setup_fonts.py                 # matplotlib fonts only
    python scripts/setup_fonts.py --download      # also fetch the extras
"""
from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request

# Distinct families from matplotlib's bundle. Deliberately not every weight —
# oblique/bold variants of one family are a weak within-modality control.
MPL_FAMILIES = {
    "dejavu-sans": "DejaVuSans.ttf",
    "dejavu-serif": "DejaVuSerif.ttf",
    "dejavu-mono": "DejaVuSansMono.ttf",
    "stix-general": "STIXGeneral.ttf",
    "computer-modern-serif": "cmr10.ttf",
    "computer-modern-sans": "cmss10.ttf",
    "computer-modern-mono": "cmtt10.ttf",
}

# Verified reachable at raw.githubusercontent.com. Note github.com/.../raw/ 403s
# behind some proxies — use the raw host. Square brackets are percent-encoded
# because these are variable fonts; Pillow loads their default instance fine.
GOOGLE_FONTS = {
    "lato": "ofl/lato/Lato-Regular.ttf",
    "space-mono": "ofl/spacemono/SpaceMono-Regular.ttf",
    "open-sans": "ofl/opensans/OpenSans%5Bwdth,wght%5D.ttf",
    "noto-sans": "ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf",
    "inconsolata": "ofl/inconsolata/Inconsolata%5Bwdth,wght%5D.ttf",
    "oswald": "ofl/oswald/Oswald%5Bwght%5D.ttf",
    "merriweather": "ofl/merriweather/Merriweather%5Bopsz,wdth,wght%5D.ttf",
    "source-serif": "ofl/sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf",
    "ibm-plex-sans": "ofl/ibmplexsans/IBMPlexSans%5Bwdth,wght%5D.ttf",
}
GF_BASE = "https://raw.githubusercontent.com/google/fonts/main/"

# Held out from training entirely, so Gate 2's "held-out fonts" condition has
# something to test against. One serif, one sans, one mono — a font the model
# never trained on should not be a font whose *category* it never trained on.
HELD_OUT = {"stix-general", "computer-modern-sans", "source-serif", "space-mono"}

USER_FONT_DIR = os.path.expanduser("~/.local/share/fonts/freeflow")


def matplotlib_fonts() -> dict[str, str]:
    import matplotlib

    ttf_dir = os.path.join(
        os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    found = {}
    for name, filename in MPL_FAMILIES.items():
        path = os.path.join(ttf_dir, filename)
        if os.path.exists(path):
            found[name] = path
        else:
            print(f"  ! {name}: {filename} missing from this matplotlib build")
    return found


def download_fonts() -> dict[str, str]:
    os.makedirs(USER_FONT_DIR, exist_ok=True)
    found = {}
    for name, rel in GOOGLE_FONTS.items():
        dest = os.path.join(USER_FONT_DIR, name + ".ttf")
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            found[name] = dest
            print(f"  = {name} (cached)")
            continue
        try:
            urllib.request.urlretrieve(GF_BASE + rel, dest)
            found[name] = dest
            print(f"  + {name} ({os.path.getsize(dest) // 1024} KB)")
        except (urllib.error.URLError, OSError) as exc:
            print(f"  ! {name}: {exc}")
            if os.path.exists(dest):
                os.remove(dest)
    return found


def verify(fonts: dict[str, str]) -> dict[str, str]:
    """Render with each font. A file that exists but won't rasterise is worse
    than one that's absent, because it fails later and silently."""
    from PIL import Image, ImageDraw, ImageFont

    good = {}
    for name, path in sorted(fonts.items()):
        try:
            font = ImageFont.truetype(path, 22)
            img = Image.new("RGB", (240, 34), "white")
            ImageDraw.Draw(img).text((4, 4), "the quick brown", "black", font=font)
            if len(img.getcolors(maxcolors=65536) or []) < 2:
                raise ValueError("rendered blank")
            good[name] = path
        except Exception as exc:
            print(f"  ! {name} failed to render: {type(exc).__name__}: {exc}")
    return good


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="also fetch Google Fonts extras into ~/.local/share/fonts")
    ap.add_argument("--out", default="configs/fonts.yaml")
    args = ap.parse_args()

    print("matplotlib bundled fonts:")
    fonts = matplotlib_fonts()
    print(f"  {len(fonts)} found")

    if args.download:
        print("\ngoogle fonts extras:")
        fonts.update(download_fonts())

    print("\nverifying by rendering:")
    fonts = verify(fonts)
    print(f"  {len(fonts)} usable")

    if len(fonts) < 4:
        print("\nFAIL: fewer than 4 usable families. The within-modality control "
              "(V_I') would be too weak to give the MSG a meaningful denominator.")
        return 1

    train = {k: v for k, v in fonts.items() if k not in HELD_OUT}
    held = {k: v for k, v in fonts.items() if k in HELD_OUT}
    if not held:
        print("\nFAIL: no held-out fonts survived. Gate 2 condition (a) requires "
              "held-out fonts; fix the HELD_OUT set before proceeding.")
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("# Generated by scripts/setup_fonts.py — do not hand-edit.\n")
        fh.write("# Frozen once Gate 0 passes. Held-out fonts never appear in training.\n")
        for split, group in (("train", train), ("held_out", held)):
            fh.write(f"\n{split}:\n")
            for name, path in sorted(group.items()):
                fh.write(f"  {name}: {path}\n")

    print(f"\nwrote {args.out}")
    print(f"  train:    {len(train)} families — {', '.join(sorted(train))}")
    print(f"  held out: {len(held)} families — {', '.join(sorted(held))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
