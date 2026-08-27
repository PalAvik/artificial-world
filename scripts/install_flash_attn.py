#!/usr/bin/env python3
"""Find and install a flash-attn wheel matching the installed torch/CUDA/Python.

FlashAttention-2 is the project default (docs/COMPUTE.md). Its kernels are
compiled per architecture, so the wheel must match the torch build exactly —
a `+cu124torch2.5` wheel will not import against `torch 2.13.0+cu130`, and a
mismatch shows up as an ImportError or an undefined symbol, not a clear message.

FA2 does cover both GPUs in this project: its setup.py defaults to
FLASH_ATTN_CUDA_ARCHS="80;90;100;110;120" and emits sm_100 gencode on CUDA
toolkits >= 12.8, so a CUDA 13 build serves the A100 (sm_80) and the B200
(sm_100) from one wheel.

Usage:
    python scripts/install_flash_attn.py            # find and print candidates
    python scripts/install_flash_attn.py --install  # install the best match
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

REPO = "mjun0812/flash-attention-prebuild-wheels"
API = f"https://api.github.com/repos/{REPO}/releases?per_page=100"


def local_tags() -> tuple[str, str, str]:
    """Return (cuda_tag, torch_tag, py_tag) as they appear in wheel names,
    e.g. ("cu130", "torch2.13", "cp311")."""
    import torch

    if torch.version.cuda is None:
        sys.exit("This torch is a CPU build — install a CUDA build first "
                 "(docs/ENVIRONMENT.md §1).")
    cuda = "cu" + torch.version.cuda.replace(".", "")
    tv = re.match(r"(\d+)\.(\d+)", torch.__version__)
    torch_tag = f"torch{tv.group(1)}.{tv.group(2)}"
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return cuda, torch_tag, py


def fetch_releases(pages: int = 3) -> list[dict]:
    out = []
    for page in range(1, pages + 1):
        req = urllib.request.Request(
            f"{API}&page={page}",
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "freeflow-setup"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                batch = json.load(r)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            print(f"  ! GitHub API unreachable ({exc}) — falling back to source build")
            return out
        if not batch:
            break
        out.extend(batch)
    return out


def source_build_help(cuda: str) -> None:
    """Restricting the arch list matters: the default builds five architectures
    and takes well over an hour. Two takes a fraction of that."""
    cuda_ok = int(cuda[2:]) >= 128
    print("\nNo matching wheel. Build from source instead:\n")
    print("  uv pip install ninja packaging psutil")
    print("  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}")
    print("  nvcc --version        # must be 12.8+ for sm_100 (B200)")
    print('  MAX_JOBS=4 FLASH_ATTN_CUDA_ARCHS="80;100" \\')
    print("      uv pip install flash-attn --no-build-isolation")
    print("\n  Only 80 (A100) and 100 (B200) are built — the default list of five")
    print("  architectures takes several times longer for kernels you won't run.")
    print("  MAX_JOBS=4 keeps nvcc from exhausting RAM; raise it if you have headroom.")
    if not cuda_ok:
        print(f"\n  WARNING: this torch is built for {cuda}. sm_100 needs CUDA 12.8+,")
        print("  so a build here will not produce B200 kernels.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="pip install the best match")
    args = ap.parse_args()

    cuda, torch_tag, py = local_tags()
    import torch
    print(f"installed: torch {torch.__version__} (CUDA {torch.version.cuda}), "
          f"Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"wheel must match: +{cuda}{torch_tag}-{py}-{py}-linux_x86_64\n")

    want = re.compile(
        rf"^flash_attn-(?P<ver>[\d.post]+)\+{re.escape(cuda)}{re.escape(torch_tag)}"
        rf"-{py}-{py}-linux_x86_64\.whl$")

    matches = []
    for rel in fetch_releases():
        for asset in rel.get("assets", []):
            m = want.match(asset["name"])
            if m:
                matches.append((m.group("ver"), asset["browser_download_url"],
                                rel.get("tag_name", "?")))

    if not matches:
        source_build_help(cuda)
        return 1

    # Newest flash-attn version wins.
    matches.sort(key=lambda t: [int(x) for x in re.findall(r"\d+", t[0])], reverse=True)
    print(f"found {len(matches)} matching wheel(s):")
    for ver, url, tag in matches[:5]:
        print(f"  flash_attn {ver}  (release {tag})")
    best_ver, best_url, _ = matches[0]
    print(f"\nbest match: {best_url}")

    if not args.install:
        print("\nRe-run with --install, or install it yourself:")
        print(f"  uv pip install {best_url}")
        return 0

    print(f"\ninstalling flash_attn {best_ver} ...")
    for cmd in (["uv", "pip", "install", best_url],
                [sys.executable, "-m", "pip", "install", best_url]):
        try:
            if subprocess.run(cmd).returncode == 0:
                break
        except FileNotFoundError:
            continue
    else:
        print("install failed")
        return 1

    print("\nverifying against the GPU (installing is not the same as working):")
    check = subprocess.run(
        [sys.executable, "-c",
         "import torch, flash_attn;"
         "from flash_attn import flash_attn_func;"
         "q=torch.randn(1,8,4,64,device='cuda',dtype=torch.bfloat16);"
         "flash_attn_func(q,q,q);"
         "print('  OK: flash-attn', flash_attn.__version__, 'usable on',"
         " torch.cuda.get_device_name(0))"])
    if check.returncode != 0:
        print("  FAILED on this device — use attn_implementation=\"sdpa\" here.")
        print("  Note the backend in results/RESULTS.md: runs compared against each")
        print("  other must share it (hardware policy, docs/GATES.md).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
