#!/usr/bin/env python3
"""Report GPU architecture, driver, torch arch coverage, and MIG topology.

Run this on every machine before running anything else. It exists because
"no kernel image is available for execution on the device" is a compile-time
mismatch, not a runtime bug: the installed torch simply has no kernels for the
GPU's compute capability, and no amount of debugging the model code fixes it.

Usage:
    python scripts/check_gpu.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

# Compute capability -> the CUDA toolkit generation that first shipped kernels
# for it, and the minimum driver for that toolkit's runtime.
ARCH_NOTES = {
    "sm_80": ("A100 (Ampere)", "CUDA 11.0+", 525),
    "sm_86": ("A10/A40/RTX 30xx (Ampere)", "CUDA 11.1+", 525),
    "sm_89": ("L40S/RTX 40xx (Ada)", "CUDA 11.8+", 525),
    "sm_90": ("H100/H200 (Hopper)", "CUDA 11.8+", 525),
    "sm_100": ("B200/GB200 (Blackwell)", "CUDA 12.8+", 570),
    "sm_120": ("RTX 50xx / B40 (Blackwell)", "CUDA 12.8+", 570),
}


def nvidia_smi(args: list[str]) -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True,
                              text=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def report_driver() -> tuple[str, int | None]:
    out = nvidia_smi(["--query-gpu=name,driver_version,memory.total",
                      "--format=csv,noheader"])
    if not out.strip():
        print("nvidia-smi unavailable — is this a GPU node?")
        return "", None
    print("Physical GPUs")
    driver = None
    for line in out.strip().splitlines():
        print("  ", line.strip())
        m = re.search(r",\s*(\d+)\.\d+", line)
        if m and driver is None:
            driver = int(m.group(1))
    return out, driver


def report_mig() -> list[str]:
    """MIG instances are addressed by UUID. Numeric CUDA_VISIBLE_DEVICES
    indices are unreliable once MIG is enabled, which is a common and
    confusing way to end up on the wrong slice."""
    out = nvidia_smi(["-L"])
    uuids = re.findall(r"MIG\s+(\S+)\s+Device\s+\d+:\s+\(UUID:\s+(MIG-[0-9a-f-]+)\)",
                       out)
    if not uuids:
        return []
    print("\nMIG instances")
    for profile, uuid in uuids:
        print(f"   {profile:<12} {uuid}")
    print("\n  Address a slice by UUID, never by index:")
    print(f"    export CUDA_VISIBLE_DEVICES={uuids[0][1]}")
    print("  One process sees exactly one slice. No NCCL, no P2P across slices.")
    return [u for _, u in uuids]


def report_torch(driver: int | None) -> int:
    try:
        import torch
    except ImportError:
        print("\ntorch not installed in this environment")
        return -1

    print(f"\ntorch {torch.__version__}  (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("  torch.cuda.is_available() is False — driver or visibility problem")
        return 1

    arch_list = torch.cuda.get_arch_list()
    print(f"  kernels compiled for: {' '.join(a for a in arch_list if a.startswith('sm_'))}")

    problems = 0
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        sm = f"sm_{major}{minor}"
        name = torch.cuda.get_device_name(i)
        desc, toolkit, min_drv = ARCH_NOTES.get(sm, ("unknown", "?", 0))
        ok = sm in arch_list

        print(f"\n  device {i}: {name}")
        print(f"    capability:  {sm}  ({desc})")
        print(f"    covered by this torch build: {'YES' if ok else 'NO'}")

        if not ok:
            problems += 1
            print(f"    -> This torch has no kernels for {sm}. Needs {toolkit}.")
            if driver and driver < min_drv:
                print(f"    -> Driver {driver} is also below the {min_drv}+ that "
                      f"{toolkit} requires. Ask the cluster admins before upgrading torch.")
            else:
                print(f"    -> Driver {driver or '?'} is new enough; upgrading torch is "
                      "sufficient. See docs/ENVIRONMENT.md §1a.")
            continue

        # Arch coverage is necessary but not sufficient — actually launch a kernel.
        try:
            x = torch.randn(64, 64, device=f"cuda:{i}", dtype=torch.bfloat16)
            torch.cuda.synchronize(i)
            _ = (x @ x).sum().item()
            free, total = torch.cuda.mem_get_info(i)
            print(f"    bf16 matmul: OK")
            print(f"    memory:      {free / 1e9:.0f} GB free / {total / 1e9:.0f} GB total")
        except RuntimeError as exc:
            problems += 1
            print(f"    bf16 matmul FAILED: {exc}")

    return problems


def report_native_jit() -> None:
    """Report whether torch 2.13's native DSL op overrides are active.

    torch/_native registers accelerated CuTeDSL/Triton/nvmath implementations
    that override default ATen kernels for bmm_outer_product, foreach_mm, norm
    (including RMSNorm), scatter_add and topk. TORCH_DISABLE_NATIVE_JIT=1 turns
    them off, and they are also skipped silently when their optional
    dependencies are absent.

    This changes which kernels run, so it changes numerics — it belongs in a
    run's identity alongside architecture and attention backend
    (docs/GATES.md, hardware policy).
    """
    import importlib.util

    disabled = os.getenv("TORCH_DISABLE_NATIVE_JIT", "0") == "1"
    deps = {name: importlib.util.find_spec(mod) is not None
            for name, mod in (("nvidia-cutlass-dsl", "cutlass"),
                              ("apache-tvm-ffi", "tvm_ffi"))}
    missing = [n for n, ok in deps.items() if not ok]

    print("\nnative DSL op overrides (torch/_native)")
    if disabled:
        print("  TORCH_DISABLE_NATIVE_JIT=1 — overrides OFF, stock ATen kernels")
        print("  Record this per run: it changes which kernels execute.")
        if not missing:
            print("  Both optional deps are present, so the flag may no longer be")
            print("  needed — worth re-testing without it, on its own, once green.")
    else:
        print("  enabled" + (f" but inactive: missing {', '.join(missing)}"
                             if missing else " with deps present"))


def report_attention() -> None:
    try:
        import torch
        import flash_attn
    except ImportError:
        print("\nflash-attn: not installed — use attn_implementation=\"sdpa\"")
        return
    print(f"\nflash-attn {flash_attn.__version__}")
    try:
        from flash_attn import flash_attn_func
        q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q)
        print("  usable on this device")
    except Exception as exc:
        print(f"  NOT usable here: {type(exc).__name__}: {exc}")
        print("  -> pass attn_implementation=\"sdpa\" instead; it costs ~15% and")
        print("     PyTorch's own fused backends cover newer architectures sooner.")


def main() -> int:
    _, driver = report_driver()
    report_mig()
    problems = report_torch(driver)
    if problems == 0:
        report_native_jit()
        report_attention()

    print()
    if problems < 0:
        print("INCOMPLETE: install torch, then re-run — device coverage is unknown.")
        return 1
    if problems:
        print(f"FAIL: {problems} device(s) unusable with this torch build.")
        print("Fix the environment before recording any numbers — and note that")
        print("results from different architectures must never be compared "
              "within one experiment (docs/GATES.md, hardware policy).")
        return 1
    print("PASS: every visible device is usable with this torch build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
