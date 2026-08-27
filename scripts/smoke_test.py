#!/usr/bin/env python3
"""Gate-0 prerequisite: confirm the model loads and both modalities are reachable.

Run this before writing anything else. It answers five questions:
  1. Does the model load in bf16 with the attention backend we plan to use?
  2. Can it actually take images, or is this a text-only checkpoint?
  3. Are per-layer hidden states reachable (the metric suite depends on them)?
  4. How many visual tokens does a rendered span really cost, and is the
     processor silently upscaling the strip to hit a minimum resolution?
  5. What is throughput at a realistic batch size, so docs/COMPUTE.md can be
     corrected with measurements instead of estimates?

Usage:
    python scripts/smoke_test.py --model Qwen/Qwen3.5-2B
    python scripts/smoke_test.py --model Qwen/Qwen3.5-2B --min-pixels 1024
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from PIL import Image, ImageDraw, ImageFont

FONT_IS_FALLBACK = False


def find_font(size: int):
    """Resolve a TrueType font without assuming any system font packages.

    Order: the frozen font set if setup_fonts.py has run, then matplotlib's
    bundled TTFs (always present, since matplotlib is a dependency), then
    Pillow's bitmap default. No root, no fontconfig.
    """
    global FONT_IS_FALLBACK
    try:
        import yaml
        with open("configs/fonts.yaml") as fh:
            cfg = yaml.safe_load(fh)
        return ImageFont.truetype(next(iter(cfg["train"].values())), size)
    except Exception:
        pass
    try:
        import matplotlib
        path = os.path.join(os.path.dirname(matplotlib.__file__),
                            "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        FONT_IS_FALLBACK = True
        return ImageFont.load_default()


def render_span(text: str, height: int = 32, pad: int = 6) -> Image.Image:
    """Render a text span as a narrow strip — the Tier B substitution primitive.

    Narrow strips rather than square canvases: see docs/COMPUTE.md decision 2.
    """
    font = find_font(height - 2 * pad)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = int(probe.textlength(text, font=font)) + 2 * pad
    img = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(img).text((pad, pad), text, fill="black", font=font)
    return img


def describe_processor(processor) -> None:
    """Print the knobs that control visual token count.

    Visual tokens are the dominant cost of the image view and the main source
    of cardinality asymmetry against the text view, so these are not incidental
    settings — they are experiment parameters.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    knobs = {k: getattr(ip, k) for k in
             ("min_pixels", "max_pixels", "patch_size", "merge_size", "size",
              "do_resize", "temporal_patch_size")
             if hasattr(ip, k)}
    print(f"  image processor: {type(ip).__name__}")
    for k, v in knobs.items():
        print(f"    {k}: {v}")


def build_inputs(processor, msgs, batch: int):
    text = processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    images = [c["image"] for m in msgs for c in m["content"]
              if c.get("type") == "image"]
    return processor(text=[text] * batch,
                     images=(images * batch) or None,
                     padding=True, return_tensors="pt").to("cuda:0")


def timed_forward(model, inputs, reps: int = 3) -> tuple[float, float]:
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(**inputs, output_hidden_states=False)   # warm up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(**inputs, output_hidden_states=False)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, torch.cuda.max_memory_allocated() / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn", default="flash_attention_2",
                    help='Attention backend. Default flash_attention_2 — resolve a '
                         'matching wheel with scripts/install_flash_attn.py. Pass '
                         '"sdpa" to fall back. Part of the run\'s identity under the '
                         'hardware policy in docs/GATES.md, so record it.')
    ap.add_argument("--batch", type=int, default=8,
                    help="batch size for the throughput measurement")
    ap.add_argument("--min-pixels", type=int, default=None,
                    help="processor min_pixels. Raise the floor and a narrow strip "
                         "gets upscaled, costing many more visual tokens than its "
                         "content needs.")
    ap.add_argument("--max-pixels", type=int, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__} (cuda {torch.version.cuda})")

    from transformers import AutoConfig, AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    print(f"\nloading {args.model} ...")

    # Catch the text-only trap before it becomes an unreadable wall of config
    # names. Ask the same mapping AutoVLM resolves through (PLAN.md §6.1).
    cfg = AutoConfig.from_pretrained(args.model)
    model_type = getattr(cfg, "model_type", None)
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES as VLM_TYPES)
        can_see = model_type in VLM_TYPES
    except ImportError:
        can_see = None  # unknown: warn, but never block on a heuristic

    if can_see is False:
        print(f"  FAIL: model_type '{model_type}' is not an image-text-to-text "
              "model — it cannot take images.")
        print("  Run scripts/find_model.py --inspect to pick one that can.")
        return 1
    if can_see is None and not any(
            hasattr(cfg, a) for a in ("vision_config", "vision_encoder", "visual")):
        print(f"  ! {type(cfg).__name__} exposes no obvious vision config — "
              "continuing, but expect the image view to fail below.")

    proc_kwargs = {k: v for k, v in
                   (("min_pixels", args.min_pixels), ("max_pixels", args.max_pixels))
                   if v is not None}
    try:
        processor = AutoProcessor.from_pretrained(args.model, **proc_kwargs)
    except (TypeError, ValueError) as exc:
        print(f"  ! processor rejected {list(proc_kwargs)}: {exc}")
        print("    loading with defaults; see the knob names printed below")
        processor = AutoProcessor.from_pretrained(args.model)

    load_kwargs = dict(attn_implementation=args.attn, device_map="cuda:0")
    try:
        # `torch_dtype` was renamed to `dtype`. Retry only on the rename itself;
        # a blanket retry would re-run a slow load for an unrelated TypeError.
        model = AutoVLM.from_pretrained(args.model, dtype=torch.bfloat16, **load_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoVLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                        **load_kwargs)
    model = model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded: {type(model).__name__}, {n_params / 1e9:.2f}B params, "
          f"attn={args.attn}")
    describe_processor(processor)

    # The paired view the whole project is built on: same context, one span as
    # text tokens in V_T and as pixels in V_I.
    span = "on the"
    strip = render_span(span)
    v_t = [{"role": "user", "content": [
        {"type": "text", "text": f"a cat sat {span} mat. What is the cat on?"}]}]
    v_i = [{"role": "user", "content": [
        {"type": "text", "text": "a cat sat "},
        {"type": "image", "image": strip},
        {"type": "text", "text": " mat. What is the cat on?"}]}]

    seq_lens = {}
    for name, msgs in (("V_T (text span)", v_t), ("V_I (rendered span)", v_i)):
        print(f"\n{name}")
        try:
            single = build_inputs(processor, msgs, 1)
        except Exception as exc:
            print(f"  FAIL preprocessing: {type(exc).__name__}: {exc}")
            print("  -> check the model card's processor example and adapt this script")
            return 1

        with torch.no_grad():
            out = model(**single, output_hidden_states=True)
        seq_lens[name] = single["input_ids"].shape[1]
        print(f"  seq len:       {seq_lens[name]}")
        print(f"  logits:        {tuple(out.logits.shape)}")
        print(f"  hidden states: {len(out.hidden_states)} layers, "
              f"last {tuple(out.hidden_states[-1].shape)}")

        dt1, mem1 = timed_forward(model, single)
        batched = build_inputs(processor, msgs, args.batch)
        dtb, memb = timed_forward(model, batched)
        bs, sl = batched["input_ids"].shape
        print(f"  bs=1:          {dt1 * 1e3:7.1f} ms   {mem1:5.1f} GB")
        print(f"  bs={bs:<11d}{dtb * 1e3:7.1f} ms   {memb:5.1f} GB   "
              f"{bs * sl / dtb:,.0f} tok/s")
        del batched
        torch.cuda.empty_cache()

    len_t, len_i = seq_lens["V_T (text span)"], seq_lens["V_I (rendered span)"]
    visual = len_i - len_t
    print(f"\nrendered strip: {strip.size[0]}x{strip.size[1]} px "
          f"-> {visual} visual tokens")
    if visual <= 0:
        print("FAIL: image view is not longer than text view — the image was dropped")
        return 1

    # Each visual token covers (patch_size * merge_size)^2 pixels. If the strip's
    # own area implies far fewer tokens than we got, the processor upscaled it to
    # satisfy a minimum resolution — paying for pixels the span does not contain.
    vc = getattr(cfg, "vision_config", None)
    px = getattr(vc, "patch_size", 16) * getattr(vc, "spatial_merge_size", 2)
    native = max(1, (strip.size[0] // px) * (strip.size[1] // px))
    print(f"  native cost at {px}x{px} px/token would be ~{native} tokens")
    if visual > 3 * native:
        print(f"  ! the processor is upscaling this strip ~{visual / native:.0f}x.")
        print("    Lower --min-pixels (or widen the strip so upscaling is moot):")
        print("    every wasted visual token costs compute AND widens the")
        print("    cardinality gap between the two views for no added content.")

    print()
    if FONT_IS_FALLBACK:
        print("INCOMPLETE: rendered with Pillow's bitmap fallback, not a real font.")
        print("The visual-token count and any read-back accuracy measured here are")
        print("not meaningful. Run scripts/setup_fonts.py --download, then re-run.")
        return 1
    print("PASS — model loads, both modalities reachable, hidden states available.")
    print("Record the throughput numbers above in docs/COMPUTE.md, and the model id,")
    print("architecture and attention backend in results/RESULTS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
