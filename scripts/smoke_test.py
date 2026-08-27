#!/usr/bin/env python3
"""Gate-0 prerequisite: confirm the model loads and both modalities are reachable.

Run this before writing anything else. It answers four questions:
  1. Does the model load in bf16 with the attention backend we plan to use?
  2. Do text-only and image-bearing prompts both produce logits?
  3. Are per-layer hidden states reachable (the metric suite depends on them)?
  4. What is peak memory and per-forward wall clock, so docs/COMPUTE.md can be
     corrected with real numbers instead of estimates?

Usage:
    python scripts/smoke_test.py --model /data/models/Qwen3-VL-2B-Instruct
"""
import argparse
import time

import torch
from PIL import Image, ImageDraw, ImageFont


def find_font(size: int):
    """Resolve a TrueType font without assuming any system font packages.

    Order: the frozen font set if setup_fonts.py has run, then matplotlib's
    bundled TTFs (always present, since matplotlib is a dependency), then
    Pillow's bitmap default as a last resort. No root, no fontconfig.
    """
    try:
        import yaml
        with open("configs/fonts.yaml") as fh:
            cfg = yaml.safe_load(fh)
        path = next(iter(cfg["train"].values()))
        return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        import matplotlib
        path = os.path.join(os.path.dirname(matplotlib.__file__),
                            "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        print("  ! no TrueType font resolved — run scripts/setup_fonts.py before Gate 0")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn", default="flash_attention_2",
                    help='"flash_attention_2" or "sdpa" if the flash-attn build failed')
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__} (cuda {torch.version.cuda})")

    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"\nloading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
        device_map="cuda:0",
        trust_remote_code=True,
    ).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded: {n_params / 1e9:.2f}B params, attn={args.attn}")

    # The paired view that the whole project is built on: same context, one span
    # expressed as text tokens in V_T and as pixels in V_I.
    span = "on the"
    v_t = [{"role": "user", "content": [
        {"type": "text", "text": f"a cat sat {span} mat. What is the cat on?"}]}]
    v_i = [{"role": "user", "content": [
        {"type": "text", "text": "a cat sat "},
        {"type": "image", "image": render_span(span)},
        {"type": "text", "text": " mat. What is the cat on?"}]}]

    results = {}
    for name, msgs in (("V_T (text span)", v_t), ("V_I (rendered span)", v_i)):
        print(f"\n{name}")
        try:
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            images = [c["image"] for m in msgs for c in m["content"]
                      if c.get("type") == "image"] or None
            inputs = processor(text=[text], images=images,
                               return_tensors="pt").to("cuda:0")
        except Exception as exc:  # processor APIs vary across VLM families
            print(f"  FAIL preprocessing: {type(exc).__name__}: {exc}")
            print("  -> check the model card's processor example and adapt this script")
            return 1

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                model(**inputs, output_hidden_states=True)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 3

        n_layers = len(out.hidden_states)
        hidden = out.hidden_states[-1]
        print(f"  seq len:       {inputs['input_ids'].shape[1]}")
        print(f"  logits:        {tuple(out.logits.shape)}")
        print(f"  hidden states: {n_layers} layers, last {tuple(hidden.shape)}")
        print(f"  peak mem:      {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        print(f"  fwd (bs=1):    {dt * 1e3:.0f} ms")
        results[name] = (n_layers, inputs["input_ids"].shape[1])

    # The image view must be longer — if it isn't, the image never entered the sequence.
    len_t = results["V_T (text span)"][1]
    len_i = results["V_I (rendered span)"][1]
    print(f"\nvisual tokens contributed by the rendered span: {len_i - len_t}")
    if len_i <= len_t:
        print("FAIL: image view is not longer than text view — the image was dropped")
        return 1
    if len_i - len_t > 200:
        print("WARN: rendered strip costs >200 tokens — shrink the strip (COMPUTE.md #2)")

    print("\nPASS — model loads, both modalities reachable, hidden states available.")
    print("Record the throughput numbers above in docs/COMPUTE.md, replacing the estimates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
