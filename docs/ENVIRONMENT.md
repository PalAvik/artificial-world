# Environment setup — A100 box

Run these on the server. Nothing here needs to run in the authoring container.

## 0. Confirm the machine

```bash
nvidia-smi                      # note the GPU model AND the driver version
nvcc --version                  # needed only if you build flash-attn from source
python3 --version               # 3.10 or 3.11
df -h .                         # need ~200 GB free (see docs/COMPUTE.md)
```

Then, once torch is installed, always:

```bash
python scripts/check_gpu.py
```

It reports compute capability, driver, which architectures your torch build
actually has kernels for, MIG topology, and whether flash-attn works on *this*
device. Run it on every machine before recording any number.

## 1a. Multiple GPU architectures (A100 / B200 / MIG)

**"CUDA error: no kernel image is available for execution on the device"** is not
a bug in the model code. It means the installed torch was compiled without
kernels for that GPU's compute capability, and nothing in the training script
can fix it.

| GPU | Capability | Needs |
|---|---|---|
| A100 80GB | `sm_80` | CUDA 11.0+ — any recent torch |
| H100 | `sm_90` | CUDA 11.8+ |
| **B200** | **`sm_100`** | **CUDA 12.8+ (driver 570+); `cu124` builds will never work** |

A `cu124` torch build stops at `sm_90`, which is why `torch==2.5.1+cu124` fails on
B200 with *"no kernel image is available for execution on the device"*. One
environment can serve both machines, because CUDA 13.x still includes `sm_80`.

**On this cluster (driver 610.43.02), the default PyPI wheel is the answer:**

```bash
uv pip install --force-reinstall torch torchvision   # 2.13.0+cu130 at time of writing
uv pip uninstall flash-attn                          # the cu124 wheel can't import here
```

Confirmed arch coverage for `2.13.0+cu130`:

```
arch list: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
  sm_80  (A100): YES        sm_100 (B200): YES
```

If the driver is ever older on another node:

- **580+** — as above, default PyPI wheel (CUDA 13).
- **570–579** — CUDA 13 needs 580+, so pin cu128:
  `uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- **< 570** — B200 unusable whatever you install; that is an admin conversation.

Verify, don't assume — the arch list is the ground truth, and it reads the
compile-time constant so it works even on a node with no GPU attached:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"   # must contain sm_100
python scripts/check_gpu.py
```

### What upgrading torch breaks

- **flash-attn.** The `+cu124torch2.5` wheel is built against that exact stack and
  will not import after the upgrade. Uninstall it and re-resolve:
  `python scripts/install_flash_attn.py --install`. FA2 covers both GPUs here —
  its build defaults to `FLASH_ATTN_CUDA_ARCHS="80;90;100;110;120"` and emits
  `sm_100` gencode on CUDA 12.8+, so one CUDA 13 wheel serves A100 and B200.
  Keep `sdpa` as the per-machine fallback, and record whichever you used: the
  backend is part of a run's identity under the hardware policy, so an
  `sdpa` run and a `flash_attention_2` run are not comparable.
- **bitsandbytes 8-bit optimizers** may lag on `sm_100`. It doesn't matter there —
  B200's 180 GB makes plain fp32 AdamW affordable. Keep 8-bit for the A100.

### MIG slices

MIG partitions expose several isolated instances. Two things routinely go wrong:

```bash
nvidia-smi -L        # lists MIG UUIDs
export CUDA_VISIBLE_DEVICES=MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

**Address slices by UUID, never by numeric index** — indices are unreliable once
MIG is on, and landing on the wrong slice is silent. And **a process sees exactly
one instance**: there is no NCCL, no peer-to-peer, no multi-GPU across slices.

Slice sizing for this project:

| Slice | Fits |
|---|---|
| `1g.10gb` | Nothing useful — 2B bf16 inference alone needs ~12 GB |
| `2g.20gb` | Phase 1 inference and metric sweeps |
| `3g.40gb` | LoRA training (~30 GB peak) |
| `7g.80gb` | Full fine-tune |

MIG's real value here is **concurrency, not speed**: Phase 1's metric sweep and
the Phase 2 ablations are embarrassingly parallel, so several slices running
independent jobs beat one slice running them in sequence. See the hardware policy
in `docs/GATES.md` before splitting a comparison across machines.

## 1. Python environment

```bash
# uv: fast and reproducible. (Use conda instead if that's your habit — see below.)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /path/to/artificial-world
uv venv --python 3.11 .venv
source .venv/bin/activate

# PyTorch. The default PyPI wheel is CUDA-enabled and covers every architecture
# in this project — verified arch list for 2.13.0+cu130:
#   sm_75 sm_80 sm_86 sm_90 sm_100 sm_120   (A100 = sm_80, B200 = sm_100)
# Needs driver 580+. If yours is older, see §1a.
uv pip install torch torchvision

# Core stack
uv pip install "transformers>=4.57.0" accelerate peft datasets \
               numpy scipy scikit-learn pandas matplotlib pillow \
               einops sentencepiece protobuf bitsandbytes \
               "huggingface_hub[cli]" wandb pyyaml tqdm

# Qwen-VL image preprocessing helpers
uv pip install qwen-vl-utils

# FlashAttention-2 — the project default. The wheel must match the torch build
# exactly, so resolve it rather than guessing a URL:
#
#   uv pip uninstall flash-attn          # drop any earlier mismatched wheel
#   python scripts/install_flash_attn.py --install
#
# That script reads the installed torch/CUDA/Python, finds the matching prebuilt
# wheel, installs it, and then launches a kernel to prove it works on this GPU.
# If no wheel exists for your combination it prints the source-build command with
# the arch list already narrowed to the two GPUs in play.
```

`install_flash_attn.py --install` runs this check itself. To repeat it later:

```bash
python - <<'PYCHK'
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
print("flash-attn", flash_attn.__version__, "->", tuple(flash_attn_func(q, q, q).shape))
PYCHK
```

`scripts/smoke_test.py` (§6) then loads the model with
`attn_implementation="flash_attention_2"` and fails loudly if the backend is
unusable. Pass `--attn sdpa` to fall back on a machine where no wheel matches.

<details>
<summary>conda equivalent</summary>

```bash
conda create -n freeflow python=3.11 -y && conda activate freeflow
pip install torch torchvision
pip install "transformers>=4.57.0" accelerate peft datasets numpy scipy scikit-learn \
            pandas matplotlib pillow einops sentencepiece protobuf bitsandbytes \
            "huggingface_hub[cli]" wandb pyyaml tqdm qwen-vl-utils
pip install flash-attn --no-build-isolation
```
</details>

**Version caveat:** Qwen3-VL support landed in a recent `transformers`. If
`Qwen3VLForConditionalGeneration` doesn't import, install from main:

```bash
uv pip install --upgrade "git+https://github.com/huggingface/transformers.git"
```

## 1b. Required env var: `TORCH_DISABLE_NATIVE_JIT=1`

```bash
export TORCH_DISABLE_NATIVE_JIT=1
```

Needed for the scripts to run on this cluster. It is not a cosmetic flag — record
it with every run.

**What it disables.** torch 2.13 ships `torch/_native`, a registry of accelerated
CuTeDSL / Triton / nvmath implementations that *override* the default ATen kernels
for a specific set of ops:

| Overridden op | Relevance here |
|---|---|
| `norm` (incl. **RMSNorm**) | **Every transformer layer.** The hot one. |
| `bmm_outer_product` | attention-adjacent matmuls |
| `foreach_mm` | fused optimizer-style batched matmuls |
| `scatter_add` | gather/scatter paths |
| `topk` | sampling |

Setting the flag falls back to stock ATen for all of them. They are *also* skipped
silently when their optional dependencies (`nvidia-cutlass-dsl`, `apache-tvm-ffi`)
are absent, which is the likeliest reason the flag was needed at all.

**Why it belongs in the run log.** Different kernels for RMSNorm mean different
floating-point reduction order, so hidden states differ in the last bits. The metric
suite measures *distances between hidden states*, and Gate 2 turns on a 40% change in
one. A run with the flag and a run without it are not comparable, exactly as with
architecture and attention backend — see the hardware policy in `docs/GATES.md`.

**Worth revisiting once, later.** If the flag was a workaround for a missing optional
dependency, installing it may restore the faster kernels:

```bash
uv pip install nvidia-cutlass-dsl apache-tvm-ffi
python scripts/check_gpu.py     # reports whether the overrides are active
```

Do that as its own change, re-baseline, and never mid-experiment. It is a
performance question, not a correctness one — so it is not urgent, and it is not
worth risking a mixed comparison for.

## 2. Fonts for Tier B rendering — no root required

Pillow loads TrueType files **by absolute path**, so no system font installation is
needed. The `apt-get install fonts-*` line that was here before was unnecessary.

Two sources, neither needing sudo:

- **matplotlib's bundled TTFs** — guaranteed present, since matplotlib is already a
  dependency. Seven distinct families: DejaVu Sans/Serif/Mono, STIX General, and
  Computer Modern serif/sans/mono. Enough on their own.
- **Google Fonts extras** — downloaded into `~/.local/share/fonts/freeflow`, widening
  both the within-modality control and the render-robustness ablation axes.

```bash
python scripts/setup_fonts.py --download     # omit --download for matplotlib's only
```

Expect ~16 usable families. The script renders a sample with each one and drops any that
won't rasterise — a font file that exists but fails to render is worse than an absent
one, because it fails later and silently.

It writes `configs/fonts.yaml` with absolute paths **already split train / held-out**.
Font diversity is not cosmetic here: re-rendering a span in a different family *is*
`V_I′`, the denominator of the normalized MSG, and Gate 2 condition (a) requires
held-out fonts. The held-out set deliberately spans one serif, one sans and one mono —
a font the model never trained on should not also be a font whose whole *category* it
never trained on.

Freeze `configs/fonts.yaml` and `configs/render.yaml` once Gate 0 passes, and don't
touch them again.

### No sudo anywhere?

Nothing else in this file needs it. Keep `HF_HOME` and every dataset path under a
user-writable directory — use `~/data` in place of `/data` throughout if that applies,
and adjust the commands below to match.

## 3. Model weights

The exact repo id needs confirming (I could not reach the Hub from the authoring
container). List what's actually published, with vision models flagged:

```bash
python scripts/find_model.py
```

`pipeline_tag: image-text-to-text` means the model takes images. But that tag is
uploader-set metadata; `config.json` is what transformers actually loads, so settle
it there:

```bash
python scripts/find_model.py --inspect Qwen/Qwen3.5-2B --inspect Qwen/Qwen3-VL-2B-Instruct
```

A `vision_config` block means it sees; its absence means it does not.

> **The Qwen3.5 family is natively multimodal**, in the same shape as Gemma 3:
> `model_type: qwen3_5` is `Qwen3_5Config`, carrying `vision_config` +
> `text_config`, and resolves to `Qwen3_5ForConditionalGeneration`.
> `model_type: qwen3_5_text` is `Qwen3_5TextConfig` and resolves to
> `Qwen3_5ForCausalLM`, which is text-only. So there is **no separate
> "Qwen3.5-VL" line to hunt for** — vision is folded into the main one, and only
> the checkpoint's own config says which variant you have.
>
> If `Qwen/Qwen3.5-2B` reports `qwen3_5` with a `vision_config`, **prefer it**:
> it is the newest generation at the right size, and Qwen3-VL-2B becomes the
> generational control. If it reports `qwen3_5_text`, use Qwen3-VL-2B as the base
> and keep Qwen3.5-2B as the text-only ablation baseline.

> Note: `HfApi.list_models()` dropped the `direction` argument in
> `huggingface_hub` 1.x, where `sort="downloads"` already returns descending.
> The script detects this and passes `direction` only on older versions, where
> omitting it would silently list the *least*-downloaded repos first.

Then download. If a small **Qwen3.5-VL** exists, prefer it and keep Qwen3-VL-2B as the
generational control (see PLAN.md §6.1).

```bash
# Point these at scratch, not $HOME — quotas bite at ~200 GB (docs/COMPUTE.md).
export FREEFLOW_ROOT=${FREEFLOW_ROOT:-$PWD}
export HF_HOME="$FREEFLOW_ROOT/hf"
export MODELS="$FREEFLOW_ROOT/models"
mkdir -p "$HF_HOME" "$MODELS"

hf download Qwen/Qwen3-VL-2B-Instruct --local-dir "$MODELS/Qwen3-VL-2B-Instruct"
# text-only ablation baseline
hf download Qwen/Qwen3.5-2B           --local-dir "$MODELS/Qwen3.5-2B"
```

(On older `huggingface_hub`, the command is `huggingface-cli download` with the same args.)

## 4. Datasets

**Tier B needs no download** — it is synthesized on CPU from any text corpus. That is
part of why it leads.

```bash
mkdir -p "$FREEFLOW_ROOT/datasets" && cd "$FREEFLOW_ROOT/datasets"

# --- Tier A: Flickr30k Entities (phrase <-> box, natively span-aligned)
git clone https://github.com/BryanPlummer/flickr30k_entities.git
cd flickr30k_entities && unzip annotations.zip && cd ..
# Images require accepting the Flickr30k terms — request via the form at
# https://forms.illinois.edu/sec/229675 , or use the HF mirror if your use permits:
#   hf download nlphuji/flickr30k --repo-type dataset --local-dir flickr30k

# --- Tier A/C: Visual Genome (region descriptions + relationships)
mkdir -p visual_genome && cd visual_genome
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip -q '*.zip' && cd ..

# --- Tier C: GQA (relation annotations for the relational tier)
mkdir -p gqa && cd gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
unzip -q '*.zip' && cd ..
```

Mirrors move. If a URL 404s, search the Hub for a maintained mirror rather than hunting
the original host.

## 5. Weights & Biases

```bash
wandb login
export WANDB_PROJECT=freeflow
```

Log **MSG and probe accuracy on the same chart, every eval step**. The whole point of
the frontier plot is that collapse is visible during training, not discovered after it.

## 6. Smoke test

Run this before anything else. It confirms the model loads, both modalities work,
hidden states are reachable, and reports peak memory so `docs/COMPUTE.md` estimates can
be corrected with real numbers.

```bash
python scripts/smoke_test.py --model "$MODELS/Qwen3-VL-2B-Instruct"
```

Expected: both views produce logits, hidden states come back with the layer count the
config advertises, peak memory well under 80 GB, and a wall-clock number for a single
forward pass at batch 8.

## 7. Environment record

Capture the exact environment once it works, so a result three weeks from now is
reproducible:

```bash
uv pip freeze > configs/requirements.lock.txt
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))" \
    >> configs/requirements.lock.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> configs/requirements.lock.txt
git add configs/requirements.lock.txt && git commit -m "Lock environment"
```
