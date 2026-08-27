# Environment setup — A100 box

Run these on the server. Nothing here needs to run in the authoring container.

## 0. Confirm the machine

```bash
nvidia-smi                      # expect A100-SXM4-80GB or A100-80GB-PCIe, driver >= 535
nvcc --version                  # needed only if you build flash-attn from source
python3 --version               # 3.10 or 3.11
df -h /data                     # need ~200 GB free (see docs/COMPUTE.md)
```

## 1. Python environment

```bash
# uv: fast and reproducible. (Use conda instead if that's your habit — see below.)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /path/to/artificial-world
uv venv --python 3.11 .venv
source .venv/bin/activate

# PyTorch, CUDA 12.4 build
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Core stack
uv pip install "transformers>=4.57.0" accelerate peft datasets \
               numpy scipy scikit-learn pandas matplotlib pillow \
               einops sentencepiece protobuf bitsandbytes \
               "huggingface_hub[cli]" wandb pyyaml tqdm

# Qwen-VL image preprocessing helpers
uv pip install qwen-vl-utils

# FlashAttention-2 (Ampere-supported). Prefer a prebuilt wheel over the ~10 min
# source build — this one matches the pinned stack exactly (cu124 / torch2.5 / cp311):
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu124torch2.5-cp311-cp311-linux_x86_64.whl

# If no wheel matches your stack, either build from source
# (uv pip install flash-attn --no-build-isolation) or skip it and use
# attn_implementation="sdpa" — it costs ~15% and is not worth fighting over.
```

Verify it loaded against the GPU, not merely installed:

```bash
python - <<'PYCHK'
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
print("flash-attn", flash_attn.__version__, "->", tuple(flash_attn_func(q, q, q).shape))
PYCHK
```

The real test is `scripts/smoke_test.py` (§6), which loads the model with
`attn_implementation="flash_attention_2"` and fails loudly if the backend is unusable.

<details>
<summary>conda equivalent</summary>

```bash
conda create -n freeflow python=3.11 -y && conda activate freeflow
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
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
container). List what's actually published first:

```bash
python - <<'PY'
from huggingface_hub import HfApi
for m in HfApi().list_models(search="Qwen3-VL", sort="downloads", direction=-1, limit=25):
    print(m.id)
print("---")
for m in HfApi().list_models(search="Qwen3.5-VL", sort="downloads", direction=-1, limit=25):
    print(m.id)
PY
```

Then download. If a small **Qwen3.5-VL** exists, prefer it and keep Qwen3-VL-2B as the
generational control (see PLAN.md §6.1).

```bash
export HF_HOME=${HF_HOME:-$HOME/data/hf}   # any user-writable path
mkdir -p "$HF_HOME"

hf download Qwen/Qwen3-VL-2B-Instruct --local-dir /data/models/Qwen3-VL-2B-Instruct
# text-only ablation baseline
hf download Qwen/Qwen3.5-2B          --local-dir /data/models/Qwen3.5-2B
```

(On older `huggingface_hub`, the command is `huggingface-cli download` with the same args.)

## 4. Datasets

**Tier B needs no download** — it is synthesized on CPU from any text corpus. That is
part of why it leads.

```bash
mkdir -p /data/datasets && cd /data/datasets

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
python scripts/smoke_test.py --model /data/models/Qwen3-VL-2B-Instruct
```

Expected: both views produce logits, hidden states come back with the layer count the
config advertises, peak memory well under 80 GB, and a wall-clock number for a single
forward pass at batch 8.

## 7. Environment record

Capture the exact environment once it works, so a result three weeks from now is
reproducible:

```bash
uv pip freeze > configs/requirements.lock.txt
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" \
    >> configs/requirements.lock.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> configs/requirements.lock.txt
git add configs/requirements.lock.txt && git commit -m "Lock environment"
```
