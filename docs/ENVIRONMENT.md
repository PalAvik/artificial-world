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

# FlashAttention-2 (Ampere-supported). Takes ~10 min to build.
# If this fails, skip it — use attn_implementation="sdpa", costs ~15%.
uv pip install flash-attn --no-build-isolation
```

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

## 2. Fonts for Tier B rendering

Tier B needs a controlled font set, and font diversity is the within-modality control in
the MSG denominator — so install several deliberately.

```bash
sudo apt-get update && sudo apt-get install -y \
    fonts-dejavu-core fonts-liberation2 fonts-noto-core fonts-freefont-ttf
fc-list | wc -l                  # sanity check
fc-list : file family | grep -iE "dejavu|liberation|noto" | head
```

Pin the exact font file paths in `configs/render.yaml` in Phase 0 and never change them
after Gate 0 — the render config is frozen once it passes.

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
export HF_HOME=/data/hf
mkdir -p $HF_HOME

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
