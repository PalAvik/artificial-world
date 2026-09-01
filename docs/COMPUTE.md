# Compute plan — 1 × A100 80GB

## The headline: compute is not your binding constraint

Ten weeks on one A100 is ~1,680 GPU-hours. The plan below spends **~190**. At 2B
parameters with a frozen vision tower, the GPU sits idle most of the time.

The scarce resources are **engineering time** and **the number of distinct ideas you can
test before the clock runs out**. Strictness in `GATES.md` is therefore about *decision
discipline*, not compute rationing. Do not let a comfortable GPU budget turn into an
open-ended exploration.

## Budget

| Phase | Work | GPU-hours | Wall clock |
|---|---|---|---|
| 0 | Gate 0 render read-back check | ~1 | Week 1 |
| 1 | Full metric sweep, incl. dev iteration | ~10 | Weeks 2–3 |
| 2 | 6 LoRA runs + 3 seeds + 1 full-FT confirm | ~90 | Weeks 4–7 |
| 3 | H5 downstream, cycle metrics, re-runs | ~60 | Weeks 8–10 |
| — | Slack for failed runs | ~30 | — |
| | **Total** | **~190** | 10 weeks |

## Four decisions that make this fit comfortably

**1. Freeze the vision tower.** Train the LM and the vision→LM merger only. This lets
image features be **precomputed once and cached to disk**, which removes the vision
encoder from every training step. It is also defensible on its own terms: the claim is
about how the language model integrates visual tokens, not about re-learning vision.

**2. Render spans as narrow strips, not full images.** A rendered word needs a
`~224×32px` strip, not a `448×448` canvas. Qwen3-VL's native dynamic resolution means
this costs ~20–30 visual tokens instead of several hundred. Cheaper *and* scientifically
cleaner — it minimises the token-count mismatch the merge-position design already works
around.

**3. LoRA for every ablation; full fine-tune once.** All six ablation configs run as
LoRA r=32 on attention and MLP projections. Exactly one full fine-tune, on the winning
config, as a confirmation that the effect isn't a LoRA artifact.

**4. Cap every run at 50M tokens.** A run that needs more than ~6 hours to show signal
is a run whose config is wrong. Fixed budget, no exceptions — see `GATES.md`.

**5. Pin the processor's `min_pixels`.** Left at its default it upscales a narrow strip
~35× (see the finding below), inflating both compute and the cardinality gap between
the two views. Sweep it at Gate 0 and freeze it.

## Measured so far

Smoke test on **B200** — provisional. None of it is a gated number; the A100 is the
reference machine (`docs/GATES.md`).

| Fact | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` → `Qwen3_5ForConditionalGeneration`, **multimodal** |
| Parameters | 2.21 B |
| Hidden states | 25 (24 layers + embeddings), d = 2048 |
| Vocabulary | 248,320 |
| Image processor | `Qwen2VLImageProcessor`, `patch_size 16`, `merge_size 2` |
| Peak memory, bf16 inference | 4.5 GB at bs=1, 4.8 GB at bs=8 |
| Attention | `flash_attention_2` loaded cleanly on `sm_100` |

### Resolved: `min_pixels` controls visual token count

`min_pixels` maps to the processor's `size.shortest_edge` and is an **area in
pixels**, not an edge length. At `patch_size 16 × merge_size 2` one visual token
covers `32×32 = 1024` px, so `min_pixels=1024` is "at least one token".

| `min_pixels` | Visual tokens for a `75×32` strip | V_T / V_I sequence |
|---|---|---|
| default | 71 | 25 / 96 |
| 1024 | **3** | 25 / **28** |

A 24× reduction. That second column is the one that matters: the cardinality gap is
the asymmetry the merge-position design works around (`PLAN.md` §1.1).

**Frozen at Gate 0: height 48, `min_pixels` 1024 — 6 visual tokens per span**
(`configs/render.yaml`). The 3-token configs were cheaper but could not clear the
read-back thresholds at the CI lower bound, and legibility is the constraint the
objective is subordinate to. Use **6 tokens per substituted span** in every
sequence-length and cost estimate from here; for a one-word span that leaves V_I
about 4–5 tokens longer than V_T, which is small but not parity — hence the merge
position.

Re-check or reproduce with `python scripts/gate0_sweep.py --model <base-model>`.
Treat visual-token count as an **experiment parameter**, not an incidental default:
it sets both the compute cost of every sweep and the V_T/V_I cardinality gap.

### Measured throughput — A100 80GB (the reference machine)

`seq=512`, bf16, `flash_attention_2`, `TORCH_DISABLE_NATIVE_JIT=1`:

| batch | ms/step | tok/s | GB | vs bs=1 | marginal |
|---|---|---|---|---|---|
| 1 | 162.7 | 3,147 | 4.8 | 1.0× | — |
| 4 | 164.4 | 12,459 | 5.6 | 4.0× | 4× work in **1.01× the time** — free |
| 16 | 408.6 | 20,050 | 9.0 | 6.4× | 4× work in 2.49× the time — 1.61× |
| 64 | 1474.3 | 22,226 | 22.5 | 7.1× | 4× work in 3.61× the time — 1.11× |

**The knee is at bs≈16.** Below bs=4 the GPU is almost entirely idle — quadrupling the
batch costs 1% more wall clock. Past bs=16 time grows nearly linearly with work, which
is what compute-bound looks like. bs=64 buys 11% more throughput for 2.5× the memory.

**Operating point: bs=32 for inference sweeps** — near-peak throughput with memory to
spare on an 80 GB card. Never run the Phase 1 sweep at bs=1: it would be ~7× slower
than necessary for identical results.

### The A100 is *faster* than the B200 here

| | A100 | B200 |
|---|---|---|
| bs=1, seq=25 | **126 ms** | 190 ms |
| bs=8, short seq | **1,559 tok/s** | 984 tok/s |

Not a measurement error — it is what dispatch-bound work looks like. At 2 B parameters
and short sequences the bottleneck is host-side launch cost, and the B200's extra
compute buys nothing it can use. So naming the A100 the reference machine costs this
project no speed at all; on this workload it is the better card as well as the stable
one.

### Revised phase estimates

Phase 1 is ~35k items × 4 views ≈ 140k forwards. At ~20k tok/s and ~256 tokens per
item, that is **~0.5 h of pure compute** — not the 30 GPU-hours first budgeted, which
assumed unbatched throughput. Raising the allowance to ~10 h for dev iteration and
re-runs still cuts the phase by two thirds.

Phase 2's estimate survives contact with the data. Training is roughly 3× a forward
(forward + backward), and the ISO objective runs two views, so ~6× — implying
~3.3k tok/s, against the 3.5k first assumed. With gradient checkpointing, expect
4–6 h per 50M-token run. **No change to the Phase 2 or Phase 3 budgets.**

## Throughput and memory (estimates — measure on day one and correct these)

The ISO objective needs **two forward passes per example** (V_T and V_I), so per-step
cost is roughly 2.2× a normal SFT step.

| Config | Peak mem | Est. throughput | 50M tokens |
|---|---|---|---|
| Inference, bs=32, bf16 | ~12 GB | ~800 items/s | — |
| LoRA r=32, bf16, FA2, grad-ckpt | ~30 GB | ~3.5k tok/s | ~4 h |
| Full FT, bf16 + 8-bit AdamW, grad-ckpt | ~45 GB | ~2.2k tok/s | ~6.5 h |
| Full FT, bf16 + fp32 AdamW, grad-ckpt | ~62 GB | ~2.0k tok/s | ~7 h |

Full-FT memory math (2B params): bf16 weights 4.4 GB + bf16 grads 4.4 GB + fp32 Adam
states 17.6 GB + fp32 master 8.8 GB ≈ 35 GB before activations. 8-bit AdamW drops the
optimizer states to ~4.4 GB. Both fit; use 8-bit unless it destabilises.

**Attention backend: FlashAttention-2.** One wheel covers both GPUs — FA2 builds
`sm_100` gencode on CUDA 12.8+, so a CUDA 13 build serves the A100 (`sm_80`) and the
B200 (`sm_100`) alike. Resolve it with `python scripts/install_flash_attn.py
--install`, which matches the wheel to the installed torch and then launches a kernel
to prove it works, rather than trusting that a successful `pip install` means a
working backend.

`sdpa` stays the per-machine fallback, at roughly 15% slower. Whichever a run used
goes in its `results/RESULTS.md` row: the backend is part of a run's identity under
the hardware policy, so an `sdpa` run and a `flash_attention_2` run are not
comparable to each other.

**A100 is Ampere:** bf16 yes; FP8 and FlashAttention-3 no (Hopper and later only).

## B200 and MIG: opportunistic only

B200 access here is intermittent, A100 access is reliable, so **the A100 is the
reference machine and every gated number comes from it** (`docs/GATES.md`, hardware
policy). That is not a limitation to work around — at 2B parameters with a frozen
vision tower the A100 is not the bottleneck, and a stable machine is worth more to
this project than a fast one.

Use the other hardware for work that is never compared against A100 runs:

| Hardware | Good for | Not for |
|---|---|---|
| B200, when available | Hyperparameter search for λ weights; the optional 8B scale point (as its own self-contained comparison, baseline included) | Any run in the Gate 2 ablation table |
| MIG slices | Phase 1 sweeps and independent ablations run concurrently — several slices, several jobs | Making one run faster; they can't |

If a B200 result ever needs to enter the paper, port its **baseline** to B200 too and
report the pair. A ported run without a ported baseline is not a result.

## Training config to start from

```yaml
model:            Qwen3-VL-2B          # freeze vision tower + train merger & LM
precision:        bfloat16
attn:             flash_attention_2    # fallback sdpa; record which, per run
grad_checkpoint:  true
lora:             {r: 32, alpha: 64, dropout: 0.05, targets: [q,k,v,o,gate,up,down]}
max_seq_len:      512                  # text view ~256, image view ~320
micro_batch:      8
grad_accum:       8                    # effective batch 64
lr:               1e-4                 # LoRA; use 1e-5 for full FT
schedule:         cosine, 3% warmup
token_budget:     50_000_000           # hard cap
eval_every:       500 steps            # MSG + probe accuracy, both, every time
```

## Disk

Budget **~200 GB** on the A100 box:

| Item | Size |
|---|---|
| Model weights + HF cache | ~15 GB |
| Visual Genome images | ~15 GB |
| Flickr30k images | ~4 GB |
| GQA images | ~20 GB |
| Cached vision features (Tiers A/B/C) | ~40 GB |
| Rendered Tier B corpus | ~10 GB |
| Checkpoints (LoRA adapters are small; 1 full FT ≈ 9 GB) | ~30 GB |
| Hidden-state dumps for Phase 1 analysis | ~20 GB |

Only dump hidden states for a **selected subset of layers** (suggest 8 evenly spaced,
plus the final). Dumping all 36 layers for 35k items × 4 views is ~20× larger and buys
nothing the sparse set doesn't show.
