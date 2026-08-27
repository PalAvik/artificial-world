# Compute plan — 1 × A100 80GB

## The headline: compute is not your binding constraint

Ten weeks on one A100 is ~1,680 GPU-hours. The plan below spends **~210**. At 2B
parameters with a frozen vision tower, the GPU sits idle most of the time.

The scarce resources are **engineering time** and **the number of distinct ideas you can
test before the clock runs out**. Strictness in `GATES.md` is therefore about *decision
discipline*, not compute rationing. Do not let a comfortable GPU budget turn into an
open-ended exploration.

## Budget

| Phase | Work | GPU-hours | Wall clock |
|---|---|---|---|
| 0 | Gate 0 render read-back check | ~1 | Week 1 |
| 1 | Full metric sweep, incl. dev iteration | ~30 | Weeks 2–3 |
| 2 | 6 LoRA runs + 3 seeds + 1 full-FT confirm | ~90 | Weeks 4–7 |
| 3 | H5 downstream, cycle metrics, re-runs | ~60 | Weeks 8–10 |
| — | Slack for failed runs | ~30 | — |
| | **Total** | **~210** | 10 weeks |

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

**A100 is Ampere:** bf16 and FlashAttention-2 yes; FP8 and FlashAttention-3 no (Hopper
only). If the flash-attn build fails, `attn_implementation="sdpa"` costs perhaps 15%
and is not worth fighting over.

## Training config to start from

```yaml
model:            Qwen3-VL-2B          # freeze vision tower + train merger & LM
precision:        bfloat16
attn:             flash_attention_2    # fallback: sdpa
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
