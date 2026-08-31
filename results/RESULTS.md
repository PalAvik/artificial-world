# Run log

**Rule: no new run starts until the previous run's row is filled in.** See `docs/GATES.md`.

| Date | Run ID | Phase | Config | Arch | torch | Attn | JIT | Tokens | MSG (held-out) | Probe acc | Bench retention | GPU-h | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-27 | smoke-b200 | 0 | smoke, min_pixels=1024 | B200 | 2.13.0+cu130 | fa2 | off | — | — | — | — | ~0 | 3 visual tokens; overhead-bound |
| 2026-08-27 | smoke-a100 | 0 | smoke + bench, seq=512 | A100 | 2.13.0+cu130 | fa2 | off | — | — | — | — | ~0 | knee at bs≈16; 20k tok/s |
| 2026-08-27 | gate0-n512 | 0 | sweep 8 cfg x 512 spans | A100 | 2.13.0+cu130 | fa2 | off | — | — | — | — | 0.2 | **GATE 0 PASS** h=48/min_px=1024, 6 tok |

Arch / torch / attention backend are not bookkeeping: runs compared against each
other must match on all three. See the hardware policy in `docs/GATES.md`.

## Environment of record

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` (`qwen3_5`, `Qwen3_5ForConditionalGeneration`, 2.21 B) |
| Reference GPU | A100 80GB — all gated numbers |
| torch | 2.13.0+cu130 |
| Attention | flash_attention_2 |
| `TORCH_DISABLE_NATIVE_JIT` | 1 (stock ATen kernels; see docs/ENVIRONMENT.md §1b) |
| Processor `min_pixels` | 1024 (frozen at Gate 0) |
| Render config | height 48, pad 4 — **6 visual tokens/span** (frozen at Gate 0) |
| Tier B read-back | train 0.988 / held-out 0.941 (1-word) |
| Generational control | `Qwen3-VL-2B` |

Any run departing from this row records the departure in its own row.

## Baselines — Phase 1, 2026-08-27

n=2000/tier for MSG, 256 for the forced choice. A100, `Qwen/Qwen3.5-2B`, fa2,
`TORCH_DISABLE_NATIVE_JIT=1`, seed 0, render config frozen at Gate 0.

| Tier | MSG | 95% CI | acc(V_T) | acc(V_I) | delta | span-free floor | n | valid? |
|---|---|---|---|---|---|---|---|---|
| A — referential | not run | | | | | | | needs Flickr30k/VG |
| B — orthographic | **3.023** | [2.953, 3.096] | 0.941 | 1.000 | −0.059 | 0.449 | 2000 | **yes** |
| C — relational | 4.052 | [3.981, 4.127] | 0.992 | 0.477 | +0.516 | 0.516 | 2000 | **no** — image at floor |

**Reading the two valid columns.** Tier B's forced choice is sound: the floor of
0.449 says the choice is not decidable without the span, and both views sit far
above it. Its delta is not usable — the image view is saturated at 1.000, so the
number bounds the functional gap near zero rather than measuring it. Tier C's
image view is *at* its floor (0.477 vs 0.516) while its text view scores 0.992,
so the diagrams carry nothing and MSG 4.052 measures the instrument.

**MSG denominator.** Both rows use `ControlKind.SURFACE` — a capitalisation
variant. So "3.023" means the modality gap is 3x the distance between `wisdom`
and `WISDOM`. The synonym control (`--control synonym`) is the stronger
comparison and has not been run.

### The gap against progressively weaker nulls

| Tier | raw | offset-free | share explained by a constant offset | final-layer CKA |
|---|---|---|---|---|
| B | 3.023 [2.953, 3.096] | 3.003 [2.946, 3.059] | 1.0% | 0.906 |
| C | 4.052 [3.981, 4.127] | 3.377 [3.327, 3.426] | 22.1% | 0.709 |

H3 is falsified for Tier B: the gap is not a translation. Rotation-free and
linear-map-free MSG are implemented (`msg_procrustes`, `msg_linear`) and **not
yet run** — they decide the Gate 1 extension, because a gap a fitted linear map
removes is one a standard projector already closes.

### Per-layer, Tier B

| layer | cross raw | cross offset-free | offset norm | CKA |
|---|---|---|---|---|
| 0 | −0.0000 | 0.0000 | 0.000 | 1.000 |
| 3 | 0.2412 | 0.2982 | 1.603 | 0.990 |
| 7 | 0.3018 | 0.4340 | 2.672 | 0.978 |
| 10 | **0.3781** | 0.5008 | 3.214 | 0.959 |
| 14 | 0.3245 | 0.4772 | 4.108 | 0.963 |
| 17 | 0.2247 | 0.3068 | 6.151 | 0.979 |
| 21 | 0.1659 | 0.3316 | 8.883 | 0.963 |
| 24 | 0.1485 | 0.4029 | 50.065 | 0.906 |

The gap peaks at layer 10 and closes by the output — the two views diverge
mid-stack and reconcile, which is consistent with the image view answering the
forced choice perfectly. The layer-24 offset norm of 50.065 is a 6x jump on
layer 21 and is the usual final-layer outlier scale, not a finding.

Tier C's layer 0 row (offset-free 1.000, CKA 0.000) is degenerate rather than
meaningful: with 5 unique spans every merge-position embedding is identical, so
mean-subtraction leaves the zero vector. It is a symptom of that tier's low
diversity, which is a second reason not to trust its numbers.

## Run log

| date | run | tiers | n | arch | torch | attn | JIT | outcome |
|---|---|---|---|---|---|---|---|---|
| 2026-08-27 | Gate 0 sweep (n=128) | B | 128 | A100 | 2.13.0+cu130 | fa2 | 1 | too noisy to freeze; rule tightened |
| 2026-08-27 | Gate 0 sweep (n=512) | B | 512 | A100 | 2.13.0+cu130 | fa2 | 1 | no config cleared; constraint miscalibrated |
| 2026-08-27 | Gate 0 sweep (final) | B | 512 | A100 | 2.13.0+cu130 | fa2 | 1 | **PASS** — h=48, min_px=1024, 6 tokens |
| 2026-08-27 | Phase 1 #1 | B, C | 2000 | A100 | 2.13.0+cu130 | fa2 | 1 | Tier C read-back 0.000 — category error in the check |
| 2026-08-27 | Phase 1 #2 | B, C | 2000 | A100 | 2.13.0+cu130 | fa2 | 1 | forced choice mis-specified (image > text) |
| 2026-08-27 | Phase 1 #3 | B, C | 2000 | A100 | 2.13.0+cu130 | fa2 | 1 | same, after rephrasing; PMI + floor added |
| 2026-08-27 | Phase 1 #4 | B, C | 2000 | A100 | 2.13.0+cu130 | fa2 | 1 | **numbers above.** Gate 1 not passed, not a drop |
| 2026-08-27 | offset inspection | B, C | — | — | — | — | — | H3 falsified for B (1.0%) |
