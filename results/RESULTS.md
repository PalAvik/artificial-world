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

## Baselines (fill after Gate 1)

| Tier | MSG mean | 95% CI | acc(V_T) | acc(V_I) | delta | n |
|---|---|---|---|---|---|---|
| A — referential | | | | | | |
| B — orthographic | | | | | | |
| C — relational | | | | | | |
