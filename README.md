# artificial-world / FreeFlow

Research code for **cross-modal token substitutability** in small vision-language models:
can a span of a sequence be expressed as text tokens or as image tokens interchangeably,
without changing what the model does with it?

**Start here: [`PLAN.md`](./PLAN.md)** — problem framing, metric definitions, hypotheses,
kill criteria, and the phase plan.

## Layout

```
freeflow/
  data/       substitution corpus builders (Tier A/B/C, §2 of PLAN.md)
  metrics/    the MSG suite (§5) — build this before anything else
  models/     base-model adapters and the generation head
  train/      the ISO objective (§6)
configs/      experiment configs
scripts/      run entrypoints
tests/        unit tests (corpus builders are CPU-testable without a GPU)
docs/         notes, related-work sweep, results
```

## Documents

| File | What it holds |
|---|---|
| [`PLAN.md`](./PLAN.md) | Problem framing, metrics, hypotheses, phases |
| [`docs/GATES.md`](./docs/GATES.md) | Gate thresholds and drop rules — the strict part |
| [`docs/COMPUTE.md`](./docs/COMPUTE.md) | 1 × A100 budget, per-run configs, memory math |
| [`docs/ENVIRONMENT.md`](./docs/ENVIRONMENT.md) | Setup commands for the A100 box |
| [`results/RESULTS.md`](./results/RESULTS.md) | One row per run. No new run until the last row is filled. |
| [`results/DECISIONS.md`](./results/DECISIONS.md) | Gate calls, written on the gate's date |

## Order of work

Phase 0 builds `freeflow/data/` and `freeflow/metrics/`. Phase 1 is inference-only
measurement against an off-the-shelf VLM and is the first go/no-go. Nothing in `train/`
gets written until Phase 1 reports. See `PLAN.md` §7.

## Hardware

One A100 80GB (B200 and MIG are opportunistic only — see `docs/GATES.md` hardware
policy). The base model is a ~2B VLM; which one is a Gate 0 decision settled by
`python scripts/find_model.py --inspect ...`, because the Qwen3.5 family is natively
multimodal and only a checkpoint's `config.json` says whether it carries a vision
tower (`PLAN.md` §6.1). Pixel-level image generation is out of scope for v1 as a
direct consequence of the hardware budget; see `PLAN.md` §7.

Start with `docs/ENVIRONMENT.md`, then `python scripts/smoke_test.py` before anything else.
