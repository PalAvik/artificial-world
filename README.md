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

## Order of work

Phase 0 builds `freeflow/data/` and `freeflow/metrics/`. Phase 1 is inference-only
measurement against an off-the-shelf VLM and is the first go/no-go. Nothing in `train/`
gets written until Phase 1 reports. See `PLAN.md` §7.

## Environment note

The base model is a ~2B VLM (see `PLAN.md` §6.1 — note that `Qwen/Qwen3.5-2B` is text-only
and cannot host this experiment as-is). Training and inference run on external GPU hardware;
this repo holds code, configs, and results.
