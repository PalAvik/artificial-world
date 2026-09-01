# artificial-world / FreeFlow

Measuring **cross-modal token substitutability** in vision-language models: can a
span of a sequence be expressed as text tokens or as image tokens
interchangeably, and does the model's internal state care which?

## What this found

Substituting a word with a rendered image *of that word*, and comparing the two
views' hidden states at the first shared token after the span:

- The two views sit far apart — 2.3–5.2× the within-modality control distance,
  depending on the model.
- **That distance is largely a linear re-expression, not a difference in
  information.** A fitted out-of-fold isometry removes **53–75%** of it across
  five Qwen VLMs (2B–7B, three generations); a general linear map removes
  **76–85%** where the fit was well-determined.
- A per-modality *translation* — the standard account of the "modality gap",
  and the standard remedy — never explains the bulk of it.

A linear map is what a vision-to-language projector already implements, so the
training program this project was built to run has no target. Gate 1 called that
on 2026-08-31 and **Phase 2 is cancelled**; `results/DECISIONS.md` records it
against thresholds fixed before the numbers existed.

**Two limits stated up front.** Every valid row is a Qwen model — non-Qwen
families are unresolved, and the frozen render config may not transfer to them.
And MSG's magnitude moves ~10× with the choice of within-modality control, which
is why the headline above is the *cross-distance reduction*: the numerator does
not depend on that choice, so it needs no convention quoted beside it.

## Layout

```
freeflow/
  data/       substitution corpora — Tier B orthographic (the centrepiece),
              A referential, C relational, P pictorial; plus the vocabulary
  metrics/    the MSG suite: merge position, geometry, null hierarchy,
              forced-choice validity, probe, JSD, read-back cycle
configs/      frozen render config, sweep model list
scripts/      entrypoints — all runnable with no install step
tests/        231 CPU tests, no GPU required
docs/         gates, environment, data provenance, positioning
paper/        ICLR-format draft (abstract, introduction, method)
```

## Documents

| File | What it holds |
|---|---|
| [`TODO.md`](./TODO.md) | Current scope and what is blocked |
| [`docs/GATES.md`](./docs/GATES.md) | Gate thresholds and drop rules — the strict part |
| [`results/DECISIONS.md`](./results/DECISIONS.md) | Every gate call, on its date, with the errors that changed it |
| [`results/RESULTS.md`](./results/RESULTS.md) | One row per run |
| [`docs/POSITIONING.md`](./docs/POSITIONING.md) | What is and is not novel against the literature |
| [`docs/DATA.md`](./docs/DATA.md) | Corpora, licences, fetch commands |
| [`docs/ENVIRONMENT.md`](./docs/ENVIRONMENT.md) | Cluster setup |
| [`docs/COMPUTE.md`](./docs/COMPUTE.md) | Measured throughput and budget |
| [`PLAN.md`](./PLAN.md) | The original design. Superseded — see its header. |

## Running things

Scripts work from any directory with no install step.

```bash
pytest -q                                          # 231 tests, no GPU
python scripts/check_gpu.py                        # architecture, driver, backends
python scripts/setup_fonts.py --download           # Tier B fonts, no root needed

# validate a run configuration in seconds, before spending GPU time
python scripts/phase1_measure.py --model <id> --tiers B --n 16000 \
    --max-distinct-spans --dry-run

# measure, then read the nulls
python scripts/phase1_measure.py --model <id> --tiers B --n 16000 \
    --max-distinct-spans --out results/run
python scripts/show_offset.py --results results/run/results.json

# across models
python scripts/model_sweep.py --preflight
python scripts/model_sweep.py --n 16000
```

`pip install -e .` is optional, only for `import freeflow` from elsewhere.

## Reading a result

`show_offset.py` prints validity before geometry, and refuses to issue a verdict
from a fit that could not support one. Three things decide whether a number
means anything, and all three are printed:

- **read-back** — can the model read the rendered span at all? Below 0.95 and
  misread items contaminate the measurement.
- **span-free floor** — the same forced choice with the span blanked. Chance is
  the wrong reference; this is the right one.
- **the fit line** — whether folds held out distinct *spans* rather than rows,
  how many spans per hidden dimension, and how much within-modality structure
  the map preserved. A `[D, D]` map memorises when D exceeds the number of
  distinct spans, and extra renderings of the same word do not help.

## Hardware

One A100 80GB. Measurement is inference-only.
