# Gates and drop rules

Four gates. Each has a **date**, a **number**, and a **decision**. The decision is made
on the date with the numbers that exist on that date — not a week later, and not after
"one more run".

Recording rule: `results/RESULTS.md` gets one row per run — config hash, token count,
MSG, probe accuracy, benchmark retention, wall clock, **and the GPU architecture,
torch version and attention backend** (see the hardware policy below). **No new run starts until the
previous run's row is filled in.** This is the single most effective guard against a
research project dissolving into undocumented runs.

---

## Gate 0 — end of Week 1 · Render config
**Cost: ~1 GPU-hour. Run it before writing anything else.**

Tier B assumes the model can *read* rendered spans. If it can't, every Tier B number
measures OCR failure rather than representational geometry, and the centrepiece of the
project is invalid.

This is **not** a pass/fail on a fixed config. Two knobs trade off directly against
each other:

- **Fewer visual tokens** — smaller strips, lower `min_pixels` — keep the `V_T`/`V_I`
  cardinality gap near zero and the sweep cheap.
- **More visual tokens** — bigger glyphs, higher resolution — make the span legible.

So Gate 0 is a constrained minimisation: **find the render config with the fewest
visual tokens that still clears the read-back thresholds.** Sweep font size, strip
height and `min_pixels` jointly; the smoke test reports visual-token count per config.

**Run it with:** `python scripts/gate0_sweep.py --model <base-model>`

**Metric:** exact-match read-back accuracy, case- and wrapping-punctuation-insensitive,
judged at the **lower bound of the 95% Wilson interval** rather than the point
estimate. A point estimate that grazes the threshold is not evidence of clearing it:
at n=128, 0.953 carries an interval of [0.90, 0.98]. Default n is 512, the smallest
that can resolve a 0.97 measurement against a 0.95 threshold.

**Second constraint: the config must be legible on held-out fonts too.** Gate 2(a)
evaluates the trained model on held-out fonts; if the base model cannot read them at
the frozen render config, that condition is floor-limited by OCR and measures nothing.
This checks a validity precondition of the instrument rather than tuning toward the
split — but it does look at it, so it is stated here rather than left implicit.

| Outcome | Threshold | Action |
|---|---|---|
| PASS | ≥95% single word, ≥88% three-word span, at the cheapest config meeting both | Freeze font set, strip geometry and `min_pixels` into `configs/render.yaml`. Proceed. |
| CONDITIONAL | 80–95% at every config tried | **One** wider sweep — contrast, padding, anti-aliasing, DPI. Re-test once. |
| FAIL | <80% after that sweep | Tier B cannot lead. Demote to a Tier-A-led plan and re-cost. If Tier A grounding also can't be measured cleanly → **DROP**. |

Record the chosen config *and its visual-token count* — the latter is an experiment
parameter that appears in every subsequent cost estimate, not an incidental default.
Baseline from the first smoke test: a `75×32` strip at `min_pixels=1024` costs
**3 visual tokens**, against 71 at the processor's default.

## Gate 1 — end of Week 3 · Does the gap exist?
**The first real go/no-go. Everything after this depends on the answer being yes.**

**Metrics:** normalized MSG with bootstrap 95% CI, n ≥ 2000 held-out spans per tier;
plus functional accuracy delta `|acc(V_T) − acc(V_I)|` on matched QA.

| Outcome | Threshold | Action |
|---|---|---|
| PASS | MSG ≥ 1.5 with CI lower bound > 1.25 on ≥2 of 3 tiers, **and** accuracy delta ≥ 5 pts | Proceed to Phase 2. |
| MARGINAL | MSG in 1.25–1.5 | **Exactly one week** of extension to test config sensitivity — resolution, layer choice, prompt format, pooling. No second extension. |
| **DROP** | CI upper bound < 1.25 on all tiers **and** accuracy delta < 2 pts | The phenomenon does not exist at 2B. The training program has no target to aim at. **Do not proceed to Phase 2.** Write it up as a short negative note or shelve it. |

The drop case is not a failure of execution — it is the measurement doing its job for
1% of the project's cost. Treat it as a good outcome for a bad hypothesis.

---

## Gate 2 — end of Week 7 · Is it trainable, and is the effect real?

All four conditions must hold. Any one failing has a defined consequence.

| # | Condition | Threshold |
|---|---|---|
| a | MSG reduction vs. baseline, on held-out spans **and** held-out fonts/images | ≥ 40% |
| b | Span-probe accuracy retained (anti-collapse) | within 5 pts of baseline |
| c | Standard VLM benchmark retention | within 2 pts |
| d | Render-augmentation-only ablation's MSG reduction | **< 50%** of the full objective's |

**If (d) fails — the augmentation ablation matches the full objective — the method
contribution is data augmentation, not the ISO terms. DROP the method claim.** Fall
back to publishing the Phase 1 measurement study alone. This is the single most likely
way this project produces a result that looks good and means nothing, and it is
non-negotiable: run the (d) ablation in week 5, not week 7.

**If (a) passes but (b) fails** — invariance rising as decodability falls — that is
collapse. **One** redesign attempt: raise λ_g, add the feature-reconstruction anchor if
it isn't already carrying weight. Re-test once. A second collapse → **DROP**.

**If (c) fails badly** (>5 pts) the model has been damaged; treat as a failed run, not
a result, and fix the recipe before anything else.

---

## Gate 3 — end of Week 10 · Does invariance buy anything?

**Metric:** held-out relational composition vs. a matched-compute baseline, ≥3 seeds.

| Outcome | Threshold | Action |
|---|---|---|
| PASS | ≥3 pts improvement, non-overlapping std bars across 3 seeds | Full claim. Paper includes H5. |
| FAIL | below that, or seed variance swamps the effect | **Downgrade, not drop.** Publish measurement + method, with no world-model claim. State the negative H5 result explicitly. |

A negative H5 with a solid H1–H4 is still a real paper. Overclaiming it is what makes
it not one.

---

## Global rules

1. **Two gate failures ends the project.** Not "two consecutive" — any two.
2. **A phase overrunning its wall clock by >50% is cut, not extended.** Ship what the
   phase produced and move to the next gate with it.
3. **Every gate decision gets written down on its date** in `results/DECISIONS.md`:
   the numbers, the call, and one sentence of reasoning. A gate that gets silently
   rolled forward has failed at its only job.
4. **Seed discipline under one GPU:** ablations get 1 seed, the main config gets 3.
   State this as a limitation in the paper rather than pretending to more.
5. **Held-out means held-out.** Fonts, images, and spans used in training never appear
   in a reported metric. Set the splits in Phase 0 and never revisit them.

## Hardware policy

More than one GPU architecture is available (A100, B200, MIG slices). That is
useful for throughput and dangerous for comparisons.

1. **Never split one comparison across architectures.** Every run that gets
   compared — the main config against its ablations, a trained checkpoint against
   its baseline, three seeds against each other — runs on the same architecture,
   with the same torch build and the same attention backend. Kernel and reduction
   -order differences across architectures are easily large enough to manufacture
   or mask a 40% MSG change, and a Gate 2 decision made on a mixed comparison is
   worthless.
2. **Record the architecture, torch version, attention backend and
   `TORCH_DISABLE_NATIVE_JIT` in every `results/RESULTS.md` row.** A run whose
   environment is unrecorded cannot be compared to anything later. That env var
   swaps the RMSNorm kernel, which changes reduction order and therefore the
   hidden states this project measures distances between.
3. **The A100 is the reference machine.** All gated numbers come from it. B200
   and MIG are for exploration, parameter sweeps, and independent jobs whose
   outputs are not compared against A100 runs.
4. **MIG gives concurrency, not speed.** Use slices to run independent jobs in
   parallel — the Phase 1 sweep, or several ablations at once. Do not use them to
   make a single run faster; they can't.
5. **If a run must be repeated on different hardware, repeat its baseline too.**
   A ported run without a ported baseline is not a result.

## Scope decisions already made

**Pixel-level image generation is out of scope for v1.** On one A100 in ten weeks, a
generative decoder trained to publication quality is not achievable, and pursuing it is
the most likely way to end with nothing. The anti-collapse anchor is instead
**feature-space reconstruction** — predicting the frozen vision encoder's features for
the swapped span — which provides the same structural guarantee against collapse and
the same "information retained" measurement, at a fraction of the cost.

Consequences: the `T→I→T` cycle still runs in full (it needs only read-back, no
generation). The `I→T→I` cycle runs in feature space. Pixel reconstruction moves to
future work, where it belongs given the hardware.
