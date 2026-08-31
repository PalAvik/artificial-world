# TODO

The working task list. `results/DECISIONS.md` records what was decided and why;
this records what is left. One line per task, grouped by what blocks what.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done ·
`[!]` blocked or broken

**The clock.** Today is **2026-08-31**. Gate 1 is due **2026-09-16** — sixteen
days. Phase 1 ran early, so the one-week extension `docs/GATES.md` allows is
still unspent and stays in reserve.

---

## Blocking Gate 1 (2026-09-16)

On today's numbers Gate 1 would not pass: one instrument-valid tier of the two
required, and no positive functional delta. It is nowhere near a DROP — that
needs a CI upper bound below 1.25 on every tier, against Tier B's 3.096.

- [!] **The linear-map null is untestable with the current corpus.** A `[D, D]`
      map memorises whenever `D >= (distinct spans)`; D = 2048 against 150 words.
      The 2026-08-31 run returned MSG 0.006 — the mapped cross distance below the
      within-modality control, which nothing legitimate produces. Folds are now
      grouped by span and the guard counts distinct content, but the fix is a
      corpus one: **the vocabulary must reach several thousand distinct spans**
      (~8000 for 2 constraints per dimension) before the null can be tested.
      Blocks the Gate 1 decision, since the drop rule turns on this number.
- [ ] **Expand `freeflow/data/vocab.py` past 150 words**, keeping the four word
      classes and length matching. This is now the highest-value corpus work.
- [x] ~~Run the linear-map-free MSG at n >= 8000.~~ Run 2026-08-31; result
      withdrawn as leakage.
- [ ] **Get a second valid tier.** PASS needs >=2 of 3. Only Tier B qualifies.
      Tier A is the only route inside the window — Tier C needs a redesign *and*
      re-validation. Needs Flickr30k Entities or Visual Genome on disk
      (`docs/ENVIRONMENT.md` §4), then `--tiers A`.
- [ ] **Re-run Tier B with the synonym control.** The headline rests on a
      capitalisation flip: MSG 3.023 currently means 3x the distance between
      `wisdom` and `WISDOM`. `--control synonym --out results/phase1_synonym`.
      Cheap, and it only restates the number — lower priority than the two
      above if GPU time is tight.
- [ ] **Make the Gate 1 decision on 2026-09-16** and record it in
      `results/DECISIONS.md` on the day, per the drop rule pre-registered there.

## Broken, and known to be

- [!] **Tier C diagrams are unreadable to the model.** Image-view forced choice
      0.477 against a floor of 0.516, while the text view scores 0.992 — the
      picture carries nothing, so MSG 4.052 measures the instrument. Needs
      labelled markers or an explicit arrow rather than abstract filled/outlined
      boxes. Second defect in the same tier: only **5 unique spans** across 2000
      items, which also makes its layer-0 geometry degenerate.
- [!] **The forced choice saturates on Tier B.** Image view at 1.000 has no
      headroom, so the delta bounds the functional gap near zero rather than
      measuring it. Needs harder distractors — same word class, matched length
      and frequency — before any functional delta can be quoted.
- [ ] **Tier A has never been run.** `freeflow/data/tier_a.py` exists; the data
      does not.

## Done

- [x] Environment: torch 2.13.0+cu130 (A100 sm_80 + B200 sm_100), flash-attn 2
      by default, no-sudo fonts, `TORCH_DISABLE_NATIVE_JIT=1`.
- [x] Base model confirmed multimodal — `Qwen/Qwen3.5-2B`, loaded with
      `AutoModelForImageTextToText`.
- [x] Glyph renderer; render config frozen at Gate 0 (`configs/render.yaml`,
      `configs/fonts.yaml`). Changing either invalidates prior measurements.
- [x] Train/held-out font split fixed and not revisited.
- [x] **Gate 0 PASSED** (h=48, pad 4, min_pixels 1024 — 6 visual tokens/span).
- [x] Smoke test and throughput: knee at batch 16, ~20k tok/s on the A100.
- [x] Metric suite: MSG (ratio-of-means + bootstrap CI), teacher-forced JSD,
      geometry (cosine / offset-free / CKA / per-layer), span-identity probe,
      `T→I→T` read-back cycle, forced-choice comprehension with a span-free
      floor, out-of-fold Procrustes and ridge maps.
- [x] Corpora for Tiers B and C; Phase 1 driver.
- [x] **H3 falsified for Tier B** — a constant offset explains 1.0% of the gap.
- [x] Phase 1 baselines and run log filled in `results/RESULTS.md`.

## Paper

- [~] **Draft in `paper/`** — ICLR 2026 format, abstract + introduction + method,
      compiles to 5 pages. Written to be true whether or not the gap survives a
      linear map, since it describes an instrument rather than a result.
- [ ] **Related-work sweep.** Blocks the citations, and is the work most likely
      to change the introduction's claims — particularly the assertion that
      existing modality-gap work reports unnormalised distances.
- [ ] **Citations.** `paper/references.bib` is empty on purpose and lists the
      claims currently standing unsupported. No invented references.
- [ ] **Results and conclusions**, once the linear-map run lands.

## Ongoing

- [ ] **`results/RESULTS.md` gets a row per run, before the next run starts.**
      This slipped through four Phase 1 runs and was backfilled on 2026-08-31.
- [ ] **Keep this file and `results/DECISIONS.md` current.** DECISIONS.md is
      append-only and dated; this file is rewritten.

## After Gate 1 — not started, and moot if the gap turns out to be linear

Phase 2 (train): ISO objective · LoRA harness with frontier logging · cached
vision features · main run + 5 ablations · 3 seeds · benchmark retention eval ·
**Gate 2, 2026-10-14**.

Phase 3 (downstream): held-out relational composition eval (H5) · H5 vs
matched-compute baseline · invariance–informativeness frontier figure ·
`I→T→feature` rate–distortion curves · related-work sweep · paper draft ·
**Gate 3, 2026-11-04**.

## Known process failures, kept visible

1. **Two criteria written, hit, and then judged miscalibrated** — Gate 0's
   held-out threshold, and the forced choice's negative-delta rule. Both
   relaxations were defensible and neither changed a verdict in my favour, but
   the pattern is the warning sign. A third means doubting the experimenter.
2. **The Gate 1 read was first dated 2026-08-27 (four days early) and spent the
   one-week extension that nothing needed.** Corrected 2026-08-31. For a project
   whose discipline is "the decision is made on the date", getting the date
   wrong is not a clerical matter.
3. **`RESULTS.md` sat empty through four runs** despite the recording rule
   saying no run starts until the previous row is filled.
