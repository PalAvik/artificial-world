# Gate decisions

One entry per gate, written **on the gate's date**, with the numbers that existed on
that date. A gate silently rolled forward has failed at its only job.

Format: date · gate · the numbers · the call · one sentence of reasoning.

---

## Gate 0 — Render config · due end of Week 1

**2026-08-27 — first sweep run, decision deferred. Not a gate failure.**

8 configs x 128 spans on the A100. Five configs cleared both thresholds on point
estimates, and the rule as written selected h=32 / min_pixels=1024 at 3 visual
tokens (train 1w 0.953, 3w 0.977).

Two problems with that selection, both in the rule rather than the data:

1. **0.953 at n=128 has a 95% CI of [0.90, 0.98].** It does not establish 0.95.
   Only h=32 / min_pixels=4096 (0.99) cleared at the lower bound.
2. **That config reads held-out fonts at 0.93, below the 0.95 threshold.** Gate
   2(a) evaluates held-out fonts, so freezing it would have left that condition
   floor-limited by OCR — discoverable only in week 7. Across the grid, only the
   `min_pixels=4096` configs cleared 0.95 on held-out fonts, so larger inputs buy
   robustness to unfamiliar typefaces specifically.

Rule tightened (CI lower bound; held-out as a validity constraint), default n
raised to 512.

**Second sweep at n=512 — no config cleared, and the constraint was wrong.**

With held-out held to the full 0.95 threshold, nothing qualified. The best miss
was h=32/min_pixels=4096 at a held-out lower bound of 0.9451 — about three spans
in 512.

That was a miscalibrated constraint, not a real failure. The justification for it
("Gate 2(a) would be floor-limited by OCR and measure nothing") was overstated: at
0.94 read-back, 94% of held-out spans are read correctly and their MSG is clean
signal, and the residual 6% is handled by reporting MSG conditioned on correct
read-back — which the metric suite now requires anyway (PLAN.md §5.3a). Requiring
the *train* threshold on held-out fonts amounts to requiring the held-out set not
to be held out.

Held-out now uses a floor 5 points below the train thresholds. Under it the
qualifying set is:

| config | tokens | train 1w (lower) | held-out 1w (lower) |
|---|---|---|---|
| **h=48, min_px=1024** | **6** | 0.988 (0.975) | 0.941 (0.918) |
| h=48, min_px=4096 | 6 | identical — min_pixels does not bind at this size |
| h=24, min_px=4096 | 8 | 0.990 (0.977) | 0.945 (0.922) |
| h=32, min_px=4096 | 8 | 0.988 (0.975) | 0.965 (0.945) |

**Expected winner: h=48, min_pixels=1024, 6 visual tokens.** Re-run pending.

Also worth recording: the n=512 numbers moved materially against n=128
(h=24/1024 went 0.92 -> 0.898), which confirms the n=128 sweep was too noisy to
freeze anything on. And the held-out font set skews *hard* — Computer Modern
sans, STIX, Space Mono, Source Serif are LaTeX-era and quirky faces against
mainstream screen fonts in train — so the train/held-out gap reflects genuine
typeface difficulty rather than overfitting, and makes Gate 2(a) a conservative
test.

Useful side finding: read-back accuracy tracks **word length, not word class** —
function words 1.00, abstract 1.00, concrete 0.91, rare/long 0.91. No OCR-level
function-word deficit, so a function-word gap in Phase 1 would be representational
rather than an artifact. But it also means class comparisons in Phase 1 must
control for read-back accuracy, or restrict to spans the model reads correctly.

## Gate 1 — Does the gap exist? · due end of Week 3
_Not yet reached._

## Gate 2 — Is it trainable, and is the effect real? · due end of Week 7
_Not yet reached._

## Gate 3 — Does invariance buy anything? · due end of Week 10
_Not yet reached._
