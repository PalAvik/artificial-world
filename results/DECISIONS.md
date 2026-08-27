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
raised to 512, re-run pending. Expected winner on current evidence: h=32 /
min_pixels=4096, 8 visual tokens.

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
