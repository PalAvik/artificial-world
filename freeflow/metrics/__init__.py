"""Metric suite for cross-modal substitution. See PLAN.md section 5.

Build order (nothing downstream works without these):
  msg.py        normalized Modality Substitution Gap (5.3) - the headline number
  distribution.py  teacher-forced symmetric JSD between paired views (5.1)
  geometry.py   merge-position hidden-state distances, raw and offset-free (5.2)
  probe.py      linear span-identity probe: the anti-collapse control (5.4)
  cycle.py      T->I->T and I->T->I rate-distortion curves (5.5)

Every metric operates on a *paired view*: the same context with one span rendered as
text tokens (V_T) and as image tokens (V_I), plus the within-modality controls
(V_T' paraphrase, V_I' re-render) that form the MSG denominator.
"""
