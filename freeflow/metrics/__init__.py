"""Metric suite for cross-modal substitution. See PLAN.md section 5.

  geometry.py      merge-position distances, raw and offset-free (5.2)
  distribution.py  symmetric JSD over the shared continuation (5.1)
  msg.py           normalized Modality Substitution Gap (5.3) - the headline
  probe.py         linear span-identity probe: anti-collapse control (5.4)
  aggregate.py     breakdowns, read-back conditioning (5.3a), bootstrap CIs
  runner.py        the GPU-facing half: forwards, merge positions, capture
  cycle.py         T->I->T and I->T->I rate-distortion curves (5.5) - TODO

Every metric operates on a *paired view*: the same context with one span rendered as
text tokens (V_T) and as image tokens (V_I), plus the within-modality controls
(V_T' paraphrase, V_I' re-render) that form the MSG denominator.

The math is deliberately separated from the model plumbing so it can be tested on
CPU: tests/test_metrics.py pins the properties the plan relies on, including that
offset-free distance is blind to a pure translation and that MSG equals 1 when
crossing modalities costs exactly what rephrasing costs.
"""

from . import aggregate, distribution, geometry, msg, probe, runner  # noqa: F401

__all__ = ["aggregate", "distribution", "geometry", "msg", "probe", "runner"]
