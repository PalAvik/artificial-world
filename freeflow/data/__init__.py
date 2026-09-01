"""Substitution corpus builders. See PLAN.md section 2.

  views.py    SpanItem, contexts, and the within-modality control convention
  render.py   glyph rendering at the frozen Gate 0 config + the font split
  vocab.py    Tier B span vocabulary, stratified by word class
  tier_b.py   orthographic: glyph rendering of any span, incl. function words
  tier_c.py   relational: programmatic diagrams of spatial relations
  tier_a.py   referential: phrase <-> grounded region (Flickr30k, Visual Genome)

Tier B is built first: unlimited, exactly ground-truthed, and information-preserving
by construction, so it isolates representational routing from every other confound.
"""

from . import render, tier_a, tier_b, tier_c, views, vocab  # noqa: F401
from .views import ControlKind, SpanItem, batch_by_suffix, validate  # noqa: F401

__all__ = ["render", "tier_a", "tier_b", "tier_c", "views", "vocab",
           "SpanItem", "ControlKind", "batch_by_suffix", "validate"]
