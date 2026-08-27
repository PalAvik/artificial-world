"""Substitution corpus builders. See PLAN.md section 2.

  tier_a.py   referential: phrase <-> grounded region (Flickr30k Entities, Visual Genome)
  tier_b.py   orthographic: glyph rendering of any span, incl. function words
  tier_c.py   relational: programmatic scenes encoding relations (CLEVR-style, GQA)
  views.py    paired-view container + within-modality control generation

Tier B is built first: unlimited, exactly ground-truthed, and information-preserving
by construction, so it isolates representational routing from every other confound.
"""
