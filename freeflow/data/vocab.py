"""Span vocabulary for Tier B, stratified by the classes H2 asks about.

Four classes, chosen so that "does the gap track abstractness and function-word
status, or merely word length?" is answerable. Gate 0 already showed read-back
tracks *length* — so length is controlled here by construction: `rare_long` is
the long class, and the other three are matched to each other in mean length so
a difference between them cannot be a length effect in disguise.

Held out from `scripts/gate0_sweep.py`'s smaller list on purpose. Gate 0 chose
the render config using those words; reusing them to measure the gap would
report a number partly selected for.
"""
from __future__ import annotations

import random
import statistics

# Function words have no synonyms, which is exactly why the text control is a
# surface variant rather than a paraphrase (see data/views.py).
FUNCTION = [
    "the", "from", "of", "and", "but", "which", "into", "upon", "nor", "yet",
    "than", "whom", "amid", "per", "via", "onto", "while", "since", "though",
    "unless", "until", "within", "beyond", "across", "beneath", "toward",
    "despite", "among", "against", "between", "about", "under", "over",
    "after", "before", "during", "through", "without", "because", "although",
]

CONCRETE = [
    "dog", "table", "mountain", "bicycle", "window", "spoon", "harbour",
    "lantern", "sparrow", "kettle", "bridge", "orchard", "anvil", "quilt",
    "ferry", "walnut", "saddle", "chimney", "trellis", "cactus", "beacon",
    "marble", "pillow", "canyon", "tractor", "hammock", "gutter", "pebble",
    "antler", "cavern", "thimble", "wagon", "meadow", "cellar", "compass",
    "ribbon", "vessel", "furnace", "pasture", "lantern",
]

ABSTRACT = [
    "justice", "theory", "freedom", "irony", "purpose", "doubt", "custom",
    "merit", "hazard", "essence", "notion", "rigour", "candour", "premise",
    "tenet", "whimsy", "virtue", "malice", "leisure", "sorrow", "wisdom",
    "burden", "motive", "regret", "esteem", "menace", "solace", "tedium",
    "valour", "clarity", "dogma", "nuance", "pathos", "scruple", "vigour",
    "warrant", "fervour", "gambit", "impetus", "caprice",
]

# Deliberately long. This class exists to make the length effect visible rather
# than let it hide inside the others.
RARE_LONG = [
    "quixotic", "obfuscate", "perspicacity", "antediluvian", "sesquipedalian",
    "logorrhoea", "brobdingnagian", "eleemosynary", "pulchritude",
    "crepuscular", "obstreperous", "vicissitude", "peripatetic", "recalcitrant",
    "perspicuous", "magnanimous", "circumlocution", "verisimilitude",
    "intransigent", "pusillanimous", "grandiloquent", "surreptitious",
    "incontrovertible", "phantasmagoria", "sanctimonious", "perfunctory",
    "anachronistic", "counterintuitive", "disproportionate", "indefatigable",
]

CLASSES: dict[str, list[str]] = {
    "function": FUNCTION,
    "concrete": CONCRETE,
    "abstract": ABSTRACT,
    "rare_long": RARE_LONG,
}


def length_summary() -> dict[str, float]:
    """Mean character length per class — the length confound, made visible."""
    return {k: statistics.mean(len(w) for w in v) for k, v in CLASSES.items()}


def sample(n: int, rng: random.Random,
           classes: list[str] | None = None) -> list[tuple[str, str]]:
    """Sample `(word, class)` pairs, balanced across classes.

    Sampling with replacement once a class is exhausted, so `n` can exceed the
    vocabulary size — repeated words are fine here because each item still gets
    an independent font pair and context, and the probe labels content rather
    than surface.
    """
    names = classes or sorted(CLASSES)
    out: list[tuple[str, str]] = []
    pools = {c: list(CLASSES[c]) for c in names}
    for c in pools:
        rng.shuffle(pools[c])
    cursors = dict.fromkeys(names, 0)
    for i in range(n):
        c = names[i % len(names)]
        pool = pools[c]
        if cursors[c] >= len(pool):
            rng.shuffle(pool)
            cursors[c] = 0
        out.append((pool[cursors[c]], c))
        cursors[c] += 1
    return out
