"""Span vocabulary for Tier B, loaded from `wordlist.tsv`.

Two objects, because two measurements want different things from a vocabulary
and conflating them is what broke the linear-map null.

`SPANS` --- **volume**. The map-based nulls in `metrics/geometry.py` fit a
`[D, D]` transform between the modalities. Such a map can place each span's
image state directly onto its text state whenever `D >= (distinct spans)`, and
extra *renderings* do not help: thirteen typefaces of one word are thirteen rows
and one constraint. At `D = 2048` the pool must run to several thousand distinct
words before the null means anything. The 150-word list this replaces made it
untestable, which is how the run of 2026-08-31 returned `MSG = 0.006`.

`CLASSES` --- **contrast**. H2 asks whether the gap tracks abstractness and
function-word status rather than word length. That needs classes that are
labelled, length-controlled, and defensible; it does not need to be large.

Class labels come from Brysbaert et al. (2014) concreteness norms rather than
from anyone's intuition, and the thresholds are recorded in
`scripts/build_vocab.py`. Words in the middle of the concreteness scale are
labelled `bulk`: they are real words, usable for volume, and no class claim is
made about them.

**Two limits worth stating rather than hiding.**

*Function words are a closed class.* There are ~137 of them here and there is no
way to get more, so class-balanced sampling is capped by that number. This is
why `sample()` has an unbalanced mode: the map test needs distinct spans, and
insisting on balance would throw away the volume it exists to provide.

*Abstract words are longer than concrete ones* --- a property of English, and
precisely the confound length-matching removes. `CLASSES["concrete"]` and
`CLASSES["abstract"]` are therefore matched length-for-length, not merely in
mean. Function words cannot be resampled to match, so `length_summary()` reports
every class's distribution and the mismatch stays visible.
"""
from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

WORDLIST = Path(__file__).with_name("wordlist.tsv")

# The pool `SPANS` is trimmed to. Comfortably above 2x the 2048-dimensional
# hidden state, which is what the map nulls need to be able to conclude
# anything; see MapFit.underdetermined.
DEFAULT_POOL = 8000


def _load(path: Path = WORDLIST) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate it with "
            "`python scripts/build_vocab.py` (see that script for sources).")
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("word\t"):
            continue
        word, cls, conc, freq, pos = line.split("\t")
        rows.append({"word": word, "cls": cls, "conc": float(conc),
                     "freq": float(freq), "pos": pos})
    return rows


ROWS = _load()


def _length_matched(a: list[dict], b: list[dict],
                    per_length: int = 400) -> tuple[list[str], list[str]]:
    """Trim two classes to a shared length histogram.

    Mean-matching would leave the two classes differently *shaped*: English
    abstract nouns cluster long and concrete nouns short, so two lists can share
    a mean and still differ at every individual length. Since Gate 0 established
    that read-back tracks length, that residual difference would come back as a
    class effect.
    """
    by_len_a: dict[int, list[str]] = defaultdict(list)
    by_len_b: dict[int, list[str]] = defaultdict(list)
    for r in a:
        by_len_a[len(r["word"])].append(r["word"])
    for r in b:
        by_len_b[len(r["word"])].append(r["word"])
    out_a: list[str] = []
    out_b: list[str] = []
    for length in sorted(set(by_len_a) & set(by_len_b)):
        k = min(len(by_len_a[length]), len(by_len_b[length]), per_length)
        out_a.extend(sorted(by_len_a[length])[:k])
        out_b.extend(sorted(by_len_b[length])[:k])
    return out_a, out_b


def _build_classes() -> dict[str, list[str]]:
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in ROWS:
        by_cls[r["cls"]].append(r)
    concrete, abstract = _length_matched(by_cls["concrete"], by_cls["abstract"])
    return {
        "function": sorted(r["word"] for r in by_cls["function"]),
        "concrete": concrete,
        "abstract": abstract,
        "rare_long": sorted(r["word"] for r in by_cls["rare_long"]),
    }


CLASSES: dict[str, list[str]] = _build_classes()


def _strictly_matched() -> dict[str, list[str]]:
    """The three short classes trimmed to one shared length histogram.

    `CLASSES` matches concrete to abstract, which is the pair that carries the
    abstractness contrast. It cannot also match *function*: function words are a
    closed class and a short one (mean 4.9 against 8.3), so bringing the other
    two down to its shape is the only way to make a three-way comparison
    length-free, and it costs most of their size.

    Both are therefore kept. Use these sets for any claim of the form "the gap
    differs by word class", where a residual length difference would be
    indistinguishable from the effect; use `CLASSES` where volume matters more
    than a three-way contrast.
    """
    by_len: dict[str, dict[int, list[str]]] = {}
    for name in ("function", "concrete", "abstract"):
        d: dict[int, list[str]] = defaultdict(list)
        for w in CLASSES[name]:
            d[len(w)].append(w)
        by_len[name] = d
    lengths = set.intersection(*(set(d) for d in by_len.values()))
    out: dict[str, list[str]] = {n: [] for n in by_len}
    for length in sorted(lengths):
        k = min(len(by_len[n][length]) for n in by_len)
        for n in by_len:
            out[n].extend(sorted(by_len[n][length])[:k])
    return out


MATCHED: dict[str, list[str]] = _strictly_matched()


def _build_spans(limit: int = DEFAULT_POOL, seed: int = 0,
                 allowed: set[str] | None = None) -> list[str]:
    """The volume pool: the four classes round-robin, topped up from `bulk`.

    Round-robin rather than a truncation, because the labelled classes total
    ~11.9k and the pool is 8k, so *something* has to be dropped. Taking the
    alphabetically first 8000 would have kept `a`--`m` and thrown away the tail
    of every class, which is both a spelling bias and an uneven cut across
    classes. Rotating gives each class its share and each class an order fixed
    by `seed` rather than by orthography.
    """
    rng = random.Random(seed)
    pools = []
    for name in sorted(CLASSES):
        words = [w for w in CLASSES[name]
                 if allowed is None or w in allowed]
        rng.shuffle(words)
        pools.append(words)
    chosen: list[str] = []
    seen: set[str] = set()
    for i in range(max(len(p) for p in pools)):
        for pool in pools:
            if i < len(pool) and pool[i] not in seen and len(chosen) < limit:
                seen.add(pool[i])
                chosen.append(pool[i])
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        bulk = sorted((r for r in ROWS if r["cls"] == "bulk"
                       and (allowed is None or r["word"] in allowed)),
                      key=lambda r: (-r["freq"], r["word"]))
        for r in bulk:
            if len(chosen) >= limit:
                break
            if r["word"] not in seen:
                seen.add(r["word"])
                chosen.append(r["word"])
    return sorted(chosen)


SPANS: list[str] = _build_spans()

SYNONYMS_FILE = Path(__file__).with_name("synonyms.tsv")


def _load_synonyms(path: Path = SYNONYMS_FILE) -> dict[str, str]:
    """WordNet-derived synonym per word, from `scripts/build_synonyms.py`.

    Absent rather than fatal: only synonym-controlled runs need it, and a
    checkout without it should still reproduce the surface-control numbers.
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("word\t"):
            continue
        word, syn, _rank = line.split("\t")
        out[word] = syn
    return out


SYNONYMS: dict[str, str] = _load_synonyms()

# The pool for synonym-controlled runs. Only ~55% of words have a usable
# synonym, so restricting the 8000-word pool after the fact would leave ~3900
# distinct spans -- under the ~4096 a [2048, 2048] map needs, which would make
# the map nulls abstain on exactly the runs meant to settle the denominator.
# Selecting from covered words up front keeps the pool at full size.
SPANS_SYNONYM: list[str] = (_build_spans(allowed=set(SYNONYMS))
                            if SYNONYMS else [])


def pool_for(control: str) -> list[str]:
    """The span pool a given text control can actually be measured on.

    Takes the control's name rather than the enum: `views` owns `ControlKind`
    and importing it here would couple the vocabulary to the view machinery for
    one comparison.
    """
    if control == "synonym":
        if not SPANS_SYNONYM:
            raise FileNotFoundError(
                f"{SYNONYMS_FILE} is missing; run "
                "`python scripts/build_synonyms.py` to build the synonym "
                "control, or use --control surface")
        return SPANS_SYNONYM
    return SPANS

CLASS_OF: dict[str, str] = {w: c for c, ws in CLASSES.items() for w in ws}


def class_of(word: str) -> str:
    """The word's class, or `bulk` when no class claim is made about it."""
    return CLASS_OF.get(word, "bulk")


def length_summary() -> dict[str, dict[str, float]]:
    """Per-class length, mean and spread — the confound, kept visible."""
    out = {}
    for name, words in CLASSES.items():
        lens = [len(w) for w in words]
        out[name] = {
            "n": len(words),
            "mean": round(statistics.mean(lens), 2) if lens else 0.0,
            "sd": round(statistics.pstdev(lens), 2) if len(lens) > 1 else 0.0,
        }
    for name, words in MATCHED.items():
        lens = [len(w) for w in words]
        out[f"matched_{name}"] = {
            "n": len(words),
            "mean": round(statistics.mean(lens), 2) if lens else 0.0,
            "sd": round(statistics.pstdev(lens), 2) if len(lens) > 1 else 0.0,
        }
    lens = [len(w) for w in SPANS]
    out["_spans"] = {"n": len(SPANS), "mean": round(statistics.mean(lens), 2),
                     "sd": round(statistics.pstdev(lens), 2)}
    return out


def length_histogram(name: str) -> dict[int, int]:
    return dict(sorted(Counter(len(w) for w in CLASSES[name]).items()))


def sample(n: int, rng: random.Random, classes: list[str] | None = None,
           balanced: bool = True,
           pool: list[str] | None = None) -> list[tuple[str, str]]:
    """Sample `(word, class)` pairs.

    `balanced=True` rotates through the four classes, which is what H2 wants and
    what the closed function class caps: asking for more items than four times
    the smallest class necessarily repeats words.

    `balanced=False` draws from `SPANS` without replacement until it is
    exhausted, maximising *distinct* spans: 8000 items give 8000 spans, against
    6137 under balanced sampling, where the 137-word function class repeats to
    fill its quarter. Both now clear what the map nulls need at `D = 2048`; the
    unbalanced mode is simply the one that wastes nothing.
    """
    spans = list(pool if pool is not None else SPANS)
    if not balanced:
        rng.shuffle(spans)
        out = [(w, class_of(w)) for w in spans[:n]]
        while len(out) < n:                    # only once the pool is exhausted
            out.append((w := rng.choice(spans), class_of(w)))
        return out

    names = classes or sorted(CLASSES)
    keep = set(spans)
    out: list[tuple[str, str]] = []
    pools = {c: [w for w in CLASSES[c] if w in keep] or list(CLASSES[c])
             for c in names}
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
