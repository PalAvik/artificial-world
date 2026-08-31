"""Regenerate `freeflow/data/wordlist.tsv` from published lexical norms.

Why a generator rather than a literal list in the source: the Tier B vocabulary
needs to reach several thousand distinct spans (see below), and eight thousand
hand-written words would be both unverifiable and, in the word-class columns,
simply my own judgement wearing the costume of data.

**Why the size matters.** The linear-map null in `freeflow/metrics/geometry.py`
fits a `[D, D]` map between the two modalities' merge-position states. Such a
map can place each span's image state exactly onto its text state whenever
`D >= (number of distinct spans)`, and no quantity of extra *renderings* of the
same words changes that — thirteen typefaces of one word are thirteen rows and
one constraint. With `D = 2048` the corpus needs several thousand distinct
words before the null is testable at all. The 150-word list this replaces made
it untestable, which is how a run on 2026-08-31 produced a meaningless
`MSG = 0.006` (see results/DECISIONS.md).

**Sources**, all downloaded to `data/sources/` and read from there:

* `bry.txt` --- Brysbaert, Warriner & Kuperman (2014), *Concreteness ratings
  for 40 thousand generally known English word lemmas*, Behavior Research
  Methods. Supplies `Conc.M` (1--5), `Percent_known`, the SUBTLEX frequency and
  a dominant part of speech. This is what makes the concrete/abstract split an
  empirical measurement rather than my intuition, and it is citable.
* `google-10000-english-usa.txt` --- frequency-ordered common English, used only
  to prefer words the model has plausibly seen.
* `badwords.txt` --- the LDNOOBW English list, used to exclude slurs and
  profanity from a corpus that gets rendered into images.

**Held out.** Words used by `scripts/gate0_sweep.py` are excluded: Gate 0 chose
the render configuration using them, so measuring the gap on the same words
would report a number partly selected for.

    python scripts/build_vocab.py            # writes freeflow/data/wordlist.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Concreteness cut-offs. Deliberately far apart: the middle of the scale is
# where the rating is least reliable and where "concrete" and "abstract" stop
# being distinct claims.
CONCRETE_MIN = 4.0
ABSTRACT_MAX = 2.0
KNOWN_MIN = 0.85          # share of raters who knew the word
MIN_LEN, MAX_LEN = 3, 16  # lower bound avoids stopword-shaped noise; upper
                          # bound keeps a rendered span inside the frozen
                          # Gate 0 strip geometry
RARE_MIN_LEN = 10         # the long class exists to make length visible
RARE_MAX_FREQ = 1.0       # SUBTLEX occurrences per million

# Closed class, so it is enumerated rather than sampled. Function words are the
# one class that cannot be grown: there are only so many of them in English,
# which caps class-balanced sampling and is why `sample()` also has an
# unbalanced mode.
FUNCTION_WORDS = """
the a an this that these those some any each every either neither both all
and or but nor yet so for because although though unless until while whereas
since whether if then than as when where how why what which who whom whose
of in on at by to from with without within into onto upon over under above
below across through during before after between among against toward towards
beyond beneath behind beside besides despite about around along amid amongst
per via off out up down near inside outside throughout underneath alongside
he she it they we you him her them us its his their our your mine yours theirs
is are was were be been being am do does did done have has had having
can could shall should will would may might must ought need dare
not no nor never always often seldom rarely sometimes usually
very quite rather almost nearly just only even also too still yet already
there here now then once again more most less least much many few several
""".split()


def _load_norms(src: Path) -> list[dict]:
    rows = []
    with (src / "bry.txt").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("Bigram") != "0":
                continue                       # single words only
            word = (r.get("Word") or "").strip()
            try:
                rows.append({
                    "word": word,
                    "conc": float(r["Conc.M"]),
                    "known": float(r["Percent_known"]),
                    "freq": float(r["SUBTLEX"]),
                    "pos": (r.get("Dom_Pos") or "").strip() or "NA",
                })
            except (KeyError, ValueError):
                continue                       # malformed row
    return rows


def _blocklist(src: Path) -> set[str]:
    path = src / "badwords.txt"
    if not path.exists():
        raise SystemExit(f"missing {path}; refusing to build a rendered corpus "
                         "without a profanity filter")
    return {w.strip().lower() for w in path.read_text().split("\n") if w.strip()}


def _gate0_words() -> set[str]:
    """Words Gate 0 selected the render config on. Held out by construction."""
    import re
    text = (ROOT / "scripts" / "gate0_sweep.py").read_text()
    block = re.search(r"WORD_CLASSES.*?\n\}", text, re.S)
    return set(re.findall(r'"([a-z]+)"', block.group(0))) if block else set()


def classify(rows: list[dict], blocked: set[str], held_out: set[str]) -> list[dict]:
    """Assign each usable word to one of the four classes, or to `bulk`."""
    out = []
    function = set(FUNCTION_WORDS)
    seen: set[str] = set()
    for r in rows:
        w = r["word"]
        if w in seen or w in blocked or w in held_out:
            continue
        if not (w.isalpha() and w.islower() and MIN_LEN <= len(w) <= MAX_LEN):
            continue
        if r["known"] < KNOWN_MIN:
            continue
        seen.add(w)
        if w in function:
            cls = "function"
        elif len(w) >= RARE_MIN_LEN and r["freq"] <= RARE_MAX_FREQ:
            cls = "rare_long"
        elif r["conc"] >= CONCRETE_MIN:
            cls = "concrete"
        elif r["conc"] <= ABSTRACT_MAX:
            cls = "abstract"
        else:
            cls = "bulk"        # mid-scale: real word, no class claim made
        out.append({**r, "cls": cls})

    # Function words are largely absent from the concreteness norms, so add any
    # that the norms missed. They carry no rating, and none is claimed.
    for w in sorted(function - seen):
        if MIN_LEN <= len(w) <= MAX_LEN and w not in blocked and w not in held_out:
            out.append({"word": w, "conc": 0.0, "known": 1.0, "freq": 0.0,
                        "pos": "Function", "cls": "function"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="data/sources")
    ap.add_argument("--out", default="freeflow/data/wordlist.tsv")
    args = ap.parse_args()

    src = ROOT / args.sources
    rows = _load_norms(src)
    blocked = _blocklist(src)
    held_out = _gate0_words()
    words = classify(rows, blocked, held_out)

    counts: dict[str, int] = {}
    for r in words:
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1

    out = ROOT / args.out
    with out.open("w") as f:
        f.write("# generated by scripts/build_vocab.py — do not edit by hand\n")
        f.write("word\tclass\tconcreteness\tfrequency\tpos\n")
        for r in sorted(words, key=lambda r: (r["cls"], r["word"])):
            f.write(f"{r['word']}\t{r['cls']}\t{r['conc']:.2f}\t"
                    f"{r['freq']:.1f}\t{r['pos']}\n")

    print(f"wrote {out}  ({len(words)} words)")
    for k in sorted(counts):
        print(f"    {k:10} {counts[k]:6}")
    print(f"    held out from Gate 0: {len(held_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
