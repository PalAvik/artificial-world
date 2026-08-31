"""Build the text-side within-modality control: a genuine synonym per span.

**Why this decides the headline.** MSG divides the cross-modal distance by a
within-modality control, and the Tier B run of 2026-08-31 showed the two
controls disagree by ~9x: a capitalisation flip moves the merge state 0.0817, a
font change 0.0090. That disagreement is not a detail — it moves the reported
gap by 10x depending on which control you normalise by, and it decides whether
a fitted linear map removes the gap entirely or only most of it.

Both controls are bad, in opposite directions:

* `wisdom -> WISDOM` is **too strong**. It re-tokenises the span completely, so
  it measures the model's sensitivity to tokenisation rather than to surface.
* one font -> another is **too weak**. The glyph sequence is identical and the
  model reads the same word, so it measures almost nothing.

A synonym is the control the metric was defined for: a genuinely different
lexical item carrying the same content. This script derives one per span from
WordNet, which is citable and not anyone's judgement.

**Selection.** Among single-word lemmas sharing a synset with the span:

* the first synset is preferred, being the most common sense;
* candidates close in **length** are preferred, since Gate 0 established that
  read-back tracks length and a systematically longer control would reintroduce
  the confound the corpus is built to avoid;
* candidates close in **frequency** are preferred, because a rare synonym for a
  common word would move the state for reasons of familiarity rather than
  surface;
* morphological variants are rejected — `quick`/`quickly` is an inflection, not
  a second way of saying the same thing.

    python scripts/build_synonyms.py        # writes freeflow/data/synonyms.tsv
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Brysbaert's dominant-POS labels to WordNet POS tags.
POS_MAP = {"Noun": "n", "Verb": "v", "Adjective": "a", "Adverb": "r"}
MIN_LEN, MAX_LEN = 3, 16


def _shares_stem(a: str, b: str) -> bool:
    """True for inflections and derivations of one lemma.

    `wisdom`/`wiseness` are distinct lexical items; `quick`/`quickly` is one
    item in two shapes and tells us nothing about re-expression.
    """
    if a.startswith(b) or b.startswith(a):
        return True
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= max(4, int(0.7 * min(len(a), len(b))))


def choose(word: str, rows: dict, blocked: set, wn) -> tuple[str, int] | None:
    """Best single-word synonym for `word`, or None."""
    info = rows.get(word, {})
    want = POS_MAP.get(info.get("pos", ""))
    synsets = wn.synsets(word, pos=want) if want else wn.synsets(word)
    if not synsets:
        synsets = wn.synsets(word)

    best = None
    for rank, syn in enumerate(synsets):
        for lemma in syn.lemmas():
            cand = lemma.name().lower()
            if "_" in cand or not cand.isalpha():
                continue
            if cand == word or not (MIN_LEN <= len(cand) <= MAX_LEN):
                continue
            if cand in blocked or _shares_stem(word, cand):
                continue
            len_gap = abs(len(cand) - len(word))
            f_src = info.get("freq", 0.0)
            f_dst = rows.get(cand, {}).get("freq", 0.0)
            freq_gap = abs(math.log1p(f_src) - math.log1p(f_dst))
            # Sense rank dominates: a synonym of a rare sense is not a synonym
            # of the word as a reader would meet it in a neutral context.
            score = (rank * 3.0) + (len_gap * 0.5) + freq_gap
            if best is None or score < best[0]:
                best = (score, cand, rank)
    return (best[1], best[2]) if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="freeflow/data/synonyms.tsv")
    args = ap.parse_args()

    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test")
    except Exception as exc:                            # noqa: BLE001
        raise SystemExit(
            f"WordNet unavailable ({exc}). Install nltk and place the corpus at "
            "~/nltk_data/corpora/wordnet — see docs/ENVIRONMENT.md.")

    from freeflow.data import vocab
    rows = {r["word"]: r for r in vocab.ROWS}
    blocked = {w.strip().lower()
               for w in (ROOT / "data" / "sources" / "badwords.txt").read_text().split("\n")
               if w.strip()}

    # Built over the whole word list, not just the 8000-word pool: a
    # synonym-controlled corpus can only use covered words, and at ~50%
    # coverage drawing from 8000 would leave ~3900 distinct spans — just under
    # the ~4096 a [2048, 2048] map needs. Covering the full list lets
    # `vocab.spans_with_synonym()` assemble a pool that clears it.
    candidates = [r["word"] for r in vocab.ROWS]
    found, missing = [], 0
    for word in candidates:
        got = choose(word, rows, blocked, wn)
        if got:
            found.append((word, got[0], got[1]))
        else:
            missing += 1

    out = ROOT / args.out
    with out.open("w") as f:
        f.write("# generated by scripts/build_synonyms.py from WordNet — "
                "do not edit by hand\n")
        f.write("word\tsynonym\tsense_rank\n")
        for word, syn, rank in sorted(found):
            f.write(f"{word}\t{syn}\t{rank}\n")

    import statistics
    dl = [len(s) - len(w) for w, s, _ in found]
    print(f"wrote {out}")
    covered_pool = sum(1 for w in vocab.SPANS if w in {x for x, *_ in found})
    print(f"    {len(found)} of {len(candidates)} words have a synonym "
          f"({len(found) / len(candidates):.0%}); {missing} without")
    print(f"    of the {len(vocab.SPANS)}-word pool: {covered_pool} covered")
    print(f"    length delta: mean {statistics.mean(dl):+.2f}, "
          f"sd {statistics.pstdev(dl):.2f}")
    print(f"    from the first synset: "
          f"{sum(1 for *_, r in found if r == 0) / max(1, len(found)):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
