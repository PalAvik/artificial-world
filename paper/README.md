# Paper draft

ICLR 2026 format. **Abstract, introduction and method only** — written while the
deciding experiment was still running, and deliberately framed so that it does
not presuppose the outcome.

## Build

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires `texlive-latex-base`, `texlive-latex-recommended`,
`texlive-fonts-recommended`. Verified to compile to 5 pages. `bibtex` exits
non-zero because there are no citations yet — see below — which is expected and
does not affect the output.

## Layout

| file | contents |
|---|---|
| `main.tex` | preamble and includes |
| `sections/abstract.tex` | |
| `sections/introduction.tex` | the problem, why it is ill-posed, contributions |
| `sections/method.tex` | views, merge position, MSG, tiers, validity, protocol |
| `figures/method.tex` | Figure 1, TikZ; the merge position and the MSG ratio |
| `figures/span_*.png` | the example stimulus, rendered by `scripts/make_figure_assets.py` at the frozen Gate 0 geometry |
| `references.bib` | **empty on purpose** — see below |
| `iclr2026_conference.{sty,bst}` | official template, fetched from the ICLR master template |
| `template_reference.tex.orig` | the template's own example file, for formatting reference |

## Figure 1

Regenerate the stimulus strips with:

```bash
python scripts/make_figure_assets.py --word wisdom
```

They are the *actual* Tier B stimulus at the frozen render geometry, not a
drawing of one, so the figure cannot drift from the corpus and a reader can see
for themselves how legible a six-visual-token span is. The figure needs
`texlive-pictures` for TikZ; it fits `\textwidth` with no overfull box.

## What is deliberately missing

**Citations.** `references.bib` is empty and no `\cite` appears in the text. The
related-work sweep has not been done, and inventing plausible-looking references
is worse than having none: a fabricated citation is undetectable to a co-author
skimming a draft and embarrassing to a reviewer who checks it. The bib file
lists the specific claims that currently stand unsupported and need a real
reference before this is circulated.

**Results and conclusions.** No numbers appear. The deciding measurement — does
an out-of-fold linear map remove the modality gap? — was still running when this
was drafted, and the answer changes what the paper is:

- *gap survives a fitted linear map* → the representational difference is not a
  change of basis, and the measurement supports a training program.
- *gap dies under a fitted linear map* → the difference is one a standard
  vision-to-language projector already closes. The paper becomes a short
  negative note, and the method sections here are still most of it.

The method section is written to be true under both, because it describes an
instrument rather than a result. What would have to change under the second
outcome is the framing of the contributions list and the abstract's final
sentence, not the method.

**Related work.** Not drafted. It is the section most likely to change the
introduction's claims, particularly the assertion that existing modality-gap
work reports unnormalised distances.
