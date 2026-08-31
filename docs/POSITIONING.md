# What this measures that nobody else does

Written 2026-08-31, after the Tier B result came back null (the gap is a change
of basis). The question it answers: *given that, is there a real research gap
here, or should the whole thing be shelved?*

Sources are listed at the bottom. This is a first sweep, not the full
related-work section — treat the "nobody does this" claims as provisional until
that is finished.

---

## First, what is NOT new

**"A linear map connects image and text representations" is known.** Merullo et
al., *Linearly Mapping from Image to Text Space* (ICLR 2023), train a single
linear projection from a frozen image encoder into a frozen LM's input space and
get competitive captioning and VQA. Our linear-map-free MSG of 0.86 is
*consistent with* that result, not a discovery beyond it.

So the headline cannot be "we found the gap is linear". Anyone who knows LiMBeR
will say it was already known. The contribution has to be elsewhere.

---

## Gap 1 — the standard picture of the modality gap does not describe a decoder VLM

Liang et al., *Mind the Gap* (2022), established the dominant account: the
modality gap is a **cone effect**, two modalities occupying separate cones, and
it is quantified by the **distance between centroids** and remedied by shifting
one mean onto the other.

That account was developed on CLIP-style **pooled contrastive embeddings**. We
measure inside a decoder VLM's residual stream, at a single token position, and
it does not hold:

| account | what it predicts | what we measure |
|---|---|---|
| centroid offset (cone effect) | subtracting per-modality means closes the gap | closes **6%** |
| rotation | — | closes **89%** |
| any linear map | — | closes **all** of it |

The gap is overwhelmingly **orientation**, not displacement. A mean-shift — the
standard remedy in that literature — barely touches it. This is a correction to
a widely-cited characterisation, in a setting (decoder VLMs) where it is
routinely assumed to carry over.

## Gap 2 — nobody normalises the distance

Modality-gap work reports raw distances, cosine similarities, or centroid
separations. None of those have a scale. "The gap is 0.15" means nothing without
knowing how far apart two expressions of the *same* content sit.

Dividing by a within-modality control on the same items gives a number with a
meaningful value of 1. It also surfaced something no raw-distance paper could
see: **our two controls differ by 9.1x** — recasing a word moves the merge state
0.0817, re-rendering it in another typeface moves it 0.0090. Any unnormalised
distance claim about this space is unanchored, and the choice of control is
doing most of the work.

## Gap 3 — a content-identical cross-modal probe

Comparing "dog" with a photograph of a dog always confounds representational
divergence with a genuine content difference: the word denotes a category, the
photo denotes one animal. Every grounding-based probe has this problem, and it
means a nonzero gap is expected under *every* hypothesis.

Replacing a word with **a picture of that word** removes the confound entirely —
same lexical item, different encoder — and extends to abstract nouns and
function words, which have no visual referent and which grounding-based probes
therefore cannot test at all.

---

## Where this actually matters: two live problems

### A. Typographic jailbreaks — why does rendering text defeat safety training?

FigStep (AAAI 2025) renders forbidden instructions as images and reports ~82.5%
attack success across six open VLMs. The standard explanation is "imperfect
transfer of text-centred alignment to multimodal representation space", which
describes the symptom rather than the mechanism.

Our result sharpens it into a testable claim. At 2B, the rendered text is *not*
poorly represented — the content is there, in rotated coordinates, and the model
reads it near-perfectly. So the failure is unlikely to be that safety training
cannot see the content. A better hypothesis: **safety behaviour is learned as a
direction in the text frame, and a rotated copy of the same content misses it.**

That makes MSG — specifically the *residual after the best rotation* — a
candidate predictor of typographic-jailbreak susceptibility. If the rotation
residual across models correlates with attack success rate, that is a mechanistic
account of a known and unexplained vulnerability. This is the highest-value
direction the project has.

### B. Visual-text context compression — is rendering text lossless?

Rendering long text as images to save decoder tokens is now an active technique:
Glyph, DeepSeek-OCR, and VIST all do it, and *Text or Pixels?* reports roughly
half the tokens for comparable performance. Whether the compression is
representationally lossless is currently argued from downstream benchmark scores.

Our instrument measures it directly, per span, with controls — and the
font-invariance number (0.009, an order of magnitude below the text control) says
the visual pathway is remarkably stable to typography, which is exactly what
those systems need in order to choose a renderer.

---

## Honest assessment

The original framing — "align the modalities by training" — is dead, and the
measurement says so. The reframing is:

> **a normalised, controlled instrument for cross-modal token substitutability,
> the finding that the gap inside a decoder VLM is orientation rather than
> displacement or information, and the consequences of that for typographic
> safety transfer and visual-text compression.**

That is a smaller paper than originally imagined and a more defensible one. The
riskiest remaining claim is Gap 1's novelty: it needs the full related-work sweep
to confirm nobody has run this decomposition inside a decoder VLM.

## Sources

- Merullo et al., *Linearly Mapping from Image to Text Space*, ICLR 2023 — https://arxiv.org/abs/2209.15162
- Liang et al., *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*, 2022 — https://arxiv.org/abs/2203.02053
- Gong et al., *FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts*, AAAI 2025 — https://ojs.aaai.org/index.php/AAAI/article/view/34568
- *Glyph: Scaling Context Windows via Visual-Text Compression*, 2025 — https://arxiv.org/pdf/2510.17800
- *Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs*, 2025 — https://arxiv.org/pdf/2510.18279
- *Vision-centric Token Compression in Large Language Model*, NeurIPS 2025 — https://arxiv.org/pdf/2502.00791
