# FreeFlow: Cross-Modal Token Substitutability in Small VLMs

**Status:** research plan, v0.1 · **Base model target:** ~2B VLM · **Horizon:** ~10 weeks to a submittable result

---

## 0. TL;DR

The goal is a model in which *meaning is carried by content, not by modality*: a span of a
sequence can be expressed as text tokens or as image tokens, and the model behaves the same
either way — same next-token distribution, same downstream answer, and enough information
retained that the span can be regenerated in either modality.

This plan does three things to the idea as originally stated:

1. **Replaces "equivalence" with a measurable, task-conditional quantity.** Literal token
   substitutability is ill-posed (§1). *Substitution invariance relative to a task and
   relative to within-modality variation* is well-posed, measurable today, and is the paper.
2. **Solves the "abstract words" problem constructively.** You do not need a picture *of*
   "the". You need a picture of the *word* "the" (§2, Tier B). Glyph rendering makes every
   token in the vocabulary substitutable, gives an information-preserving bridge in one
   direction, and turns the hardest part of the idea into the cleanest controlled experiment
   in the project.
3. **Puts measurement before training.** Phase 1 is a pure evaluation study on an
   off-the-shelf VLM. It is independently publishable, it is cheap, and if it comes back
   with the wrong shape it saves nine weeks (§7).

The headline figure the project is aiming at: **the invariance–informativeness frontier**
(§5.4). Cross-modal invariance is trivially achievable by discarding information. The
contribution is showing where current VLMs sit on that trade-off, and moving the frontier.

---

## 1. Sharpening the goal: what "equivalence" can and cannot mean

The original framing — "replace a text token with the corresponding image tokens and the
model understands it the same" — has three problems that must be designed around, not
ignored.

### 1.1 There is no bijection

One word maps to *N* image patches, *N* varying with resolution and content. There is no
token-for-token swap. **Design consequence:** substitution operates on *spans*, not tokens,
and equivalence is asserted at the **merge position** — the first position *after* the
swapped span, where the model has had to integrate the span into its running state. The span
positions themselves *should* differ; forcing them to match is a modeling error, and it is a
mistake we will explicitly ablate (§6.3).

### 1.2 There is no canonical rendering

"Dog" has many images; "from" has none in the referential sense. §2 handles this by splitting
substitution into three tiers with different expectations, rather than pretending one relation
covers all tokens.

### 1.3 Reconstruction is asymmetric, and one direction is information-theoretically impossible

`text → image → text` can be near-lossless: render glyphs, read them back. `image → text →
image` cannot be. A 448×448 image carries far more bits than any caption of it. Demanding
"accurate reconstruction" from an image span routed through text is asking for a violation of
the source coding theorem.

**Design consequence:** reconstruction is never reported as a single number. It is reported as
a **rate–distortion curve** — bits in the intermediate representation on the x-axis,
reconstruction quality on the y-axis — and distortion is measured *task-conditionally*
(does the reconstruction answer the same questions?) rather than pixel-wise. A model that
reconstructs a semantically identical but pixel-different dog has succeeded at the thing we
actually care about.

### 1.4 The reframed goal

> **Task-conditional substitution invariance.** For a task distribution *T* and a substitution
> operator *σ* that re-expresses a span in another modality, the model's behaviour on
> *σ(x)* should differ from its behaviour on *x* by no more than the model's own variation
> across two same-modality paraphrases of that span.

That last clause is the important one and is developed in §5.3. It converts an unbounded
"how close is close enough" question into a ratio with a natural target of 1.0.

---

## 2. The substitution taxonomy

Three tiers, three different relations, three different success criteria. Conflating them is
the main way this project could produce uninterpretable numbers.

### Tier A — Referential (concrete)
A photograph or crop depicting the referent replaces the phrase. `"a dog"` ↔ image region of
a dog.
- **Bridge:** grounding. Natively annotated in Flickr30k Entities and Visual Genome region
  descriptions — phrase↔box pairs, no synthesis needed.
- **Expectation:** invariance should be *achievable but lossy*; the image specifies a
  particular dog, the phrase specifies a class. Report invariance conditioned on questions
  answerable from the phrase alone.
- **This is the tier where existing VLMs are already partly aligned**, so it is the weakest
  test and the easiest win. It is the control, not the contribution.

### Tier B — Orthographic (renderable — the abstract-word solution)
The span is rendered as glyphs and inserted as pixels. Works for *every* token in the
vocabulary, including `"the"`, `"from"`, `"("`, and subword fragments.
- **Bridge:** typography. Synthesized on demand, unlimited quantity, exactly controlled
  (font, size, background, kerning, noise — all become ablation axes).
- **Expectation:** this direction is *information-preserving in principle*. The pixels contain
  the string exactly. So any behavioural gap is attributable purely to the model's internal
  routing — the OCR pathway landing somewhere other than where the embedding table lands.
- **Why this tier is the centre of the project:** it isolates the phenomenon from every
  confound. There is no semantic ambiguity, no annotation noise, no grounding error. If
  `render("the")` and `token("the")` produce different model states, that is a pure statement
  about representational geometry. Strong prior that current VLMs fail here badly, and that
  is a result worth reporting on its own.
- **Sub-study:** does the gap scale with word frequency, concreteness, or length? A gap that
  vanishes for concrete nouns but persists for function words would be a genuinely
  interesting finding about what multimodal pretraining does and does not fuse.

### Tier C — Relational / structural
A diagram, arrow, or spatial arrangement encodes a *relation* rather than an entity:
`"on top of"`, `"before"`, `"causes"`.
- **Bridge:** synthetic scene generation with programmatic ground truth (CLEVR-style), plus
  GQA's relation annotations.
- **Expectation:** hardest tier, likely large gaps. Entities may transfer across modalities
  while relations do not — that asymmetry, if it holds, is the strongest world-model claim
  in the paper, because it says multimodal pretraining fuses *nouns* and not *structure*.

---

## 3. Falsifiable hypotheses

- **H1 (gap exists).** In an off-the-shelf 2B VLM, normalized MSG (§5.3) ≫ 1 across all
  three tiers — the model treats a cross-modal restatement as far more different than a
  within-modality paraphrase.
- **H2 (gap is structured, not uniform).** MSG is ordered Tier A < Tier B < Tier C, and
  within Tier B correlates with abstractness/function-word status rather than string length.
- **H3 (gap is largely a removable offset).** A substantial fraction of the geometric gap is
  a constant per-modality mean offset that the readout could ignore; the *residual* gap after
  removing it is much smaller than the raw gap (§5.2). This matters enormously: if H3 holds,
  the problem is far more tractable than the modality-gap literature suggests.
- **H4 (trainable).** The ISO objective (§6) reduces normalized MSG toward 1.0 without
  degrading standard VLM benchmarks by more than a small, stated margin.
- **H5 (the real claim).** Training for substitution invariance *improves* compositional and
  relational generalization — i.e. it is not merely a representational tidy-up but buys
  world-model quality. Tested on held-out relational composition, not on the training tasks.

### Kill criteria

State these now, before any results exist.

- If **H1 fails** (gap already ≈ 1 everywhere): the phenomenon does not exist at this scale.
  Pivot to the generation half, or to larger models where the gap may be scale-dependent.
- If **H3 fails badly** *and* H4 shows invariance rising only alongside falling probe
  decodability (§5.4): the objective is buying invariance by destroying information. Stop
  and redesign around the generation loss as the primary anchor.
- If **H4 succeeds but H5 fails** (invariance achieved, no downstream benefit): this is a
  negative result, still publishable as a measurement paper, but do not oversell it as
  world-model progress. Write it as the measurement paper it is.

---

## 4. What already exists (positioning)

Adjacent work to position against, not to duplicate:

- **Modality-gap literature** — established that VLM embeddings occupy separate cones and
  that extracted concepts fire for one modality only. Mostly at the *whole-embedding*,
  CLIP-style level. **Our delta:** token/span-level, inside an autoregressive LM's residual
  stream, and *interventional* (we substitute and measure behaviour) rather than observational.
- **Pixel/glyph language models and vision-text compression** — render text as images to
  extend context or bypass tokenizers. **Our delta:** they use rendering as a *compression
  mechanism*; we use it as a *measurement instrument* for representational fusion, and we
  care about the equality of the two pathways rather than the efficiency of one.
- **Unified understanding+generation models** (shared discrete visual tokenizer + diffusion
  decoder architectures). **Our delta:** they unify at the *architecture* level and evaluate
  on standard generation/understanding benchmarks. Nobody reports whether the unified
  representation is actually *substitutable*. Our metric suite is the missing evaluation for
  that entire model class — which is a good position to occupy.
- **Cross-modal token alignment / attention-anchor methods** — align tokens to improve
  benchmark scores. **Our delta:** invariance as the object of study with an explicit
  informativeness control, not as a means to a leaderboard number.

*Action item for week 1: a proper related-work sweep. The above is a positioning sketch from
a shallow search, and the "nobody reports" claim needs verification before it goes in a paper.*

---

## 5. The metric suite (build this first)

All metrics live in `freeflow/metrics/`. Nothing else is built until these are.

Notation: context `C` containing span `s`. `V_T` = `C` with `s` as text tokens. `V_I` = `C`
with `s` as image tokens. Shared continuation `y` for teacher forcing.

### 5.1 Distributional
`JSD( p(y | V_T) ‖ p(y | V_I) )`, teacher-forced token-by-token over the shared continuation.
Jensen–Shannon rather than KL: symmetric (neither view is privileged) and bounded (comparable
across items). Report mean and the full distribution — the tail matters more than the mean.

### 5.2 Geometric
At the merge position (first position after `s`), for each layer `l`, compare `h_T^(l)` and
`h_I^(l)`:
- raw cosine distance,
- **offset-free distance**: same, after subtracting the per-modality batch mean. A constant
  per-modality translation is harmless if downstream readouts are affine — reporting only the
  raw distance conflates a benign offset with genuine representational divergence, which is a
  recurring flaw in modality-gap reporting,
- linear CKA over a batch,
- the offset norm `‖E[h_T] − E[h_I]‖` itself, reported separately as the thing we are
  factoring out.

### 5.3 Normalized MSG — the headline number
A raw distance is uninterpretable: is 0.3 large? Normalize by the model's own within-modality
variation.

```
                    d( h(V_T) , h(V_I) )
normalized MSG  =  ──────────────────────────────────────────
                   ½[ d(h(V_T), h(V_T')) + d(h(V_I), h(V_I')) ]
```

where `V_T'` is a *text paraphrase* of the span and `V_I'` is a *different image of the same
content* (different photo for Tier A; different font/size/background for Tier B).

- **MSG ≈ 1** → crossing modalities costs no more than rephrasing. This is the target, and it
  is the operational definition of "free-flowing".
- **MSG ≫ 1** → modality dominates content.

The denominator is what makes the whole suite credible, and building the paraphrase/re-render
machinery is therefore a Phase-0 dependency, not an afterthought.

### 5.4 Informativeness probe — the anti-collapse control
Invariance is trivially maximized by mapping everything to a constant. Guard against it:
train a linear probe on `h` at the merge position to recover the identity of the substituted
span. Track **probe accuracy** alongside **MSG** at every checkpoint.

Plot the two against each other across training runs and hyperparameters. That plot — the
**invariance–informativeness frontier** — is the paper's central figure. A method that moves
down-and-left along the existing frontier has achieved nothing; a method that moves the
frontier outward has.

### 5.5 Cycle consistency
- `T → I → T`: render span, feed, ask the model to recover the string. Exact match + character
  error rate. Near-lossless is the expectation; failure here is a pure routing failure.
- `I → T → I`: encode span to text, regenerate an image from that text, compare to the
  original. **Not pixel metrics.** Use (a) embedding similarity in a *third-party* encoder
  held out from training, and (b) a **VQA agreement score**: pose a fixed question set to
  original and reconstruction, measure answer agreement. Task-conditional distortion, per §1.3.
- Both reported as rate–distortion curves by sweeping the intermediate budget (caption
  length / number of retained tokens).

### 5.6 Capability retention
Standard VLM benchmarks throughout, to catch invariance being bought with general competence.
Any invariance gain reported alongside its retention cost, always.

---

## 6. Model and training design

### 6.1 Base model — one correction to the starting point

`Qwen/Qwen3.5-2B` (released March 2026) is a **text-only** LLM. It has no vision encoder, so
it cannot host this experiment as-is. Verified against its model card; the Qwen3.5 small line
is 0.8B / 2B / 4B / 9B, text.

Recommendation:

- **Workhorse: `Qwen3-VL-2B`** — confirmed to exist, right size, native interleaved
  text/image context, strong OCR (which matters a lot for Tier B). Understanding-only, which
  is fine for Phases 0–2.
- **Check first:** whether the Qwen3.5-VL line ships a ≤4B variant. Sources conflict on
  whether Qwen3.5's vision capability is a separate VL series or folded into the main line,
  and `huggingface.co` is blocked from this container so I could not resolve it directly.
  If a small Qwen3.5-VL exists, prefer it and keep Qwen3-VL-2B as the generational control.
- **Keep `Qwen3.5-2B` as the text-only ablation** — genuinely useful for isolating what the
  vision tower contributes versus what the LM already does with rendered text descriptions.
- **Phase 3 (generation)** needs either a discrete visual tokenizer + decoder bolted onto the
  VLM, or a unified any-to-any base. Decide at the Phase 2/3 boundary with Phase 1 data in
  hand; do not pre-commit now.

### 6.2 The ISO objective (Invariant Substitution Objective)

```
L =        L_LM        (standard next-token loss on modality-mixed sequences)
  + λ_d ·  L_dist      (symmetric JSD between the two views' next-token distributions)
  + λ_r ·  L_repr      (offset-free distance between merge-position hidden states)
  + λ_c ·  L_cycle     (regenerate the swapped span in the *opposite* modality)
  + λ_g ·  L_gen       (image-token reconstruction — the anti-collapse anchor)
```

Design points that matter:

- **`L_repr` applies only at merge positions and after**, never on the span tokens themselves.
  The span representations *should* differ — they have different cardinality and different
  surface form. Ablated in §6.3.
- **`L_dist` uses alternating stop-gradient**, each view teaching the other, rather than a
  fixed teacher. Neither modality should be privileged as the canonical one; a fixed
  text-teacher would bake in exactly the text-centrism we are trying to remove.
- **`L_gen` is not optional.** It is the structural reason the model cannot satisfy the
  invariance terms by discarding content. If Phase 3 slips, substitute a reconstruction proxy
  (predict the image encoder's features for the swapped span) so the anchor exists from
  Phase 2 onward.
- **Curriculum:** Tier B first (clean, unlimited, information-preserving), then A, then C.
- **LoRA before full fine-tuning.** Establish that the objective does anything at all cheaply.

### 6.3 Ablations that must run

| Ablation | Question it answers |
|---|---|
| `L_repr` on span positions too | Is forcing span-level matching harmful, as §1.1 predicts? |
| Drop `L_gen` | Does invariance collapse the representation without the anchor? |
| Fixed text-teacher vs alternating | Does a privileged modality bake in text-centrism? |
| Tier B only vs all tiers | Does orthographic invariance transfer to referential? |
| Render-augmentation only (no ISO terms) | Is plain data augmentation the whole effect? |

That last row is the reviewer's first question and the most likely way this result gets
deflated. Run it early, not at the end.

---

## 7. Phases (1 × A100 80GB, 10 weeks)

Compute budget and per-run configs: `docs/COMPUTE.md`. Gate thresholds and drop rules:
`docs/GATES.md`. Setup commands: `docs/ENVIRONMENT.md`.

**Scope decision forced by the hardware:** pixel-level image generation is out of scope
for v1. On one A100 in ten weeks a generative decoder trained to publication quality is
not achievable, and chasing it is the most likely way to finish with nothing. The
anti-collapse anchor becomes **feature-space reconstruction** — predicting the frozen
vision encoder's features for the swapped span — which gives the same structural
guarantee against collapse and the same "information retained" measurement at a fraction
of the cost. The `T→I→T` cycle still runs in full (read-back needs no generation);
`I→T→I` runs in feature space. Pixel reconstruction is future work.

**Phase 0 — Instrument (week 1).** Substitution corpus builder: glyph renderer with
controlled font/size/background/noise; span aligner for Flickr30k Entities and Visual
Genome; paraphrase and re-render generators for the MSG denominator. Fix the train/held-out
splits now and never revisit them. CPU-only apart from the Gate 0 check.
*Exit:* corpus builder produces all three tiers with paired views, unit-tested, and
**Gate 0** passed (~1 GPU-hour).

**Phase 1 — Measure (weeks 2–3).** Full metric suite against off-the-shelf Qwen3-VL-2B.
Inference only, ~30 GPU-hours including dev iteration. Tests H1–H3.
*Exit:* the modality-substitution-gap study, per-tier and per-word-class. **Gate 1 —
the first real go/no-go, with an explicit DROP branch.**

**Phase 2 — Train for invariance (weeks 4–7).** ISO objective, LoRA for all six configs,
one full fine-tune to confirm the winner isn't a LoRA artifact. ~90 GPU-hours.
Frontier plot logged at every eval step, not reconstructed afterwards.
**Run the render-augmentation-only ablation in week 5, not week 7** — it is the most
likely way this project produces a result that looks good and means nothing.
*Exit:* H4 answered. **Gate 2**, whose condition (d) can drop the method claim outright.

**Phase 3 — Downstream and cycles (weeks 8–10).** H5 on held-out relational composition,
3 seeds. `T→I→T` read-back cycle in full; `I→T→I` in feature space, as rate–distortion
curves. Paper drafting runs in parallel from week 8, not after. ~60 GPU-hours.
*Exit:* **Gate 3** — pass gives the full world-model claim, fail downgrades to a
measurement-plus-method paper with an explicitly negative H5.

**Scale-up is not in the 10 weeks.** One A100 cannot deliver an 8B scale point on top of
the above. If reviewers demand one, it is a post-submission run on borrowed hardware —
plan the paper so the claim does not depend on it.

---

## 8. Risks

- **Collapse.** The dominant failure mode. Mitigated by `L_gen` + the probe, but it is the
  thing to watch every single checkpoint, not at the end.
- **The augmentation deflation.** "You just trained on rendered text." Pre-empted by the
  §6.3 ablation and by H5's held-out compositional test.
- **Tier A annotation noise** swamping the signal — grounding boxes are imprecise. Mitigate by
  leading with Tier B, where ground truth is exact by construction.
- **OCR ceiling.** If the base model cannot reliably read rendered spans at all, Tier B
  measures OCR failure rather than representational geometry. **Check this in week 1** with a
  simple read-back accuracy test; it gates the entire Tier B design and it is cheap to run.
- **Compute.** No GPU in the authoring container and `huggingface.co` is egress-blocked here.
  All runs happen on external hardware; this repo holds code, configs, and results only.

---

## 9. Open questions for you

1. ~~**Compute**~~ — answered: 1 × A100 80GB. Plan re-costed in `docs/COMPUTE.md`;
   pixel generation descoped as a consequence (§7).
2. **Generation base:** bolt a decoder onto Qwen3-VL, or switch to a unified any-to-any base
   at Phase 3? Decide with Phase 1 data — but if you already have a preference it changes what
   Phase 0 builds.
3. **Venue and clock:** which deadline are we aiming at? It determines whether Phase 1 ships
   as its own workshop paper or gets folded into the main submission.
4. **Scope discipline:** are you willing to ship the Phase 1 measurement paper alone if H4
   fails? Deciding yes now makes the kill criteria real rather than decorative.
