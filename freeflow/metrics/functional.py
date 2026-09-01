"""Forced-choice comprehension: the validity check and the functional delta.

One mechanism, two jobs that Gate 1 needs and that hidden-state geometry cannot
supply on its own.

**Validity.** MSG compares representations, so it produces a confident number
whether or not the image span means anything to the model. A diagram the model
cannot decode yields a large gap for an uninteresting reason. Accuracy on the
*image* view alone answers that: at chance, the image span carries nothing and
that tier's MSG is measuring the failure of the instrument.

Tier B got this check for free — read-back transcribes glyphs. Tier C did not,
because there is no text in a relation diagram to transcribe, and the first
Phase 1 run reported read-back 0.000 on Tier C for exactly that reason. This
module is the missing generalisation: *can the model recover the span's content
from the image, whatever kind of span it is?*

**Functional delta.** Gate 1's second criterion is `|acc(V_T) - acc(V_I)| >= 5
points` on a matched task. The same forced choice, asked of both views, is that
task.

Scoring is by likelihood over the two options rather than by parsing generated
text: a 2-alternative choice has no refusals, no formatting drift, and no
prompt-following confound between modalities.

**An ablated view is scored alongside the two real ones**: the same choice with
the span replaced by a blank, so neither modality carries it. It is the floor
the other two have to clear. Without it, a high accuracy is unreadable — the
first three Phase 1 runs reported image *above* text on Tier B, which cannot
happen if both views are recovering the span, and can happen easily if neither
is and the context alone decides. Chance is the wrong floor for that reason:
chance assumes the options are indistinguishable without the span, and same-group
distractors are not.

**Scored as PMI against a null context**, not as raw likelihood. Raw likelihood
lets an option's unconditional frequency compete with the in-context evidence: a
rare true span loses to a common distractor even when the answer is written in
the context. Subtracting `logP(option | question alone)` cancels that, and the
same null score is used for both views so the delta stays a statement about the
modalities rather than about the vocabulary.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch

from ..data.views import SpanItem
from .runner import build_view

# Modality-neutral on purpose. The first version asked about the "hidden part",
# which is only coherent for the image view — nothing is hidden when the span is
# written out — so the two views were not answering the same question and the
# delta measured the mismatch rather than the substitution.
QUESTION = " The missing word is"

# What the span is replaced by in the ablated view. A blank rather than a
# deletion: removing it would change the sentence's shape as well as its
# content, and the floor is meant to isolate the content.
BLANK = "___"

# Above this the image view has no headroom left, and a delta measured against
# it is a bound rather than a measurement.
CEILING = 0.99


@dataclass
class ChoiceResult:
    accuracy: float
    n: int
    chance: float = 0.5
    correct: list[bool] | None = None

    def __str__(self) -> str:
        return f"{self.accuracy:.3f} (n={self.n}, chance {self.chance:.2f})"


@dataclass
class FunctionalResult:
    """Both views' accuracy, and the delta Gate 1 tests."""

    text: ChoiceResult
    image: ChoiceResult
    # The span-free floor. Optional so a caller can construct a result without
    # it, but Phase 1 always measures it.
    ablated: ChoiceResult | None = None

    @property
    def delta(self) -> float:
        return self.text.accuracy - self.image.accuracy

    @property
    def floor(self) -> float:
        """What the choice scores with no span in either modality."""
        return self.ablated.accuracy if self.ablated else self.image.chance

    @property
    def image_above_floor(self) -> float:
        return self.image.accuracy - self.floor

    @property
    def image_above_chance(self) -> float:
        return self.image.accuracy - self.image.chance

    @property
    def image_at_ceiling(self) -> bool:
        return self.image.accuracy >= CEILING

    def task_sanity(self, floor: float = 0.90) -> str | None:
        """None when the task itself works.

        The text view is the positive control: the span is written out in the
        context, so a model that cannot answer there is telling you the question
        is broken, not that the representation is.
        """
        # The floor first: it subsumes the others. If the choice is decidable
        # without the span, then every accuracy here is a statement about the
        # contexts and the distractors, and the delta between two views of a
        # span neither view needs is not interpretable at all.
        if self.ablated is not None and self.text.accuracy - self.floor < 0.05:
            return (f"the span-free view scores {self.floor:.3f} against a text "
                    f"view of {self.text.accuracy:.3f} — the choice is decidable "
                    "from the context alone, so neither view is being scored on "
                    "the span and the delta means nothing")
        if self.text.accuracy < floor:
            return (f"text view scores {self.text.accuracy:.3f} with the span "
                    f"written out in front of it — below {floor:.2f}, the task "
                    "is mis-specified and the delta measures the task")
        return None

    def delta_verdict(self) -> str | None:
        """None when the functional delta is a usable measurement.

        Separate from `validity` because the two questions are independent: the
        image view can recover the span perfectly (valid) and still yield a
        delta that means nothing (saturated).

        Order matters. A *saturated* image view is checked before a negative
        delta, because saturation explains a negative delta and the reverse is
        not true. This ordering was wrong in the run of 2026-08-27: Tier B
        scored image 1.000 against text 0.941 and was reported as a broken task,
        when in fact the image view had simply run out of headroom. At Tier B the
        image *is* a rendering of the answer and read-back is 0.991, so the image
        view can read the span as directly as the text view can — and the frame
        ("the missing word is") is better posed for it, since nothing is missing
        when the span is written out. Image >= text is therefore expected there,
        not impossible, and the earlier rule asserted otherwise.

        The rule survives for tiers where the image does not contain the answer
        verbatim, which is where it was doing real work.
        """
        broken = self.task_sanity()
        if broken:
            return broken
        if self.image_at_ceiling:
            return (f"image view is at ceiling ({self.image.accuracy:.3f}) — a "
                    "saturated view cannot exhibit a cost, so the delta bounds "
                    "the functional gap near zero rather than measuring it. A "
                    "harder task is needed to put a number on it")
        if self.delta < -0.05:
            return (f"delta is {self.delta:+.3f}: the image view beats the text "
                    "view without being at ceiling, so the two views are not "
                    "answering the same question")
        return None

    def validity(self, margin: float = 0.15) -> str | None:
        """None when the image span demonstrably carries its content."""
        broken = self.task_sanity()
        if broken:
            return broken
        if self.image_above_floor < margin:
            label = ("the span-free floor" if self.ablated is not None
                     else "chance")
            return (f"image view scores {self.image.accuracy:.3f} against "
                    f"{label} {self.floor:.2f} — the model cannot recover the "
                    "span from the image, so this tier's MSG measures the "
                    "instrument rather than the representation")
        return None

    def __str__(self) -> str:
        base = (f"text {self.text} | image {self.image} | "
                f"delta {self.delta:+.3f}")
        if self.ablated is not None:
            base += f" | span-free floor {self.ablated.accuracy:.3f}"
        return base


def make_distractors(items: Sequence[SpanItem], seed: int = 0) -> list[str]:
    """One wrong option per item, drawn from the same group.

    Same group on purpose: a distractor from another word class would be
    separable on register alone, and the task would stop measuring whether the
    *span* was encoded.
    """
    rng = random.Random(seed)
    by_group: dict[str, list[str]] = {}
    for it in items:
        by_group.setdefault(it.group, []).append(it.span_text)
    out = []
    for it in items:
        pool = [s for s in set(by_group[it.group]) if s != it.span_text]
        if not pool:                       # single-span group: fall back globally
            pool = [s for s in {i.span_text for i in items} if s != it.span_text]
        out.append(rng.choice(pool) if pool else it.span_text + "?")
    return out


@torch.no_grad()
def _logprob_of_tail(model, inputs, n_full: int, k: int) -> float:
    """Mean logprob of the final `k` tokens of a sequence."""
    out = model(**inputs, output_hidden_states=False)
    logprobs = out.logits[0].float().log_softmax(dim=-1)
    # Position i predicts token i+1, so tokens at [n-k, n) are scored by the
    # logits at [n-k-1, n-1).
    idx = torch.arange(n_full - k - 1, n_full - 1, device=logprobs.device)
    target = inputs["input_ids"][0, n_full - k:n_full]
    return float(logprobs[idx, target].mean())


@torch.no_grad()
def _neutral_logprob(model, processor, option: str, device: str,
                     cache: dict) -> float:
    """The option's plausibility given the question frame alone.

    This is the calibration term. Without it the score is dominated by how
    common the option is: a rare true span ("pulchritude") loses to a frequent
    distractor ("justice") even with the answer written in the context, because
    the frequency gap outweighs the in-context evidence. Length-normalising the
    mean does not cancel that — only conditioning on a null context does.

    Identical for both views by construction, so subtracting it leaves the two
    modalities comparable, which is the whole point of the delta.
    """
    if option in cache:
        return cache[option]

    def encode(text: str):
        enc = processor(text=[text], padding=True, return_tensors="pt")
        enc = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in enc.items()}
        return enc, int(enc["attention_mask"][0].sum())

    # Differenced through the processor, for the reason `_option_scores` gives:
    # tokenising the question on its own can disagree with how it tokenises
    # with the option appended, and the processor may add specials that a bare
    # tokenizer call does not.
    _, n_short = encode(QUESTION)
    full, n_full = encode(QUESTION + " " + option)
    k = n_full - n_short
    cache[option] = _logprob_of_tail(model, full, n_full, k) if k > 0 else 0.0
    return cache[option]


@torch.no_grad()
def _option_scores(model, processor, items: Sequence[SpanItem], modality: str,
                   options: Sequence[str], device: str,
                   neutral_cache: dict) -> torch.Tensor:
    """Pointwise mutual information of each option with its context.

        score = logP(option | context) - logP(option | question alone)

    **The option's token count is obtained by differencing**, not by tokenising
    the option on its own. Tokenisation is not compositional: an option glued to
    the preceding text can merge across the boundary, so standalone ids need not
    match the ids in the sequence, and the scored positions would drift off the
    option silently. Same hazard the merge index avoids the same way.
    """
    scores = []
    for item, opt in zip(items, options):
        without = SpanItem(prefix=item.prefix, span_text=item.span_text,
                           suffix=item.suffix + QUESTION,
                           span_image=item.span_image,
                           span_paraphrase=item.span_paraphrase,
                           span_image_alt=item.span_image_alt,
                           span_id=item.span_id, group=item.group, tier=item.tier)
        with_opt = SpanItem(**{**without.__dict__,
                               "suffix": item.suffix + QUESTION + " " + opt})

        vb_short = build_view(processor, [without], modality, "primary", device)
        vb_full = build_view(processor, [with_opt], modality, "primary", device)
        n_short = int(vb_short.inputs["attention_mask"][0].sum())
        n_full = int(vb_full.inputs["attention_mask"][0].sum())
        k = n_full - n_short
        if k <= 0:
            scores.append(float("-inf"))
            continue

        conditioned = _logprob_of_tail(model, vb_full.inputs, n_full, k)
        neutral = _neutral_logprob(model, processor, opt, device, neutral_cache)
        scores.append(conditioned - neutral)
    return torch.tensor(scores)


def forced_choice(model, processor, items: Sequence[SpanItem],
                  seed: int = 0, device: str = "cuda:0") -> FunctionalResult:
    """Score both views on the same 2-alternative choice.

    Batched one item at a time: each item carries its own option strings, so a
    batch would mix suffix lengths, which `build_view` refuses for good reason.
    This is the slow path of Phase 1 and is worth capping with `--functional-n`.
    """
    distractors = make_distractors(items, seed)
    # One neutral score per distinct option string, shared across both views so
    # the calibration cannot differ by modality.
    neutral_cache: dict = {}
    blanked = [SpanItem(**{**it.__dict__, "span_text": BLANK,
                           "span_paraphrase": BLANK}) for it in items]
    views = {"text": (items, "text"), "image": (items, "image"),
             # Scored through the *text* path: with the span blanked there is
             # no image to substitute, and the point is to remove the span from
             # both modalities at once.
             "ablated": (blanked, "text")}

    results = {}
    for name, (view_items, modality) in views.items():
        true_lp = _option_scores(model, processor, view_items, modality,
                                 [it.span_text for it in items], device,
                                 neutral_cache)
        false_lp = _option_scores(model, processor, view_items, modality,
                                  distractors, device, neutral_cache)
        correct = (true_lp > false_lp).tolist()
        results[name] = ChoiceResult(
            accuracy=sum(correct) / max(1, len(correct)), n=len(correct),
            correct=correct)
    return FunctionalResult(text=results["text"], image=results["image"],
                            ablated=results["ablated"])
