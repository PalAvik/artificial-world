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
text: a 2-alternative choice scored by argmax logprob has no refusals, no
formatting drift, and no prompt-following confound between modalities.
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

    @property
    def delta(self) -> float:
        return self.text.accuracy - self.image.accuracy

    @property
    def image_above_chance(self) -> float:
        return self.image.accuracy - self.image.chance

    def task_sanity(self, floor: float = 0.90) -> str | None:
        """None when the task itself works.

        The text view is the positive control: the span is written out in the
        context, so a model that cannot answer there is telling you the question
        is broken, not that the representation is. A *negative* delta — image
        beating text — is the same signal, and was how the first run's
        modality-specific phrasing was caught.
        """
        # Negative delta first: it is the more specific diagnosis. It says the
        # two views are answering different questions, where a low text score
        # alone only says something is wrong.
        if self.delta < -0.05:
            return (f"delta is {self.delta:+.3f}: the image view beats the text "
                    "view, which cannot be right when the text view can simply "
                    "read the answer. The two views are not answering the same "
                    "question")
        if self.text.accuracy < floor:
            return (f"text view scores {self.text.accuracy:.3f} with the span "
                    f"written out in front of it — below {floor:.2f}, the task "
                    "is mis-specified and the delta measures the task")
        return None

    def validity(self, margin: float = 0.15) -> str | None:
        """None when the image span demonstrably carries its content."""
        broken = self.task_sanity()
        if broken:
            return broken
        if self.image_above_chance < margin:
            return (f"image view scores {self.image.accuracy:.3f} against chance "
                    f"{self.image.chance:.2f} — the model cannot recover the span "
                    "from the image, so this tier's MSG measures the instrument "
                    "rather than the representation")
        return None

    def __str__(self) -> str:
        return (f"text {self.text} | image {self.image} | "
                f"delta {self.delta:+.3f}")


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
def _option_logprobs(model, processor, items: Sequence[SpanItem], modality: str,
                     options: Sequence[str], device: str) -> torch.Tensor:
    """Mean per-token logprob of each option, appended after the item's suffix.

    Length-normalised, so a shorter option is not favoured for having fewer
    tokens to be wrong about.

    **The option's token count is obtained by differencing**, not by tokenising
    the option on its own. Tokenisation is not compositional: an option glued to
    the preceding text can merge across the boundary, so standalone ids need not
    match the ids actually in the sequence, and the scored positions would drift
    off the option silently. Same hazard the merge index avoids the same way.
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

        out = model(**vb_full.inputs, output_hidden_states=False)
        logprobs = out.logits[0].float().log_softmax(dim=-1)
        # Position i predicts token i+1, so the k option tokens at
        # [n_full-k, n_full) are scored by logits at [n_full-k-1, n_full-1).
        idx = torch.arange(n_full - k - 1, n_full - 1, device=logprobs.device)
        target = vb_full.inputs["input_ids"][0, n_full - k:n_full]
        scores.append(float(logprobs[idx, target].mean()))
    return torch.tensor(scores)


def forced_choice(model, processor, items: Sequence[SpanItem],
                  seed: int = 0, device: str = "cuda:0") -> FunctionalResult:
    """Score both views on the same 2-alternative choice.

    Batched one item at a time: each item carries its own option strings, so a
    batch would mix suffix lengths, which `build_view` refuses for good reason.
    This is the slow path of Phase 1 and is worth capping with `--functional-n`.
    """
    distractors = make_distractors(items, seed)
    results = {}
    for modality in ("text", "image"):
        true_lp = _option_logprobs(model, processor, items, modality,
                                   [it.span_text for it in items], device)
        false_lp = _option_logprobs(model, processor, items, modality,
                                    distractors, device)
        correct = (true_lp > false_lp).tolist()
        results[modality] = ChoiceResult(
            accuracy=sum(correct) / max(1, len(correct)), n=len(correct),
            correct=correct)
    return FunctionalResult(text=results["text"], image=results["image"])
