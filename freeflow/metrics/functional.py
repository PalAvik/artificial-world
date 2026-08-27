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

    def validity(self, margin: float = 0.15) -> str | None:
        """None when the image span demonstrably carries its content."""
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

    Length-normalised, so a shorter option is not favoured simply for having
    fewer tokens to be wrong about.
    """
    tok = getattr(processor, "tokenizer", processor)
    probed = [SpanItem(prefix=it.prefix, span_text=it.span_text,
                       suffix=it.suffix + QUESTION + opt,
                       span_image=it.span_image,
                       span_paraphrase=it.span_paraphrase,
                       span_image_alt=it.span_image_alt, span_id=it.span_id,
                       group=it.group, tier=it.tier)
              for it, opt in zip(items, options)]

    scores = []
    for item, opt in zip(probed, options):
        vb = build_view(processor, [item], modality, "primary", device)
        out = model(**vb.inputs, output_hidden_states=False)
        logits = out.logits[0].float().log_softmax(dim=-1)
        opt_ids = tok(opt, add_special_tokens=False)["input_ids"]
        k = len(opt_ids)
        n = int(vb.inputs["attention_mask"][0].sum())
        # Position i predicts token i+1, so the option's tokens are scored by
        # the logits immediately preceding them.
        idx = torch.arange(n - k - 1, n - 1, device=logits.device)
        target = torch.tensor(opt_ids, device=logits.device)
        scores.append(float(logits[idx, target].mean()))
    return torch.tensor(scores)


QUESTION = " Was the hidden part:"


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
