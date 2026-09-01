"""Cycle consistency: T -> I -> T read-back (PLAN.md §5.5).

Render a span, feed the pixels back, ask for the string. This direction is
*information-preserving in principle* — the pixels contain the text exactly —
so anything less than near-perfect recovery is a pure routing failure rather
than a lossy-channel result. That asymmetry is why this half of the cycle
survives the pixel-generation descope intact while `I -> T -> I` does not.

It has a second job. Every Tier B metric is reported both unconditionally and
conditioned on spans the model reads correctly (PLAN.md §5.3a), and this is
what produces that mask: a misread span yields a large MSG for reasons that
have nothing to do with representational geometry.

Canonical implementation. `scripts/gate0_sweep.py` re-exports these rather than
carrying its own copy, so the read-back that chose the render config and the
read-back that conditions the metrics cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

PROMPT = ("Transcribe the text in this image exactly. "
          "Reply with only the text, nothing else.")


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalise(s: str) -> str:
    """Judge read-back as the transcription task it is: case and wrapping
    punctuation are not what Tier B is testing."""
    return " ".join(s.strip().strip('"\'`.,:;!?').lower().split())


@torch.no_grad()
def read_back(model, processor, images: Sequence, batch: int = 32) -> list[str]:
    """Transcribe each image. Greedy, so this measures the model rather than
    sampling noise.

    Uses **left** padding: batched generation with right padding decodes the
    shorter sequences in a batch as garbage. The metric runner needs the
    opposite (right padding keeps absolute indices valid), so the side is set
    explicitly here rather than inherited.
    """
    tok = getattr(processor, "tokenizer", processor)
    prev_side = getattr(tok, "padding_side", None)
    tok.padding_side = "left"
    try:
        out: list[str] = []
        for i in range(0, len(images), batch):
            chunk = list(images[i:i + batch])
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": chunk[0]},
                {"type": "text", "text": PROMPT}]}]
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            inputs = processor(text=[text] * len(chunk), images=chunk,
                               padding=True, return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            new = gen[:, inputs["input_ids"].shape[1]:]
            out.extend(processor.batch_decode(new, skip_special_tokens=True))
        return out
    finally:
        if prev_side is not None:
            tok.padding_side = prev_side


@dataclass
class ReadBackResult:
    accuracy: float
    cer: float
    correct: list[bool]
    predictions: list[str]

    def __str__(self) -> str:
        return (f"read-back {self.accuracy:.3f} exact, CER {self.cer:.4f} "
                f"({sum(self.correct)}/{len(self.correct)})")


def score(truths: Sequence[str], predictions: Sequence[str]) -> ReadBackResult:
    """Exact-match accuracy and character error rate."""
    if len(truths) != len(predictions):
        raise ValueError(f"{len(truths)} truths for {len(predictions)} predictions")
    correct, num, den = [], 0, 0
    for truth, pred in zip(truths, predictions):
        t, p = normalise(truth), normalise(pred)
        correct.append(t == p)
        num += levenshtein(p, t)
        den += max(1, len(t))
    return ReadBackResult(accuracy=sum(correct) / max(1, len(correct)),
                          cer=num / max(1, den), correct=correct,
                          predictions=list(predictions))


def mark_read_ok(model, processor, items, batch: int = 32) -> ReadBackResult:
    """Run read-back over a corpus and set each item's `read_ok` in place.

    Scores the *primary* render only. The control render is a different font of
    the same span, and conditioning on both would discard items for a property
    of the denominator rather than of the measurement.
    """
    preds = read_back(model, processor, [it.span_image for it in items], batch)
    result = score([it.span_text for it in items], preds)
    for item, ok in zip(items, result.correct):
        item.read_ok = bool(ok)
    return result
