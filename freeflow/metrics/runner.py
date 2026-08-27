"""Model-facing half of the metric suite: forwards, merge positions, capture.

The metric math lives in the sibling modules and is testable on CPU. This is
the part that needs a GPU, and it exists to hand those modules two things:
merge-position hidden states, and logits over the shared continuation.

**Locating the merge position.** The two views have different lengths — after
Gate 0 the image view spends 6 visual tokens where the text view spends 1-2 —
so the merge position sits at a different index in each. Rather than counting
span tokens (which requires knowing how the processor expanded the image), use
the fact that the *suffix is identical and terminal* in both views:

    merge_index = len(sequence) - len(suffix_tokens)

Model-agnostic, and it never has to reason about image-token expansion. It does
assume the suffix tokenises the same way in both views, which fails if a token
straddles the span/suffix boundary — so that assumption is asserted at runtime
rather than trusted.

**Padding side matters and differs by task.** Forward passes here use *right*
padding, so absolute indices computed on the unpadded sequence stay valid.
Batched *generation* needs left padding (see scripts/gate0_sweep.py). Getting
this backwards silently corrupts either the indices or the decoded text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..data.views import SpanItem  # noqa: F401  (re-exported)
from .distribution import StreamingJSD, logits_at


@dataclass
class ViewBatch:
    """Tokenised inputs plus, for each item, where the merge position sits."""

    inputs: dict
    merge_index: torch.Tensor        # [N] index of the first suffix token
    suffix_len: int

    @property
    def batch_size(self) -> int:
        return int(self.merge_index.numel())


def _tok_len(processor, text: str) -> int:
    tok = getattr(processor, "tokenizer", processor)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _suffix_ids(processor, suffix: str) -> list[int]:
    tok = getattr(processor, "tokenizer", processor)
    return tok(suffix, add_special_tokens=False)["input_ids"]


def build_view(processor, items: Sequence[SpanItem], modality: str,
               variant: str = "primary", device: str = "cuda:0") -> ViewBatch:
    """Tokenise one view of a batch and locate each item's merge position.

    `modality` is "text" or "image"; `variant` is "primary" or "control", which
    selects the paraphrase / alternate render used for the MSG denominator.
    """
    tok = getattr(processor, "tokenizer", processor)
    prev_side = getattr(tok, "padding_side", None)
    tok.padding_side = "right"       # keep absolute indices valid
    try:
        texts, images = [], []
        for it in items:
            span = (it.span_text if variant == "primary" else it.span_paraphrase)
            img = (it.span_image if variant == "primary" else it.span_image_alt)
            if modality == "text":
                texts.append(f"{it.prefix}{span}{it.suffix}")
            else:
                texts.append(f"{it.prefix}<|vision_start|><|image_pad|>"
                             f"<|vision_end|>{it.suffix}")
                images.append(img)

        kwargs = dict(text=texts, padding=True, return_tensors="pt")
        if images:
            kwargs["images"] = images
        inputs = processor(**kwargs)

        # merge_index = unpadded length - suffix length. Uses the attention
        # mask rather than the padded shape so it is right for every item.
        lengths = inputs["attention_mask"].sum(dim=1)
        suffix_lens = [_tok_len(processor, it.suffix) for it in items]
        if len(set(suffix_lens)) != 1:
            raise ValueError(
                "items in a batch must share a suffix length; got "
                f"{sorted(set(suffix_lens))}. Group items by suffix, or the "
                "JSD slices will not line up across views.")
        k = suffix_lens[0]
        if k == 0:
            raise ValueError("suffix is empty: there is no merge position after "
                             "the span, and nothing to score the continuation on")
        merge_index = lengths - k

        _assert_clean_boundary(inputs, items, k, processor)
        inputs = {kk: (v.to(device) if torch.is_tensor(v) else v)
                  for kk, v in inputs.items()}
        return ViewBatch(inputs=inputs, merge_index=merge_index.to(device),
                         suffix_len=k)
    finally:
        if prev_side is not None:
            tok.padding_side = prev_side


def _assert_clean_boundary(inputs, items, k: int, processor) -> None:
    """Verify the suffix really is a token-level suffix of the sequence.

    Tokenisation is not compositional: a merge can straddle the span/suffix
    boundary, which would silently shift every merge index by one. Cheap to
    check, and a wrong index would corrupt every number downstream without
    ever raising.
    """
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    expected = _suffix_ids(processor, items[0].suffix)
    for row in range(ids.shape[0]):
        n = int(mask[row].sum())
        tail = ids[row, n - k:n].tolist()
        if tail != expected:
            raise ValueError(
                f"item {row}: suffix does not tokenise as a clean suffix "
                f"({tail} != {expected}). Start the suffix at an unambiguous "
                "boundary — a leading space usually suffices.")


def gather_merge_hidden(hidden_states: Sequence[torch.Tensor],
                        merge_index: torch.Tensor,
                        layers: Sequence[int]) -> list[torch.Tensor]:
    """Pull `[N, D]` merge-position states out of the requested layers."""
    idx = merge_index.view(-1, 1, 1)
    out = []
    for layer in layers:
        h = hidden_states[layer]                                  # [N, L, D]
        out.append(h.gather(1, idx.expand(-1, 1, h.shape[-1])).squeeze(1))
    return out


def continuation_logits(model, hidden_last: torch.Tensor,
                        merge_index: torch.Tensor, k: int) -> torch.Tensor:
    """Logits that predict the shared suffix, as `[N, K, V]`.

    Position i predicts token i+1, so scoring suffix tokens at
    `merge_index .. merge_index+k-1` needs logits at `merge_index-1 .. +k-2`.
    The first of those is the distribution over what comes *next* after the
    span — the single most diagnostic position in the sequence, and the one
    the whole substitution question turns on.

    Only these positions are projected to vocabulary. Letting the model produce
    logits for the full sequence would allocate ~8 GB at batch 32 / length 512
    over a 248k vocabulary, for a slice of a dozen positions.
    """
    n, _, d = hidden_last.shape
    starts = (merge_index - 1).clamp(min=0)
    offsets = torch.arange(k, device=hidden_last.device).view(1, -1)
    pos = starts.view(-1, 1) + offsets                            # [N, K]
    gathered = hidden_last.gather(1, pos.unsqueeze(-1).expand(n, k, d))
    return logits_at(model, gathered)


@dataclass
class CaptureResult:
    """Everything the metric modules need, and nothing they don't."""

    hidden: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    jsd: StreamingJSD = field(default_factory=StreamingJSD)
    span_ids: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    read_ok: list[bool] = field(default_factory=list)


def default_layers(n_hidden: int, k: int = 8) -> list[int]:
    """Evenly spaced layers plus the final one.

    Capturing all of them is ~20x the disk for nothing the sparse set does not
    already show (docs/COMPUTE.md), and the final layer is always included
    because it is what the readout actually sees.
    """
    if n_hidden <= k:
        return list(range(n_hidden))
    step = (n_hidden - 1) / (k - 1)
    idx = sorted({int(round(i * step)) for i in range(k)} | {n_hidden - 1})
    return idx


@torch.no_grad()
def capture(model, processor, items: Sequence[SpanItem], batch: int = 32,
            layers: Sequence[int] | None = None,
            device: str = "cuda:0") -> dict[str, CaptureResult]:
    """Run all four views over `items` and capture what the metrics need.

    Returns one CaptureResult per view key: `text`, `image`, `text_control`,
    `image_control`. The JSD is accumulated on the primary pair only — the
    controls exist to set the MSG's scale, not to be compared with each other.
    """
    views = {"text": ("text", "primary"), "image": ("image", "primary"),
             "text_control": ("text", "control"),
             "image_control": ("image", "control")}
    results = {k: CaptureResult() for k in views}
    chosen: list[int] | None = list(layers) if layers is not None else None

    for start in range(0, len(items), batch):
        chunk = list(items[start:start + batch])
        primary_logits = {}

        for key, (modality, variant) in views.items():
            vb = build_view(processor, chunk, modality, variant, device)
            out = model(**vb.inputs, output_hidden_states=True)
            if chosen is None:
                chosen = default_layers(len(out.hidden_states))

            results[key].hidden.setdefault("layers", chosen)
            per_layer = gather_merge_hidden(out.hidden_states, vb.merge_index, chosen)
            for i, h in enumerate(per_layer):
                results[key].hidden.setdefault(str(i), []).append(h.float().cpu())

            if key in ("text", "image"):
                primary_logits[key] = continuation_logits(
                    model, out.hidden_states[-1], vb.merge_index, vb.suffix_len)
            del out
            torch.cuda.empty_cache()

        results["text"].jsd.update(primary_logits["text"], primary_logits["image"])
        del primary_logits

        for key in views:
            results[key].span_ids.extend(it.span_id for it in chunk)
            results[key].groups.extend(it.group for it in chunk)
            results[key].read_ok.extend(bool(it.read_ok) for it in chunk)

    for res in results.values():
        for k in list(res.hidden):
            if k != "layers":
                res.hidden[k] = torch.cat(res.hidden[k])
    return results
