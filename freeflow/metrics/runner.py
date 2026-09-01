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

from ..data.views import SpanItem, batch_by_suffix  # noqa: F401
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


# Qwen's placeholder, kept as the fallback because the frozen Gate 0 config and
# every measurement so far were taken with it.
_QWEN_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"


def image_placeholder(processor) -> tuple[str, int | None]:
    """The string that marks where an image goes, for *this* processor.

    Hardcoding Qwen's `<|vision_start|><|image_pad|><|vision_end|>` silently
    confined the whole instrument to one model family: `llava-hf` uses
    `<image>`, Gemma uses its own, and a sweep meant to ask "is this true of
    every decoder VLM" could only ever answer for Qwen. Derived from the
    processor instead, with the token id returned so the caller can verify the
    image actually entered the sequence.
    """
    token = getattr(processor, "image_token", None)
    tok = getattr(processor, "tokenizer", processor)
    if not isinstance(token, str) or not token:
        return _QWEN_IMAGE, None

    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    text = token
    # Qwen-style processors expose the bare pad token and expect it wrapped.
    if "<|vision_start|>" in vocab and "<|vision_end|>" in vocab:
        text = f"<|vision_start|>{token}<|vision_end|>"
    return text, vocab.get(token)


def call_processor(processor, texts: list[str], images: list | None = None,
                   **kw):
    """Tokenise, tolerating either image-batching convention.

    Processors disagree about how a batch of images arrives: Qwen takes a flat
    list, Gemma-3 wants one list per text and rejects the flat form with
    "inconsistently sized batches". Trying flat first keeps every measurement
    taken so far byte-identical, and the retry is what lets a second family be
    measured at all.
    """
    kwargs = dict(text=texts, padding=True, return_tensors="pt", **kw)
    if not images:
        return processor(**kwargs)
    try:
        return processor(**kwargs, images=images)
    except ValueError as exc:
        if "inconsistent" not in str(exc).lower():
            raise
        return processor(**kwargs, images=[[im] for im in images])


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
        placeholder, image_id = image_placeholder(processor)
        texts, images = [], []
        for it in items:
            span = (it.span_text if variant == "primary" else it.span_paraphrase)
            img = (it.span_image if variant == "primary" else it.span_image_alt)
            if modality == "text":
                texts.append(f"{it.prefix}{span}{it.suffix}")
            else:
                texts.append(f"{it.prefix}{placeholder}{it.suffix}")
                images.append(img)

        inputs = call_processor(processor, texts, images)
        # The same batch without its suffix, so the suffix's token count comes
        # from differencing. Tokenising the suffix alone is wrong here for the
        # reason it is wrong for option strings: a SentencePiece leading space
        # is its own token at sequence start and merges into the next one
        # mid-sequence, so standalone ids need not match the ids in context.
        short = call_processor(processor,
                               [t[:t.rindex(it.suffix)]
                                for t, it in zip(texts, items)], images)

        # An image view that contains no image tokens is text compared with
        # text: MSG would come out near zero and look like perfect alignment.
        # The wrong placeholder for an unfamiliar processor is exactly how that
        # happens, so it is checked rather than assumed.
        if modality != "text" and "pixel_values" in inputs and image_id is not None:
            if not (inputs["input_ids"] == image_id).any():
                raise ValueError(
                    f"the image view contains no image tokens: placeholder "
                    f"{placeholder!r} (id {image_id}) never appears in the "
                    "tokenised sequence. This processor marks images "
                    "differently, and measuring it would compare text with "
                    "text")

        # merge_index = unpadded length - suffix length. Uses the attention
        # mask rather than the padded shape so it is right for every item.
        lengths = inputs["attention_mask"].sum(dim=1)
        suffix_lens = (lengths - short["attention_mask"].sum(dim=1)).tolist()
        if len(set(suffix_lens)) != 1:
            raise ValueError(
                "items in a batch must share a suffix length; got "
                f"{sorted(set(suffix_lens))}. Group items by suffix, or the "
                "JSD slices will not line up across views.")
        k = int(suffix_lens[0])
        if k <= 0:
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
    """Verify every row in the batch ends in the *same* k tokens.

    A merge straddling the span/suffix boundary would shift one row's merge
    index without raising, and corrupt every number downstream. The check is
    that the rows agree with *each other*, not that they agree with the suffix
    tokenised in isolation: a SentencePiece leading space is a separate token at
    sequence start and merges into the following one mid-sequence, so the
    standalone form legitimately differs while the index stays right. Comparing
    against it aborted the whole llava-v1.6 run over one token id at equal
    length.

    Agreement *across views* is the property the merge position actually needs,
    and `capture` checks that where both views exist.
    """
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    tails = set()
    for row in range(ids.shape[0]):
        n = int(mask[row].sum())
        tails.add(tuple(ids[row, n - k:n].tolist()))
    if len(tails) > 1:
        a, b = sorted(tails)[:2]
        raise ValueError(
            f"rows in this batch do not share their final {k} tokens: {list(a)} "
            f"vs {list(b)}. A token merged across the span/suffix boundary for "
            "some items, so their merge positions differ. Start the suffix at "
            "an unambiguous boundary — a leading space usually suffices.")


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
    # How deep into the shared suffix the views could be compared. Equals the
    # suffix length when every view tokenises it identically; smaller when a
    # tokeniser merged a leading space backwards across the span boundary, which
    # is a property of the model and belongs in the record.
    merge_depth: int | None = None


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
def _alignment_depth(built: dict) -> int:
    """How deep into the shared suffix the four views agree, in tokens.

    The merge position is meant to be the first token after the span, and for
    Qwen it is: the suffix tokenises identically in every view, so this returns
    each view's own suffix length and nothing changes.

    It is not universal. A SentencePiece tokeniser merges a leading space
    backwards into the preceding token, and what precedes the suffix differs by
    construction -- a written word in the text view, image tokens in the image
    view. On llava-v1.6 the text view produced `▁on` where the image view
    produced `▁` then `on`, so the image view's suffix was a token longer and
    the two "first suffix tokens" were different tokens. Comparing hidden states
    there compares different positions.

    So the position is defined as the deepest point at which every view's
    terminal tokens still agree, bounded by the shortest suffix so it can never
    reach back into the span. Backing off costs a little of the suffix and
    keeps the comparison meaningful; reaching zero means the views cannot be
    aligned at all, which is worth an exception rather than a number.
    """
    ids = {k: vb.inputs["input_ids"] for k, vb in built.items()}
    lengths = {k: vb.inputs["attention_mask"].sum(dim=1) for k, vb in built.items()}
    ceiling = min(vb.suffix_len for vb in built.values())
    rows = next(iter(ids.values())).shape[0]

    best = ceiling
    for row in range(rows):
        d = best
        while d > 0:
            tails = {tuple(ids[k][row, int(lengths[k][row]) - d:
                                  int(lengths[k][row])].tolist())
                     for k in ids}
            if len(tails) == 1:
                break
            d -= 1
        if d == 0:
            raise ValueError(
                f"item {row}: no depth at which all four views share their "
                "final tokens, so there is no position where they can be "
                "compared. The suffix tokenises differently after a written "
                "span than after image tokens all the way down — start the "
                "suffix at an unambiguous boundary, such as a newline.")
        best = min(best, d)
    return best


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

    # Batched by suffix, not by position: build_view locates merge positions as
    # len(sequence) - len(suffix_tokens), so a batch mixing suffix lengths would
    # misalign every index. Item order in the results follows these batches.
    for chunk in batch_by_suffix(items, batch):
        primary_logits = {}

        # Build every view before forwarding any of them: the merge position has
        # to be agreed *across* views and no single view can determine it.
        built = {key: build_view(processor, chunk, modality, variant, device)
                 for key, (modality, variant) in views.items()}
        depth = _alignment_depth(built)
        for key in results:
            prior = results[key].merge_depth
            results[key].merge_depth = (depth if prior is None
                                        else min(prior, depth))
        for vb in built.values():
            vb.merge_index = vb.inputs["attention_mask"].sum(dim=1) - depth
            vb.suffix_len = depth

        # After alignment this must hold. Kept as an assertion rather than
        # trusted: a silent misalignment corrupts every distance downstream and
        # nothing further along would notice.
        merge_tokens = {k: vb.inputs["input_ids"].gather(
            1, vb.merge_index.view(-1, 1)).squeeze(1) for k, vb in built.items()}
        ref = merge_tokens["text"]
        for key, got in merge_tokens.items():
            if not torch.equal(got, ref):
                bad = int((got != ref).nonzero()[0])
                raise ValueError(
                    f"alignment failed: view {key!r} holds a different token at "
                    f"the merge position than the text view (item {bad}: "
                    f"{int(got[bad])} vs {int(ref[bad])})")

        for key, vb in built.items():
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
        del built

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
