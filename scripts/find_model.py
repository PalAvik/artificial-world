#!/usr/bin/env python3
"""List candidate base models on the Hub, flagging which ones can actually see.

Gate 0 task: confirm the base model repo id. This matters more than it sounds —
`Qwen/Qwen3.5-2B` is a TEXT-ONLY LLM with no vision tower, so it cannot host this
experiment at all, and the Qwen3.5 vs Qwen3-VL naming makes that easy to miss.

A `pipeline_tag` of `image-text-to-text` means the model takes images.
`text-generation` means it does not.

Usage:
    python scripts/find_model.py
    python scripts/find_model.py --search Qwen3-VL --limit 40
"""
from __future__ import annotations

import argparse
import inspect

DEFAULT_QUERIES = ["Qwen3.5-VL", "Qwen3-VL", "Qwen3.5"]
VISION_TAGS = {"image-text-to-text", "visual-question-answering",
               "image-to-text", "any-to-any"}


def inspect_repos(repo_ids: list[str]) -> int:
    """Read each repo's config.json and say whether it can actually take images.

    `pipeline_tag` is uploader-set metadata and can be wrong or missing.
    `config.json` is what transformers loads, so it is the authority.

    This matters for the Qwen3.5 family specifically: the architecture is
    natively multimodal, in the same shape as Gemma 3 —

        model_type "qwen3_5"       -> Qwen3_5Config, has vision_config + text_config
                                      -> Qwen3_5ForConditionalGeneration (SEES)
        model_type "qwen3_5_text"  -> Qwen3_5TextConfig
                                      -> Qwen3_5ForCausalLM (TEXT ONLY)

    So there is no separate "Qwen3.5-VL" line to look for — vision is folded into
    the main one, and only the checkpoint's own config says which variant it is.
    """
    import json

    from huggingface_hub import hf_hub_download

    print("Reading config.json — the authority, unlike pipeline_tag.\n")
    for repo in repo_ids:
        try:
            path = hf_hub_download(repo, "config.json")
            cfg = json.load(open(path))
        except Exception as exc:
            print(f"{repo}\n  ! {type(exc).__name__}: {exc}\n")
            continue

        mt = cfg.get("model_type", "?")
        vision = cfg.get("vision_config")
        arch = ", ".join(cfg.get("architectures") or []) or "?"
        verdict = "SEES IMAGES" if vision else "TEXT ONLY"
        print(f"{repo}")
        print(f"  model_type:    {mt}")
        print(f"  architectures: {arch}")
        print(f"  vision_config: {'present' if vision else 'ABSENT'}")
        print(f"  -> {verdict}\n")
    print("Use a SEES IMAGES model as the base. A TEXT ONLY one is the ablation")
    print("baseline only — it cannot host the experiment (PLAN.md §6.1).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="append", dest="queries",
                    help="repeatable; defaults to the Qwen VL/text families")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--inspect", action="append", dest="inspect",
                    help="repo id to inspect authoritatively via its config.json; "
                         "repeatable")
    args = ap.parse_args()

    if args.inspect:
        return inspect_repos(args.inspect)

    from huggingface_hub import HfApi

    api = HfApi()

    # `direction` was removed in huggingface_hub 1.x, where "downloads" already
    # sorts descending. Older versions sort ascending without it, which silently
    # surfaces the least-used repos first — so pass it only where it exists.
    kwargs = {"sort": "downloads", "limit": args.limit}
    if "direction" in inspect.signature(api.list_models).parameters:
        kwargs["direction"] = -1

    for query in (args.queries or DEFAULT_QUERIES):
        print(f"\n=== {query}")
        try:
            models = list(api.list_models(search=query, **kwargs))
        except Exception as exc:
            print(f"  ! {type(exc).__name__}: {exc}")
            continue
        if not models:
            print("  (nothing found)")
            continue
        for m in models:
            tag = getattr(m, "pipeline_tag", None) or "-"
            dls = getattr(m, "downloads", None)
            sees = "VISION" if tag in VISION_TAGS else "text  "
            count = f"{dls:>10,}" if isinstance(dls, int) else " " * 10
            print(f"  {sees}  {m.id:<52} {tag:<22}{count}")

    print("\nPick the smallest VISION model in the newest family available.")
    print("If a small Qwen3.5-VL exists, prefer it and keep Qwen3-VL-2B as the")
    print("generational control. Qwen3.5-2B stays the text-only ablation baseline.")
    print("Record the chosen id in configs/ and in results/RESULTS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
