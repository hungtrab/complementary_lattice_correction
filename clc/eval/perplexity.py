"""Perplexity evaluation.

Perplexity is the headline metric, so its normalization has to be right: the
negative log-likelihood is summed over *scored* tokens only, and must be divided
by that same count.  Dividing by the full window length instead -- which is easy
to do accidentally with a strided sliding window, where each window scores only
its non-overlapping tail -- deflates every number by roughly the overlap ratio
and makes runs with different strides incomparable.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


@torch.no_grad()
def evaluate_perplexity(
    model,
    input_ids: torch.Tensor,
    max_length: int = 2048,
    stride: Optional[int] = None,
    device: Optional[torch.device | str] = None,
) -> float:
    """Sliding-window perplexity over one long token sequence.

    Args:
        input_ids: shape ``[1, total_tokens]``.
        max_length: context window fed to the model.
        stride: how far the window advances.  Defaults to ``max_length``, i.e.
            non-overlapping windows, which is the standard PTQ setting.  A
            smaller stride gives each token more context at higher cost.
    """
    stride = stride or max_length
    device = device or next(model.parameters()).device
    total_tokens = input_ids.shape[1]

    negative_log_likelihood = 0.0
    scored_tokens = 0
    previous_end = 0

    for begin in range(0, total_tokens, stride):
        end = min(begin + max_length, total_tokens)
        target_length = end - previous_end
        if target_length <= 0:
            continue

        window = input_ids[:, begin:end].to(device)
        targets = window.clone()
        # Tokens already scored by an earlier window contribute context only.
        targets[:, :-target_length] = -100
        if (targets != -100).sum() < 2:
            continue

        outputs = model(window, labels=targets)
        # HuggingFace returns the mean over scored positions; the last token of a
        # window has no successor to predict, hence the -1.
        counted = int((targets != -100).sum()) - 1
        if counted <= 0:
            continue
        negative_log_likelihood += float(outputs.loss) * counted
        scored_tokens += counted
        previous_end = end

        if end == total_tokens:
            break

    if scored_tokens == 0:
        raise ValueError("no tokens were scored; the sequence is shorter than the window")
    return math.exp(negative_log_likelihood / scored_tokens)


def load_evaluation_tokens(dataset: str, tokenizer, split: str = "test") -> torch.Tensor:
    """Concatenated token ids of a standard perplexity corpus."""
    from datasets import load_dataset

    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    if dataset == "c4":
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            streaming=True,
        )
        texts = [record["text"] for _, record in zip(range(1100), data)]
        return tokenizer(" ".join(texts), return_tensors="pt").input_ids
    raise ValueError(f"unknown evaluation dataset {dataset!r}")
