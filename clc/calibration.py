"""Calibration sample loading.

The paper uses 128 sequences of 2048 tokens from C4, the standard GPTQ/AWQ
setting.  Two details matter and are easy to get wrong:

* sequences are sampled from a random window *inside* a document rather than
  truncated from its start, so the calibration set is not biased toward document
  openings;
* the requested sequence length must survive all the way to the forward pass.
  Silently re-tokenizing at a shorter length changes both the activation
  statistics and the reported setting.
"""

from __future__ import annotations

import hashlib
import pickle
import random
from pathlib import Path
from typing import List, Optional

import torch

SUPPORTED_DATASETS = ("c4", "wikitext2")


def _cache_key(dataset: str, tokenizer, n_samples: int, seqlen: int, seed: int) -> str:
    identity = "|".join(
        str(x)
        for x in (
            dataset,
            getattr(tokenizer, "name_or_path", "unknown"),
            getattr(tokenizer, "vocab_size", 0),
            len(tokenizer),
            n_samples,
            seqlen,
            seed,
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _validate(samples: List[torch.Tensor], tokenizer, seqlen: int) -> bool:
    """Reject a cache written by a different tokenizer or a different length."""
    if not samples:
        return False
    vocab = len(tokenizer)
    for sample in samples:
        if sample.dim() != 2 or sample.shape[1] != seqlen:
            return False
        if int(sample.max()) >= vocab or int(sample.min()) < 0:
            return False
    return True


def _load_c4(tokenizer, n_samples: int, seqlen: int, seed: int) -> List[torch.Tensor]:
    from datasets import load_dataset

    dataset = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        streaming=True,
    )
    rng = random.Random(seed)
    samples: List[torch.Tensor] = []

    # A cheap character-length prefilter avoids tokenizing documents that cannot
    # possibly yield `seqlen` tokens.
    minimum_characters = seqlen * 3
    for record in dataset:
        if len(samples) >= n_samples:
            break
        text = record.get("text", "")
        if len(text) < minimum_characters:
            continue
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        start = rng.randint(0, ids.shape[1] - seqlen - 1)
        samples.append(ids[:, start : start + seqlen])

    if len(samples) < n_samples:
        raise RuntimeError(
            f"only found {len(samples)} of {n_samples} C4 documents with at least "
            f"{seqlen} tokens"
        )
    return samples


def _load_wikitext2(tokenizer, n_samples: int, seqlen: int, seed: int) -> List[torch.Tensor]:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids
    chunks = ids.shape[1] // seqlen
    if chunks < n_samples:
        raise RuntimeError(f"wikitext-2 yields only {chunks} chunks of length {seqlen}")

    rng = random.Random(seed)
    return [
        ids[:, index * seqlen : (index + 1) * seqlen]
        for index in rng.sample(range(chunks), n_samples)
    ]


def load_calibration_data(
    dataset: str,
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    cache_dir: Optional[str | Path] = None,
) -> List[torch.Tensor]:
    """Return ``n_samples`` token-id tensors of shape ``[1, seqlen]``."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{dataset}_{_cache_key(dataset, tokenizer, n_samples, seqlen, seed)}.pkl"
        if cache_path.exists():
            try:
                cached = pickle.loads(cache_path.read_bytes())
            except (pickle.UnpicklingError, EOFError):
                cache_path.unlink(missing_ok=True)
            else:
                if _validate(cached, tokenizer, seqlen):
                    return cached
                cache_path.unlink(missing_ok=True)

    loader = _load_c4 if dataset == "c4" else _load_wikitext2
    samples = loader(tokenizer, n_samples, seqlen, seed)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(samples))
    return samples
