"""Per-architecture wiring the generic pipeline cannot infer.

Two things are architecture specific: where the decoder blocks live, and -- for
AWQ -- which module feeds each linear, since folding the per-input-channel scale
requires dividing it out of that producer (see
:func:`clc.quantizers.awq.fold_channel_scale`).

The groupings below are the standard LLaMA-family layout and cover every model
in the paper: LLaMA-3 / 3.1, Mistral, Qwen2.5 / Qwen3.
"""

from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn

ScaleGroup = Tuple[str, nn.Module, List[nn.Module]]
"""``(name, producer, consumers)``: ``consumers`` share one input, made by ``producer``."""

LLAMA_LIKE = ("llama", "mistral", "qwen2", "qwen3", "mixtral")


def decoder_blocks(model: nn.Module) -> nn.ModuleList:
    """The list of transformer blocks to walk sequentially."""
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "layers"):
        current = model
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if isinstance(current, nn.ModuleList):
            return current
    raise ValueError(f"cannot locate decoder blocks on {type(model).__name__}")


def scale_groups(block: nn.Module) -> List[ScaleGroup]:
    """Producer/consumer groups for AWQ scale folding inside one decoder block.

    Returns an empty list when the block does not match a known layout, in which
    case AWQ can still run unfolded -- the result is numerically identical but
    not exportable.
    """
    attention = getattr(block, "self_attn", None)
    mlp = getattr(block, "mlp", None)
    input_norm = getattr(block, "input_layernorm", None)
    post_norm = getattr(block, "post_attention_layernorm", None)
    if attention is None or mlp is None or input_norm is None or post_norm is None:
        return []

    groups: List[ScaleGroup] = []

    projections = [
        getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj")
    ]
    projections = [p for p in projections if p is not None]
    if projections:
        groups.append(("qkv_proj", input_norm, projections))

    # o_proj can only absorb v_proj's scale when their channel counts line up,
    # which fails under grouped-query attention with fewer KV heads.
    v_proj = getattr(attention, "v_proj", None)
    o_proj = getattr(attention, "o_proj", None)
    if v_proj is not None and o_proj is not None:
        if v_proj.out_features == o_proj.in_features:
            groups.append(("o_proj", v_proj, [o_proj]))

    gate_up = [getattr(mlp, name, None) for name in ("gate_proj", "up_proj")]
    gate_up = [p for p in gate_up if p is not None]
    if gate_up:
        groups.append(("gate_up_proj", post_norm, gate_up))

    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    if up_proj is not None and down_proj is not None:
        groups.append(("down_proj", up_proj, [down_proj]))

    return groups
