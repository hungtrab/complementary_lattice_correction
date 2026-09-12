"""Writing corrected lattices out as deployable weight-only checkpoints.

Every layer is exported from ``(q, s, z)`` directly.  Nothing is re-quantized on
the way out, so unpacking a written checkpoint reproduces
``LatticeState.dequantize()`` exactly -- :func:`verify_layer` asserts precisely
that, and the export refuses to write a layer it cannot round-trip.

Format coverage, as accepted by vLLM:

===================  ==========  ==============================================
format               bit widths  notes
===================  ==========  ==============================================
``awq``              4           ``quantization="awq"`` / ``awq_marlin``
``gptq``             2, 3, 4, 8  ``quantization="gptq"`` / ``gptq_marlin``
``compressed-tensors`` 4, 8       ``pack-quantized`` WNA16
===================  ==========  ==============================================

The AWQ kernels are 4-bit only, so the 3-bit setting -- where the correction has
the most headroom -- has to be deployed through the GPTQ layout.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from clc.export.packing import (
    GPTQ_SUPPORTED_BITS,
    pack_awq,
    pack_gptq,
    unpack_awq,
    unpack_gptq,
)
from clc.lattice import LatticeState

SUPPORTED_FORMATS = ("awq", "gptq", "compressed-tensors")

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
)


class ExportError(RuntimeError):
    """A layer or model cannot be represented in the requested format."""


@dataclass
class PackedLayer:
    """One layer's packed tensors, keyed by their checkpoint suffix."""

    tensors: Dict[str, torch.Tensor]

    def prefixed(self, prefix: str) -> Dict[str, torch.Tensor]:
        return {f"{prefix}.{suffix}": tensor for suffix, tensor in self.tensors.items()}


def _check_exportable(state: LatticeState, name: str, out_features: int) -> None:
    if state.padded_in_features != state.in_features:
        raise ExportError(
            f"{name}: in_features={state.in_features} is not a multiple of "
            f"group_size={state.group_size}; the padded lattice has no packed "
            f"representation"
        )
    if state.symmetric:
        raise ExportError(
            f"{name}: symmetric lattices are not supported by the asymmetric "
            f"(q - z) * s checkpoint formats"
        )
    if out_features <= 0:
        raise ExportError(f"{name}: empty layer")
    if state.unfolded_scale is not None:
        raise ExportError(
            f"{name}: a per-input-channel scale was applied but never folded into "
            f"the producing module, so these codes describe Q(SW) while the model "
            f"computes Q(SW)/alpha; fold the scale or quantize this layer without one"
        )


def _group_tensors(state: LatticeState) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group ``scales`` and integer ``zeros``, laid out ``[n_groups, out]``."""
    scales = state.step_groups.t().contiguous()
    zeros = state.zero_groups.t().round().to(torch.int32).contiguous()
    return scales, zeros


def pack_layer_awq(state: LatticeState, name: str = "layer") -> PackedLayer:
    """AWQ GEMM layout: ``qweight [K, N/8]``, ``qzeros [K/g, N/8]``, ``scales [K/g, N]``."""
    out_features = state.codes.shape[0]
    _check_exportable(state, name, out_features)

    if state.bits != 4:
        raise ExportError(f"{name}: the AWQ format is 4-bit only, got {state.bits}")
    if out_features % 8 != 0:
        raise ExportError(f"{name}: out_features={out_features} must be a multiple of 8")

    codes = state.codes_int().t().contiguous()  # [K, N]
    scales, zeros = _group_tensors(state)

    return PackedLayer(
        {
            "qweight": pack_awq(codes, state.bits),
            "qzeros": pack_awq(zeros, state.bits),
            "scales": scales.to(torch.float16),
        }
    )


def pack_layer_gptq(state: LatticeState, name: str = "layer", v2: bool = True) -> PackedLayer:
    """GPTQ layout: ``qweight [K*b/32, N]``, ``qzeros [K/g, N*b/32]``, ``scales [K/g, N]``.

    ``v2`` writes the true zero point (``checkpoint_format: "gptq_v2"``).  The
    original format stores ``z - 1`` and has the kernel add one back, which
    cannot represent ``z = 0``; group-wise asymmetric quantization does produce
    ``z = 0``, so v2 is the default.
    """
    out_features = state.codes.shape[0]
    _check_exportable(state, name, out_features)

    if state.bits not in GPTQ_SUPPORTED_BITS:
        raise ExportError(f"{name}: GPTQ supports {GPTQ_SUPPORTED_BITS} bits, got {state.bits}")
    if state.in_features % 32 != 0:
        raise ExportError(f"{name}: in_features={state.in_features} must be a multiple of 32")

    codes = state.codes_int().t().contiguous()  # [K, N]
    scales, zeros = _group_tensors(state)

    if not v2:
        if (zeros == 0).any():
            raise ExportError(
                f"{name}: this layer has zero points equal to 0, which the "
                f"legacy GPTQ format cannot store; use v2"
            )
        zeros = zeros - 1

    group_index = (
        torch.arange(state.in_features, dtype=torch.int32) // state.group_size
    ).contiguous()

    return PackedLayer(
        {
            "qweight": pack_gptq(codes, state.bits),
            "qzeros": pack_gptq(zeros.t().contiguous(), state.bits).t().contiguous(),
            "scales": scales.to(torch.float16),
            "g_idx": group_index,
        }
    )


def pack_layer(state: LatticeState, fmt: str, name: str = "layer", **kwargs) -> PackedLayer:
    if fmt == "awq":
        return pack_layer_awq(state, name)
    if fmt == "gptq":
        return pack_layer_gptq(state, name, **kwargs)
    if fmt == "compressed-tensors":
        from clc.export.compressed_tensors_checkpoint import (
            pack_layer_compressed_tensors,
        )

        return pack_layer_compressed_tensors(state, name)
    raise ExportError(f"unknown format {fmt!r}; expected one of {SUPPORTED_FORMATS}")


def unpack_layer(
    packed: PackedLayer,
    fmt: str,
    bits: int,
    group_size: int,
    v2: bool = True,
    **kwargs,
):
    """Recover ``(codes, scales, zeros)`` from packed tensors, for verification."""
    tensors = packed.tensors
    if fmt == "awq":
        codes = unpack_awq(tensors["qweight"], bits)
        zeros = unpack_awq(tensors["qzeros"], bits)
    elif fmt == "gptq":
        codes = unpack_gptq(tensors["qweight"], bits)
        zeros = unpack_gptq(tensors["qzeros"].t().contiguous(), bits).t().contiguous()
        if not v2:
            zeros = zeros + 1
    elif fmt == "compressed-tensors":
        from clc.export.compressed_tensors_checkpoint import (
            unpack_layer_compressed_tensors,
        )

        return unpack_layer_compressed_tensors(
            tensors, bits=bits, symmetric=bool(kwargs.get("symmetric", False))
        )
    else:
        raise ExportError(f"unknown format {fmt!r}")
    return codes, tensors["scales"], zeros


def verify_layer(state: LatticeState, packed: PackedLayer, fmt: str, **kwargs) -> None:
    """Assert the packed tensors reproduce the lattice exactly.

    Codes and zero points must match bit-for-bit; scales are compared in float16
    because that is the width the checkpoint stores them at.
    """
    codes, scales, zeros = unpack_layer(
        packed, fmt, bits=state.bits, group_size=state.group_size, **kwargs
    )

    if fmt == "compressed-tensors":
        expected_codes = state.codes_int()
    else:
        expected_codes = state.codes_int().t()
    if not torch.equal(codes.to(torch.int32), expected_codes):
        mismatch = int((codes.to(torch.int32) != expected_codes).sum())
        raise ExportError(f"packed codes differ from the lattice in {mismatch} positions")

    if fmt == "compressed-tensors":
        expected_zeros = state.zero_groups.round().to(torch.int32)
    else:
        expected_zeros = state.zero_groups.t().round().to(torch.int32)
    if not torch.equal(zeros.to(torch.int32), expected_zeros):
        raise ExportError("packed zero points differ from the lattice")

    if fmt == "compressed-tensors":
        expected_scales = state.step_groups.to(torch.float16)
    else:
        expected_scales = state.step_groups.t().to(torch.float16)
    if not torch.equal(scales, expected_scales):
        raise ExportError("packed scales differ from the lattice")


def quantization_config(
    fmt: str,
    bits: int,
    group_size: int,
    skipped: Iterable[str],
    v2: bool = True,
    symmetric: bool = False,
    quantized_modules: Optional[Iterable[str]] = None,
) -> dict:
    """The ``quantization_config`` block vLLM and Transformers read from config.json."""
    skipped = sorted(set(skipped))
    if fmt == "awq":
        return {
            "quant_method": "awq",
            "bits": bits,
            "group_size": group_size,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": skipped or None,
        }
    if fmt == "gptq":
        return {
            "quant_method": "gptq",
            "bits": bits,
            "group_size": group_size,
            "desc_act": False,
            "sym": False,
            "true_sequential": True,
            "lm_head": False,
            "checkpoint_format": "gptq_v2" if v2 else "gptq",
            # vLLM uses this positive list to distinguish the packed linear
            # modules from ordinary fp16 tensors (notably lm_head).  Relying on
            # safetensors dtype inference is fragile for mixed checkpoints.
            "modules_in_block_to_quantize": sorted(set(quantized_modules or ())),
            "modules_to_not_convert": skipped or None,
        }
    if fmt == "compressed-tensors":
        # ``Linear`` is intentionally a class target rather than a list of
        # module paths.  vLLM can then apply its fused-QKV / fused-MLP mapping,
        # while the regex ignores (for example) lm_head, which is copied as a
        # normal tensor below.
        ignore = [
            item if item.startswith("re:") else f"re:.*{re.escape(item)}$"
            for item in skipped
        ]
        return {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": bits,
                        "type": "int",
                        "symmetric": symmetric,
                        "strategy": "group",
                        "group_size": group_size,
                        "dynamic": False,
                    },
                }
            },
            "ignore": ignore,
        }
    raise ExportError(f"unknown format {fmt!r}")


def _json_compatible(value: Any) -> Any:
    """Convert config values to the small JSON-safe subset used by HF configs."""
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def export_checkpoint(
    states: Dict[str, LatticeState],
    output_dir: str | Path,
    fmt: str = "awq",
    source_dir: Optional[str | Path] = None,
    extra_tensors: Optional[Dict[str, torch.Tensor]] = None,
    skipped_modules: Iterable[str] = ("lm_head",),
    verify: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
    **pack_kwargs,
) -> Path:
    """Write packed weights plus config and tokenizer files into ``output_dir``.

    Args:
        states: corrected lattices keyed by module path, e.g.
            ``"model.layers.0.self_attn.q_proj"``.
        source_dir: the original model directory, whose ``config.json`` and
            tokenizer files are copied and amended.
        extra_tensors: unquantized tensors to carry over verbatim (embeddings,
            norms, ``lm_head``).
        model_config: the loaded model's ``config.to_dict()``.  This is needed
            when ``source_dir`` is an HF model id rather than a local directory;
            exporting only the quantization block would otherwise create a
            checkpoint with no architecture metadata.
        verify: unpack every layer and check it round-trips before writing.
    """
    from safetensors.torch import save_file

    if fmt not in SUPPORTED_FORMATS:
        raise ExportError(f"unknown format {fmt!r}; expected one of {SUPPORTED_FORMATS}")
    if not states:
        raise ExportError("no quantized layers to export")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors: Dict[str, torch.Tensor] = {}
    bits = group_size = None
    symmetric = None
    for name, state in sorted(states.items()):
        packed = pack_layer(state, fmt, name=name, **pack_kwargs)
        if verify:
            verify_kwargs = dict(pack_kwargs)
            if fmt == "compressed-tensors":
                verify_kwargs["symmetric"] = state.symmetric
            verify_layer(state, packed, fmt, **verify_kwargs)
        tensors.update(packed.prefixed(name))
        if bits is None:
            bits, group_size, symmetric = state.bits, state.group_size, state.symmetric
        elif (state.bits, state.group_size, state.symmetric) != (
            bits,
            group_size,
            symmetric,
        ):
            raise ExportError(
                "all layers in one checkpoint must use the same bits, group size, "
                "and symmetry convention"
            )

    for name, tensor in (extra_tensors or {}).items():
        # ``state_dict()`` preserves tied embedding/lm_head storage.  The
        # low-level safetensors writer rejects aliases, while inference
        # backends are happy to receive two independent copies (the HF config
        # still records ``tie_word_embeddings``).  Clone only the unquantized
        # carry-through tensors so packed tensors stay zero-copy until save.
        tensors[name] = tensor.detach().clone().contiguous()

    save_file(
        {k: v.cpu().contiguous() for k, v in tensors.items()},
        output_dir / "model.safetensors",
        metadata={"format": "pt"},
    )

    config: Dict[str, Any] = {}
    if source_dir is not None:
        source_dir = Path(source_dir)
        config_path = source_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
        for filename in TOKENIZER_FILES:
            candidate = source_dir / filename
            if candidate.exists():
                shutil.copy2(candidate, output_dir / filename)

    if model_config is not None:
        config.update(_json_compatible(model_config))

    config["quantization_config"] = quantization_config(
        fmt,
        bits,
        group_size,
        skipped_modules,
        symmetric=bool(symmetric),
        quantized_modules=states.keys(),
        **pack_kwargs,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    from clc.deployment import write_deployment_manifest

    write_deployment_manifest(output_dir, fmt, bits, group_size)

    return output_dir
