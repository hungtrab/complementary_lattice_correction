"""Convert the old fake-quantized CLC/AWQ directory to a packed AWQ model.

Older experiments saved a full-size ``model.safetensors`` whose linear weights
had already been replaced by dequantized values.  That file contains no integer
codes, scales, or zero points, so the original lattice cannot be recovered
bit-for-bit.  This converter deliberately makes that limitation explicit: it
projects each eligible linear weight onto the new float16-step AWQ lattice and
then packs that lattice with the same writer used by fresh runs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

import torch

from clc.export.checkpoint import TOKENIZER_FILES, ExportError, pack_layer_awq, quantization_config, verify_layer
from clc.lattice import LatticeState


def _load_state_dict(source_dir: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    single = source_dir / "model.safetensors"
    if single.exists():
        return dict(load_file(single, device="cpu"))

    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        state: Dict[str, torch.Tensor] = {}
        for shard in sorted(set(index.get("weight_map", {}).values())):
            state.update(load_file(source_dir / shard, device="cpu"))
        return state

    legacy_bin = source_dir / "pytorch_model.bin"
    if legacy_bin.exists():
        # This is a user-selected local conversion of an old checkpoint.  The
        # explicit weights_only=True path avoids executing arbitrary objects for
        # the common state-dict-only case.
        try:
            loaded = torch.load(legacy_bin, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(legacy_bin, map_location="cpu")
        if not isinstance(loaded, dict):
            raise ExportError(f"expected a state dict in {legacy_bin}")
        return dict(loaded)

    raise ExportError(
        f"{source_dir} has no model.safetensors, model.safetensors.index.json, "
        "or pytorch_model.bin"
    )


def _looks_like_linear_weight(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.ndim != 2:
        return False
    lowered = name.lower()
    excluded = (
        "embed",
        "embedding",
        "lm_head",
        "output_projection",
        "shared.weight",
        "wte.weight",
        "tok_embeddings",
    )
    return not any(token in lowered for token in excluded)


def convert_legacy_awq_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    group_size: int = 128,
    bits: int = 4,
    verify: bool = True,
    strict: bool = False,
    skipped_modules: Iterable[str] = ("lm_head",),
) -> Path:
    """Repack a legacy fake-quantized model as an AWQ checkpoint.

    ``strict=False`` skips 2-D tensors that do not satisfy AWQ packing
    constraints (embeddings, unusual projections, and heads); they remain as
    ordinary tensors.  Set ``strict=True`` when auditing a checkpoint and an
    unsupported linear-looking tensor should be treated as an error.
    """

    if bits != 4:
        raise ExportError("legacy AWQ conversion is defined only for 4-bit weights")
    if group_size <= 0:
        raise ExportError(f"group_size must be positive, got {group_size}")

    source = Path(source_dir)
    output = Path(output_dir)
    if source.resolve() == output.resolve():
        raise ExportError("source and output directories must be different")
    source_config_path = source / "config.json"
    if not source_config_path.exists():
        raise ExportError("legacy conversion requires the source config.json")
    config = json.loads(source_config_path.read_text())
    state_dict = _load_state_dict(source)

    states: Dict[str, LatticeState] = {}
    extra: Dict[str, torch.Tensor] = {}
    skipped = set(skipped_modules)
    skipped_reasons: Dict[str, str] = {}

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not _looks_like_linear_weight(name, tensor):
            extra[name] = tensor
            continue
        module_name = name[: -len(".weight")]
        if module_name in skipped or any(module_name.endswith(item) for item in skipped):
            extra[name] = tensor
            continue
        if tensor.shape[1] % group_size != 0:
            message = (
                f"{name}: input dimension {tensor.shape[1]} is not divisible by "
                f"group_size={group_size}"
            )
        elif tensor.shape[0] % 8 != 0:
            message = f"{name}: out_features={tensor.shape[0]} is not a multiple of 8"
        else:
            message = ""

        if message:
            if strict:
                raise ExportError(message)
            skipped_reasons[name] = message
            extra[name] = tensor
            continue

        state = LatticeState.from_weight(
            tensor,
            bits=bits,
            group_size=group_size,
            symmetric=False,
        )
        states[module_name] = state

    if not states:
        raise ExportError("legacy checkpoint contains no AWQ-packable linear weights")

    packed_tensors: Dict[str, torch.Tensor] = {}
    for name, state in sorted(states.items()):
        packed = pack_layer_awq(state, name=name)
        if verify:
            verify_layer(state, packed, "awq")
        packed_tensors.update(packed.prefixed(name))
    packed_tensors.update(
        {name: tensor.detach().clone().contiguous() for name, tensor in extra.items()}
    )

    output.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        {name: tensor.cpu().contiguous() for name, tensor in packed_tensors.items()},
        output / "model.safetensors",
        metadata={"format": "pt"},
    )

    for filename in TOKENIZER_FILES:
        candidate = source / filename
        if candidate.exists():
            shutil.copy2(candidate, output / filename)

    modules_to_skip = set(skipped)
    modules_to_skip.update(
        name[: -len(".weight")] for name in skipped_reasons
    )
    config["quantization_config"] = quantization_config(
        "awq", bits, group_size, modules_to_skip, quantized_modules=states.keys()
    )
    config["clc_legacy_conversion"] = {
        "source": str(source),
        "reprojected": True,
        "skipped_tensors": skipped_reasons,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    return output


__all__ = ["convert_legacy_awq_checkpoint"]
