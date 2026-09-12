"""Convert Hugging Face checkpoints to GGUF through the official llama.cpp tools.

GGUF is deliberately a separate export target.  AWQ/GPTQ/compressed-tensors
checkpoints preserve CLC's integer lattice, while a llama.cpp quantized GGUF
usually introduces a second quantization step with a different lattice.  This
module therefore makes that boundary explicit and records it in a small
``*.gguf.clc.json`` provenance sidecar.

The converter accepts both ordinary Hugging Face directories containing
``model.safetensors`` and packed checkpoints written by this repository.  For a
packed CLC checkpoint, the integer tensors are dequantized into a temporary,
standard Hugging Face safetensors directory before llama.cpp is invoked.  The
temporary directory is removed after conversion; no intermediate model is
left in the repository.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from clc.deployment import detect_quantization_config
from clc.export.checkpoint import (
    TOKENIZER_FILES,
    ExportError,
    PackedLayer,
    unpack_layer,
)

GGUF_OUTTYPES = ("auto", "f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0")
"""Current high-level output types accepted by llama.cpp's HF converter."""

UNQUANTIZED_TYPES = {"NONE", "F32", "F16", "BF16"}
QUANT_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")


class GGUFConversionError(RuntimeError):
    """The llama.cpp conversion target cannot be built."""


def _command_text(command: Iterable[str]) -> str:
    return shlex.join(str(item) for item in command)


def _local_source(source: str | Path) -> Optional[Path]:
    """Return a local model directory, or ``None`` for an HF Hub id."""

    candidate = Path(source).expanduser()
    if not candidate.exists():
        return None
    if candidate.is_file() and candidate.suffix == ".safetensors":
        candidate = candidate.parent
    if not candidate.is_dir():
        raise GGUFConversionError(
            f"GGUF source must be a Hugging Face directory, got {candidate}"
        )
    return candidate.resolve()


def _read_config(source: Path) -> dict[str, Any]:
    path = source / "config.json"
    if not path.exists():
        raise GGUFConversionError(f"local GGUF source is missing {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GGUFConversionError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GGUFConversionError(f"expected an object in {path}")
    return value


def _safetensor_names(source: Path) -> list[str]:
    """Read tensor names without materializing a potentially huge checkpoint."""

    single = source / "model.safetensors"
    if single.exists():
        try:
            from safetensors import safe_open

            with safe_open(single, framework="pt", device="cpu") as handle:
                return list(handle.keys())
        except (ImportError, OSError, RuntimeError) as exc:
            raise GGUFConversionError(f"could not inspect {single}: {exc}") from exc

    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            names = index.get("weight_map", {}).keys()
            return [str(name) for name in names]
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise GGUFConversionError(f"could not inspect {index_path}: {exc}") from exc

    return []


def _ct_is_symmetric(config: dict[str, Any]) -> bool:
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        return False
    groups = quantization_config.get("config_groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            weights = group.get("weights")
            if isinstance(weights, dict) and "symmetric" in weights:
                return bool(weights["symmetric"])
    return bool(quantization_config.get("symmetric", False))


def _packed_source_info(
    source: Path,
) -> Optional[tuple[str, int, int, bool, bool]]:
    """Detect a packed CLC/HF checkpoint without loading its weight data."""

    config = _read_config(source)
    names = _safetensor_names(source)
    if not names:
        return None

    try:
        fmt, bits, group_size = detect_quantization_config(config)
    except ValueError:
        return None
    if group_size is None or group_size <= 0:
        raise GGUFConversionError(
            f"packed {fmt} source must declare a positive group_size; got {group_size!r}"
        )

    if fmt in {"awq", "gptq"}:
        packed = any(name.endswith(".qweight") for name in names)
    elif fmt == "compressed-tensors":
        packed = any(name.endswith(".weight_packed") for name in names)
    else:
        packed = False
    if not packed:
        return None
    quantization_config = config.get("quantization_config")
    checkpoint_format = (
        quantization_config.get("checkpoint_format")
        if isinstance(quantization_config, dict)
        else None
    )
    # CLC's default is GPTQ v2.  Honor the explicit legacy name when a user
    # asks to convert an older export that stored z - 1 on disk.
    gptq_v2 = not (
        fmt == "gptq"
        and str(checkpoint_format or "gptq_v2").lower() in {"gptq", "gptq_v1", "gptq-v1"}
    )
    return fmt, bits, group_size, _ct_is_symmetric(config), gptq_v2


def _load_safetensors(source: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    single = source / "model.safetensors"
    if single.exists():
        return dict(load_file(single, device="cpu"))

    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            shards = sorted(set(index.get("weight_map", {}).values()))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise GGUFConversionError(f"could not read {index_path}: {exc}") from exc
        state: Dict[str, torch.Tensor] = {}
        for shard in shards:
            shard_path = source / str(shard)
            if not shard_path.exists():
                raise GGUFConversionError(f"safetensors shard is missing: {shard_path}")
            state.update(load_file(shard_path, device="cpu"))
        return state

    raise GGUFConversionError(
        f"{source} has no model.safetensors or model.safetensors.index.json; "
        "GGUF conversion currently requires a safetensors source"
    )


def _dequantize_layer(
    packed: PackedLayer,
    fmt: str,
    bits: int,
    group_size: int,
    symmetric: bool,
    v2: bool = True,
) -> torch.Tensor:
    """Materialize one packed layer as a standard ``[out, in]`` float16 weight."""

    try:
        codes, scales, zeros = unpack_layer(
            packed,
            fmt,
            bits=bits,
            group_size=group_size,
            v2=v2,
            symmetric=symmetric,
        )
    except (ExportError, KeyError, RuntimeError, ValueError) as exc:
        raise GGUFConversionError(f"could not unpack {fmt} layer: {exc}") from exc

    if fmt in {"awq", "gptq"}:
        # AWQ/GPTQ verification normalizes these tensors to [in, out] and
        # [groups, out], whereas a PyTorch Linear stores [out, in].
        code_matrix = codes.to(torch.float32).t().contiguous()
        scale_matrix = scales.to(torch.float32).t().contiguous()
        zero_matrix = zeros.to(torch.float32).t().contiguous()
    else:
        code_matrix = codes.to(torch.float32).contiguous()
        scale_matrix = scales.to(torch.float32).contiguous()
        zero_matrix = zeros.to(torch.float32).contiguous()

    if scale_matrix.ndim == 1:
        scale_matrix = scale_matrix.unsqueeze(1)
    if zero_matrix.ndim == 1:
        zero_matrix = zero_matrix.unsqueeze(1)
    if scale_matrix.shape != zero_matrix.shape:
        raise GGUFConversionError(
            f"packed layer has mismatched scale/zero shapes: "
            f"{tuple(scale_matrix.shape)} vs {tuple(zero_matrix.shape)}"
        )
    if scale_matrix.shape[0] != code_matrix.shape[0]:
        raise GGUFConversionError(
            f"packed layer has {code_matrix.shape[0]} output rows but "
            f"scales describe {scale_matrix.shape[0]}"
        )

    expanded_scales = scale_matrix.repeat_interleave(group_size, dim=1)
    expanded_zeros = zero_matrix.repeat_interleave(group_size, dim=1)
    if expanded_scales.shape[1] < code_matrix.shape[1]:
        raise GGUFConversionError(
            f"group_size={group_size} does not cover packed input width "
            f"{code_matrix.shape[1]}"
        )
    expanded_scales = expanded_scales[:, : code_matrix.shape[1]]
    expanded_zeros = expanded_zeros[:, : code_matrix.shape[1]]
    return ((code_matrix - expanded_zeros) * expanded_scales).to(torch.float16)


def _copy_support_files(source: Path, target: Path) -> None:
    """Copy tokenizer/config-adjacent files needed by llama.cpp's converter."""

    extra_names = {
        "added_tokens.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
    }
    for candidate in source.iterdir():
        if not candidate.is_file():
            continue
        if candidate.name in TOKENIZER_FILES or candidate.name in extra_names:
            shutil.copy2(candidate, target / candidate.name)


def _stage_packed_source(
    source: Path,
    target: Path,
    info: tuple[str, int, int, bool, bool],
) -> None:
    """Turn this project's packed tensors into a temporary HF safetensors model."""

    fmt, bits, group_size, symmetric, gptq_v2 = info
    state = _load_safetensors(source)
    target.mkdir(parents=True, exist_ok=True)

    if fmt in {"awq", "gptq"}:
        prefixes = sorted(
            {
                name[: -len(".qweight")]
                for name in state
                if name.endswith(".qweight")
            }
        )
        required_suffixes = ("qweight", "qzeros", "scales")
    else:
        prefixes = sorted(
            {
                name[: -len(".weight_packed")]
                for name in state
                if name.endswith(".weight_packed")
            }
        )
        required_suffixes = ("weight_packed", "weight_scale", "weight_shape")

    if not prefixes:
        raise GGUFConversionError(f"{source} declares {fmt} but contains no packed layers")

    converted: Dict[str, torch.Tensor] = {}
    packed_keys: set[str] = set()
    for prefix in prefixes:
        missing = [
            f"{prefix}.{suffix}"
            for suffix in required_suffixes
            if f"{prefix}.{suffix}" not in state
        ]
        if missing:
            raise GGUFConversionError(
                f"packed layer {prefix} is missing tensor(s): {', '.join(missing)}"
            )
        tensors = {
            suffix: state[f"{prefix}.{suffix}"]
            for suffix in required_suffixes
        }
        if f"{prefix}.g_idx" in state:
            tensors["g_idx"] = state[f"{prefix}.g_idx"]
        if fmt == "compressed-tensors" and f"{prefix}.weight_zero_point" in state:
            tensors["weight_zero_point"] = state[f"{prefix}.weight_zero_point"]

        converted[f"{prefix}.weight"] = _dequantize_layer(
            PackedLayer(tensors),
            fmt,
            bits,
            group_size,
            symmetric,
            v2=gptq_v2,
        )
        packed_keys.update(f"{prefix}.{suffix}" for suffix in required_suffixes)
        packed_keys.update(
            name
            for name in (
                f"{prefix}.g_idx",
                f"{prefix}.weight_zero_point",
            )
            if name in state
        )

    # Carry biases, embeddings, norms, lm_head, and any auxiliary tensors over
    # unchanged.  Cloning also removes tied-storage aliases rejected by
    # safetensors.save_file.
    for name, tensor in state.items():
        if name in packed_keys or name in converted:
            continue
        converted[name] = tensor.detach().clone().contiguous()

    from safetensors.torch import save_file

    save_file(
        {name: tensor.cpu().contiguous() for name, tensor in converted.items()},
        target / "model.safetensors",
        metadata={"format": "pt", "source": "clc-packed-dequantized"},
    )

    config = _read_config(source)
    # llama.cpp's HF converter expects an ordinary architecture config.  The
    # CLC quantization block describes the source storage and must not make the
    # temporary model look like an AWQ/GPTQ model to Transformers.
    config.pop("quantization_config", None)
    config.pop("deployment", None)
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    _copy_support_files(source, target)


def _resolve_converter(
    llama_cpp: str | Path | None,
    convert_script: str | Path | None,
) -> Path:
    if convert_script is not None:
        path = Path(convert_script).expanduser()
        if not path.exists():
            raise GGUFConversionError(f"llama.cpp converter script not found: {path}")
        return path.resolve()

    root_value = llama_cpp or os.environ.get("LLAMA_CPP_PATH")
    candidates = []
    if root_value:
        root = Path(root_value).expanduser()
        candidates.append(root if root.is_file() else root / "convert_hf_to_gguf.py")
    found = shutil.which("convert_hf_to_gguf.py")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "$PATH"
    raise GGUFConversionError(
        "could not find llama.cpp's convert_hf_to_gguf.py; pass "
        f"--llama-cpp /path/to/llama.cpp or --convert-script PATH (searched {searched})"
    )


def _resolve_quantizer(
    llama_cpp: str | Path | None,
    quantize_binary: str | Path | None,
    converter: Path | None = None,
) -> Path:
    if quantize_binary is not None:
        path = Path(quantize_binary).expanduser()
        if not path.exists():
            raise GGUFConversionError(f"llama-quantize binary not found: {path}")
        return path.resolve()

    root_value = llama_cpp or os.environ.get("LLAMA_CPP_PATH")
    candidates = []
    roots = []
    if root_value:
        root = Path(root_value).expanduser()
        roots.append(root.parent if root.is_file() else root)
    elif converter is not None:
        # An explicit /path/to/llama.cpp/convert_hf_to_gguf.py is enough to
        # infer the usual sibling build/bin/llama-quantize location.
        roots.append(converter.parent)
    for root in roots:
        candidates.extend(
            [
                root / "build" / "bin" / "llama-quantize",
                root / "build" / "bin" / "Release" / "llama-quantize",
                root / "bin" / "llama-quantize",
            ]
        )
    found = shutil.which("llama-quantize")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "$PATH"
    raise GGUFConversionError(
        "could not find llama.cpp's llama-quantize binary; pass "
        f"--llama-cpp /path/to/llama.cpp or --quantize-bin PATH (searched {searched})"
    )


def _run(command: list[str], cwd: Path, label: str) -> None:
    try:
        subprocess.run(command, check=True, cwd=str(cwd))
    except FileNotFoundError as exc:
        raise GGUFConversionError(
            f"could not start {label}: {exc}. Command: {_command_text(command)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise GGUFConversionError(
            f"{label} failed with exit code {exc.returncode}. "
            f"Command: {_command_text(command)}"
        ) from exc


def _normalise_output(output: str | Path) -> Path:
    path = Path(output).expanduser()
    if path.suffix.lower() != ".gguf":
        path = path.with_name(path.name + ".gguf")
    return path.resolve()


def convert_to_gguf(
    source: str | Path,
    output: str | Path,
    *,
    llama_cpp: str | Path | None = None,
    convert_script: str | Path | None = None,
    quantize_binary: str | Path | None = None,
    outtype: str = "f16",
    quant_type: str = "Q4_K_M",
    threads: int | None = None,
    allow_requantize: bool = False,
    leave_output_tensor: bool = False,
    pure: bool = False,
    imatrix: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert an HF safetensors model to GGUF and optionally quantize it.

    Args:
        source: Local HF/CLC checkpoint directory or an HF Hub model id.  A
            local CLC AWQ/GPTQ/compressed-tensors directory is dequantized to a
            temporary ordinary HF directory first.
        output: Destination ``.gguf`` file.  The suffix is appended when it is
            omitted.
        llama_cpp: llama.cpp source/build root.  ``LLAMA_CPP_PATH`` is used as
            a fallback.
        convert_script: Explicit path to ``convert_hf_to_gguf.py``.
        quantize_binary: Explicit path to ``llama-quantize``.  It is required
            unless ``quant_type`` is ``NONE``, ``F16``, ``BF16``, or ``F32``.
        outtype: High-precision intermediate type understood by the official
            converter.
        quant_type: Final llama.cpp quantization type, for example
            ``Q4_K_M`` or ``Q8_0``.  Use ``NONE`` for an unquantized GGUF.
        threads: Optional positional thread count for ``llama-quantize``.

    The final GGUF is a runtime-compatible conversion artifact, not a
    lattice-preserving AWQ/GPTQ export.  The provenance sidecar says whether a
    second quantization step was performed.
    """

    outtype = str(outtype).lower()
    if outtype not in GGUF_OUTTYPES:
        raise GGUFConversionError(
            f"unsupported llama.cpp converter outtype {outtype!r}; "
            f"expected one of {GGUF_OUTTYPES}"
        )
    quant_type = str(quant_type).strip()
    if not quant_type or not QUANT_TYPE_RE.fullmatch(quant_type):
        raise GGUFConversionError(
            f"invalid llama.cpp quantization type {quant_type!r}; use a name such as Q4_K_M"
        )
    quant_type_upper = quant_type.upper()
    if quant_type_upper == "AUTO":
        raise GGUFConversionError("quant_type=auto is not a llama-quantize output type; use NONE or a concrete type")
    if threads is not None and threads <= 0:
        raise GGUFConversionError(f"threads must be positive, got {threads}")
    if imatrix is not None and not Path(imatrix).expanduser().exists():
        raise GGUFConversionError(f"importance matrix not found: {imatrix}")

    output_path = _normalise_output(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise GGUFConversionError(
            f"refusing to overwrite existing GGUF: {output_path}; pass overwrite=True/--force"
        )

    converter = _resolve_converter(llama_cpp, convert_script)
    source_path = _local_source(source)
    packed_info = _packed_source_info(source_path) if source_path is not None else None
    source_format = packed_info[0] if packed_info is not None else "hf-safetensors"
    stage_outtype = outtype
    if quant_type_upper in {"F32", "F16", "BF16"}:
        stage_outtype = quant_type_upper.lower()
    requantize = quant_type_upper not in UNQUANTIZED_TYPES

    quantizer = (
        _resolve_quantizer(llama_cpp, quantize_binary, converter=converter)
        if requantize
        else None
    )
    conversion_command: list[str]
    quantization_command: Optional[list[str]] = None

    with tempfile.TemporaryDirectory(
        prefix=".clc-gguf-",
        dir=str(output_path.parent),
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        conversion_source: str | Path = source
        staged = packed_info is not None
        if staged:
            staged_source = temporary_root / "hf-dequantized"
            assert source_path is not None
            _stage_packed_source(source_path, staged_source, packed_info)
            conversion_source = staged_source
        elif source_path is not None:
            conversion_source = source_path

        intermediate = (
            temporary_root / "model-f16.gguf" if requantize else output_path
        )
        conversion_command = [
            sys.executable,
            str(converter),
            "--outfile",
            str(intermediate),
            "--outtype",
            stage_outtype,
        ]
        if source_path is None:
            conversion_command.extend(["--remote", str(conversion_source)])
        else:
            conversion_command.append(str(conversion_source))
        _run(conversion_command, converter.parent, "convert_hf_to_gguf.py")

        if requantize:
            assert quantizer is not None
            quantization_command = [str(quantizer)]
            if allow_requantize:
                quantization_command.append("--allow-requantize")
            if leave_output_tensor:
                quantization_command.append("--leave-output-tensor")
            if pure:
                quantization_command.append("--pure")
            if imatrix is not None:
                quantization_command.extend(["--imatrix", str(Path(imatrix).expanduser().resolve())])
            quantization_command.extend(
                [str(intermediate), str(output_path), quant_type]
            )
            if threads is not None:
                quantization_command.append(str(threads))
            _run(quantization_command, quantizer.parent, "llama-quantize")

    manifest_path = Path(str(output_path) + ".clc.json")
    manifest = {
        "schema_version": 1,
        "producer": "complementary-lattice-correction",
        "target": "gguf",
        "output": str(output_path),
        "source": str(source),
        "source_format": source_format,
        "staged_dequantization": bool(packed_info),
        "lattice_preserving": False,
        "requantized": requantize,
        "converter": str(converter),
        "outtype": stage_outtype,
        "quant_type": quant_type,
        "quantization_warning": (
            "GGUF quantization is a downstream dequantize/re-quantize conversion; "
            "it is not bit-exact with the source CLC AWQ/GPTQ/compressed-tensors lattice."
        ),
        "commands": {
            "convert": _command_text(conversion_command),
            "quantize": (
                _command_text(quantization_command)
                if quantization_command is not None
                else None
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path


__all__ = [
    "GGUFConversionError",
    "GGUF_OUTTYPES",
    "convert_to_gguf",
]
