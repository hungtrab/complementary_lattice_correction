"""Deployment-engine compatibility and checkpoint inspection helpers.

The packed checkpoint is the interoperability boundary. Several inference
engines consume the same AWQ, GPTQ, or compressed-tensors layout, but their
launch flags are different. This module keeps those differences in one small,
dependency-free registry so the CLI, generated manifest, and shell wrappers
cannot drift apart.

The registry intentionally distinguishes:

* direct: the engine can consume the exported Hugging Face checkpoint;
* conversion: the engine can use the quantization family after its own
  conversion/build step;
* unsupported: the exported directory is not a native input.

The bit matrix is conservative. It records the combinations documented by
the corresponding engine rather than assuming that every engine accepts every
integer width supported by the exporter.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

FORMATS = ("awq", "gptq", "compressed-tensors")


@dataclass(frozen=True)
class EngineSpec:
    """The public deployment contract for one inference engine."""

    name: str
    label: str
    direct_formats: Mapping[str, tuple[int, ...]]
    conversion_formats: Mapping[str, tuple[int, ...]]
    documentation: str
    notes: str

    def status(self, fmt: str, bits: int) -> str:
        if bits in self.direct_formats.get(fmt, ()):
            return "direct"
        if bits in self.conversion_formats.get(fmt, ()):
            return "conversion"
        return "unsupported"


ENGINE_SPECS: tuple[EngineSpec, ...] = (
    EngineSpec(
        name="vllm",
        label="vLLM",
        direct_formats={
            "awq": (4,),
            "gptq": (2, 3, 4, 8),
            "compressed-tensors": (4, 8),
        },
        conversion_formats={},
        documentation="https://docs.vllm.ai/en/latest/features/quantization/",
        notes="Use the format-specific --quantization flag; compressed-tensors is detected from config.json.",
    ),
    EngineSpec(
        name="sglang",
        label="SGLang",
        direct_formats={
            "awq": (4,),
            "gptq": (2, 3, 4, 8),
            "compressed-tensors": (4, 8),
        },
        conversion_formats={},
        documentation="https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/quantization.md",
        notes="Offline pre-quantized loading is preferred; backend and hardware can further restrict bit widths.",
    ),
    EngineSpec(
        name="tgi",
        label="Hugging Face TGI",
        direct_formats={"awq": (4,), "gptq": (4,)},
        conversion_formats={},
        documentation="https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization",
        notes="TGI uses --quantize rather than --quantization and currently documents AWQ/GPTQ 4-bit checkpoints.",
    ),
    EngineSpec(
        name="lmdeploy",
        label="LMDeploy",
        direct_formats={
            "awq": (4,),
            "gptq": (4,),
            "compressed-tensors": (4,),
        },
        conversion_formats={},
        documentation="https://lmdeploy.readthedocs.io/en/latest/api/cli.html",
        notes="TurboMind accepts grouped AWQ/GPTQ and compressed-tensors layouts; use --model-format explicitly when needed.",
    ),
    EngineSpec(
        name="transformers",
        label="Hugging Face Transformers",
        direct_formats={
            "awq": (4,),
            "gptq": (4,),
            "compressed-tensors": (4, 8),
        },
        conversion_formats={},
        documentation="https://huggingface.co/docs/transformers/main/quantization",
        notes="Install the backend required by the selected format (for example AutoAWQ or GPT-QModel).",
    ),
    EngineSpec(
        name="tensorrt-llm",
        label="TensorRT-LLM",
        direct_formats={},
        conversion_formats={"awq": (4,), "gptq": (4,)},
        documentation="https://nvidia.github.io/TensorRT-LLM/features/quantization.html",
        notes="Use the TensorRT-LLM/Model Optimizer conversion and engine-build workflow; the HF directory is not a final TRT engine.",
    ),
    EngineSpec(
        name="llama.cpp",
        label="llama.cpp / Ollama",
        direct_formats={},
        conversion_formats={},
        documentation="https://github.com/ggml-org/llama.cpp",
        notes="These runtimes require GGUF. A direct GGUF writer is intentionally not provided because re-quantization changes the CLC lattice.",
    ),
)

ENGINE_BY_NAME = {spec.name: spec for spec in ENGINE_SPECS}


@dataclass(frozen=True)
class CheckpointInfo:
    """Quantization metadata detected from a checkpoint directory."""

    path: str
    format: str
    bits: int
    group_size: Optional[int]
    model_type: Optional[str]

    def compatibility(self) -> dict[str, str]:
        return {
            spec.name: spec.status(self.format, self.bits)
            for spec in ENGINE_SPECS
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compatibility"] = self.compatibility()
        return payload


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_weight_config(quantization_config: Mapping[str, Any]) -> Mapping[str, Any]:
    groups = _as_mapping(quantization_config.get("config_groups"))
    for group in groups.values():
        weights = _as_mapping(_as_mapping(group).get("weights"))
        if weights:
            return weights
    return {}


def detect_quantization_config(config: Mapping[str, Any]) -> tuple[str, int, Optional[int]]:
    """Extract (format, bits, group_size) from a Hugging Face config."""

    quantization_config = _as_mapping(config.get("quantization_config"))
    method = str(
        quantization_config.get("quant_method")
        or quantization_config.get("quant_method_name")
        or ""
    ).replace("_", "-").lower()
    weights = _first_weight_config(quantization_config)

    if method in {"compressed-tensors", "compressedtensors"}:
        fmt = "compressed-tensors"
        bits_value = weights.get("num_bits", quantization_config.get("bits"))
        group_value = weights.get("group_size", quantization_config.get("group_size"))
    elif method in {"awq", "gptq"}:
        fmt = method
        bits_value = quantization_config.get("bits")
        group_value = quantization_config.get("group_size")
    else:
        raise ValueError(
            "config.json does not describe a supported packed format; expected "
            "quant_method awq, gptq, or compressed-tensors"
        )

    try:
        bits = int(bits_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantization_config is missing a numeric bit width") from exc
    group_size = None if group_value is None else int(group_value)
    return fmt, bits, group_size


def inspect_checkpoint(checkpoint: str | Path) -> CheckpointInfo:
    """Inspect a local exported checkpoint without importing an inference engine."""

    checkpoint_path = Path(checkpoint)
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    config = json.loads(config_path.read_text())
    fmt, bits, group_size = detect_quantization_config(config)
    return CheckpointInfo(
        path=str(checkpoint_path),
        format=fmt,
        bits=bits,
        group_size=group_size,
        model_type=config.get("model_type"),
    )


def _quoted_args(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def render_launch_command(
    engine: str,
    checkpoint: str | Path,
    fmt: str,
    bits: int,
    extra_args: Iterable[str] = (),
) -> str:
    """Render a copy-paste launch command for a directly supported engine."""

    if engine not in ENGINE_BY_NAME:
        raise ValueError(f"unknown engine {engine!r}; expected one of {sorted(ENGINE_BY_NAME)}")
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    spec = ENGINE_BY_NAME[engine]
    if spec.status(fmt, bits) != "direct":
        raise ValueError(
            f"{spec.label} does not directly consume {bits}-bit {fmt}; "
            f"status is {spec.status(fmt, bits)}"
        )

    model = str(checkpoint)
    if engine == "vllm":
        command = ["vllm", "serve", model]
        if fmt != "compressed-tensors":
            command += ["--quantization", fmt]
    elif engine == "sglang":
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--quantization",
            fmt,
        ]
    elif engine == "tgi":
        command = [
            "text-generation-launcher",
            "--model-id",
            model,
            "--quantize",
            fmt,
        ]
    elif engine == "lmdeploy":
        command = [
            "lmdeploy",
            "serve",
            "api_server",
            model,
            "--backend",
            "turbomind",
            "--model-format",
            fmt,
        ]
    else:
        raise ValueError(
            f"{spec.label} is a library or conversion target; use its documented "
            "Python/build workflow rather than a generic server command"
        )
    return _quoted_args((*command, *extra_args))


def deployment_manifest(fmt: str, bits: int, group_size: int) -> dict[str, Any]:
    """Build the portable sidecar metadata written next to packed weights."""

    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    engines = {}
    for spec in ENGINE_SPECS:
        status = spec.status(fmt, bits)
        engines[spec.name] = {
            "label": spec.label,
            "status": status,
            "documentation": spec.documentation,
            "notes": spec.notes,
        }
    return {
        "schema_version": 1,
        "producer": "complementary-lattice-correction",
        "format": fmt,
        "bits": bits,
        "group_size": group_size,
        "engines": engines,
    }


def write_deployment_manifest(
    output_dir: str | Path, fmt: str, bits: int, group_size: int
) -> Path:
    """Write deployment.json and return its path."""

    path = Path(output_dir) / "deployment.json"
    path.write_text(
        json.dumps(deployment_manifest(fmt, bits, group_size), indent=2) + "\n"
    )
    return path


def format_engine_table(fmt: Optional[str] = None, bits: Optional[int] = None) -> str:
    """Render the compatibility registry for a human-facing CLI."""

    selected = [
        spec
        for spec in ENGINE_SPECS
        if fmt is None
        or any(
            fmt in mapping
            for mapping in (spec.direct_formats, spec.conversion_formats)
        )
    ]
    lines = [
        f"{'Engine':<24} {'AWQ':<13} {'GPTQ':<15} {'compressed-tensors':<21}",
        "-" * 76,
    ]
    for spec in selected:
        statuses = []
        for candidate in FORMATS:
            supported = []
            for candidate_bits in sorted(
                set(spec.direct_formats.get(candidate, ()))
                | set(spec.conversion_formats.get(candidate, ()))
            ):
                if bits is None or bits == candidate_bits:
                    status = spec.status(candidate, candidate_bits)
                    supported.append(f"{candidate_bits}:{status[0]}")
            statuses.append(",".join(supported) or "-")
        lines.append(
            f"{spec.label:<24} {statuses[0]:<13} "
            f"{statuses[1]:<15} {statuses[2]:<21}"
        )
    lines.append("")
    lines.append("Legend: d = direct checkpoint load, c = engine-specific conversion/build.")
    return "\n".join(lines)


__all__ = [
    "ENGINE_BY_NAME",
    "ENGINE_SPECS",
    "FORMATS",
    "CheckpointInfo",
    "EngineSpec",
    "detect_quantization_config",
    "deployment_manifest",
    "format_engine_table",
    "inspect_checkpoint",
    "render_launch_command",
    "write_deployment_manifest",
]
