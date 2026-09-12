"""GGUF conversion is explicit, testable, and honest about re-quantization."""

import json
import sys
from pathlib import Path

import pytest
import torch

from clc.export.checkpoint import export_checkpoint, pack_layer
from clc.export.gguf import (
    GGUFConversionError,
    _dequantize_layer,
    convert_to_gguf,
)
from clc.lattice import LatticeState


def _state(fmt: str):
    state = LatticeState.from_weight(
        torch.randn(32, 64), bits=4, group_size=32, symmetric=False
    )
    return state, pack_layer(state, fmt, name="layer")


@pytest.mark.parametrize("fmt", ["awq", "gptq", "compressed-tensors"])
def test_packed_layers_dequantize_back_to_the_exported_float_weight(fmt):
    state, packed = _state(fmt)
    recovered = _dequantize_layer(
        packed,
        fmt,
        bits=state.bits,
        group_size=state.group_size,
        symmetric=state.symmetric,
    )
    assert torch.equal(recovered, state.weight().to(torch.float16))


def _fake_llama_cpp(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "llama.cpp"
    binary_dir = root / "build" / "bin"
    binary_dir.mkdir(parents=True)
    converter = root / "convert_hf_to_gguf.py"
    converter.write_text("# replaced by the subprocess mock in this test\n")
    quantizer = binary_dir / "llama-quantize"
    quantizer.write_text("placeholder")
    return converter, quantizer


def test_convert_gguf_runs_official_two_stage_commands(tmp_path, monkeypatch):
    source = tmp_path / "hf-model"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    from safetensors.torch import save_file

    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.randn(8, 64)},
        source / "model.safetensors",
    )
    converter, quantizer = _fake_llama_cpp(tmp_path)
    calls = []

    def fake_run(command, check, cwd):
        calls.append((list(command), cwd))
        if str(converter) in command:
            output = Path(command[command.index("--outfile") + 1])
        else:
            output = Path(command[-3])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()

    monkeypatch.setattr("clc.export.gguf.subprocess.run", fake_run)
    output = convert_to_gguf(
        source,
        tmp_path / "out" / "model-q4.gguf",
        convert_script=converter,
        quant_type="Q4_K_M",
        threads=4,
    )

    assert output.exists()
    assert len(calls) == 2
    assert calls[0][0][:2] == [sys.executable, str(converter)]
    assert "--outtype" in calls[0][0]
    assert calls[1][0] == [str(quantizer), calls[1][0][1], str(output), "Q4_K_M", "4"]
    manifest = json.loads(Path(str(output) + ".clc.json").read_text())
    assert manifest["target"] == "gguf"
    assert manifest["requantized"] is True
    assert manifest["lattice_preserving"] is False


def test_convert_gguf_dequantizes_a_clc_packed_source_before_conversion(tmp_path, monkeypatch):
    state, _ = _state("awq")
    source = export_checkpoint(
        {"model.layers.0.mlp.up_proj": state},
        tmp_path / "clc-awq",
        fmt="awq",
        extra_tensors={"lm_head.weight": torch.randn(16, 64)},
        model_config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    )
    converter, _ = _fake_llama_cpp(tmp_path)
    observed_source = {}

    def fake_run(command, check, cwd):
        if str(converter) in command:
            source_arg = Path(command[-1])
            observed_source["path"] = source_arg
            assert source_arg != source
            assert (source_arg / "config.json").exists()
            assert (source_arg / "model.safetensors").exists()
            from safetensors.torch import load_file

            staged = load_file(source_arg / "model.safetensors")
            assert "model.layers.0.mlp.up_proj.weight" in staged
            assert "model.layers.0.mlp.up_proj.qweight" not in staged
            Path(command[command.index("--outfile") + 1]).touch()

    monkeypatch.setattr("clc.export.gguf.subprocess.run", fake_run)
    output = convert_to_gguf(
        source,
        tmp_path / "out" / "model-f16.gguf",
        convert_script=converter,
        quant_type="NONE",
    )

    assert output.exists()
    manifest = json.loads(Path(str(output) + ".clc.json").read_text())
    assert manifest["source_format"] == "awq"
    assert manifest["staged_dequantization"] is True
    assert manifest["requantized"] is False
    assert observed_source["path"].name == "hf-dequantized"


def test_convert_gguf_refuses_to_overwrite_and_requires_converter(tmp_path):
    with pytest.raises(GGUFConversionError, match="convert_hf_to_gguf.py"):
        convert_to_gguf(
            "org/model",
            tmp_path / "model.gguf",
            quant_type="NONE",
        )

    output = tmp_path / "existing.gguf"
    output.touch()
    converter, _ = _fake_llama_cpp(tmp_path)
    with pytest.raises(GGUFConversionError, match="overwrite"):
        convert_to_gguf(
            "org/model",
            output,
            convert_script=converter,
            quant_type="NONE",
        )
