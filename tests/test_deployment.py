"""Deployment registry and checkpoint-inspection tests."""

import json

import pytest

from clc.deployment import (
    detect_quantization_config,
    deployment_manifest,
    format_engine_table,
    inspect_checkpoint,
    render_launch_command,
    write_deployment_manifest,
)


def test_detects_awq_metadata():
    assert detect_quantization_config(
        {
            "model_type": "llama",
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
            },
        }
    ) == ("awq", 4, 128)


def test_detects_compressed_tensors_nested_metadata():
    assert detect_quantization_config(
        {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 8,
                            "group_size": 64,
                        }
                    }
                },
            }
        }
    ) == ("compressed-tensors", 8, 64)


def test_inspect_checkpoint_reports_engine_compatibility(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral",
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 3,
                    "group_size": 128,
                },
            }
        )
    )

    info = inspect_checkpoint(tmp_path)

    assert info.format == "gptq"
    assert info.bits == 3
    assert info.model_type == "mistral"
    assert info.compatibility()["vllm"] == "direct"
    assert info.compatibility()["sglang"] == "direct"
    assert info.compatibility()["tgi"] == "unsupported"
    assert info.compatibility()["tensorrt-llm"] == "unsupported"


def test_rendered_commands_use_engine_specific_flags():
    assert render_launch_command("vllm", "/tmp/model", "awq", 4) == (
        "vllm serve /tmp/model --quantization awq"
    )
    assert render_launch_command("sglang", "/tmp/model", "gptq", 3) == (
        "python -m sglang.launch_server --model-path /tmp/model "
        "--quantization gptq"
    )
    assert render_launch_command("tgi", "/tmp/model", "awq", 4) == (
        "text-generation-launcher --model-id /tmp/model --quantize awq"
    )
    assert render_launch_command("lmdeploy", "/tmp/model", "awq", 4) == (
        "lmdeploy serve api_server /tmp/model --backend turbomind "
        "--model-format awq"
    )


def test_rendered_commands_quote_paths_and_extra_args():
    command = render_launch_command(
        "vllm",
        "/tmp/model with spaces",
        "compressed-tensors",
        4,
        extra_args=["--tensor-parallel-size", "2"],
    )
    assert command == (
        "vllm serve '/tmp/model with spaces' --tensor-parallel-size 2"
    )


def test_non_direct_engines_are_not_presented_as_direct():
    with pytest.raises(ValueError, match="does not directly consume"):
        render_launch_command("tgi", "/tmp/model", "compressed-tensors", 4)
    with pytest.raises(ValueError, match="library or conversion target"):
        render_launch_command("transformers", "/tmp/model", "awq", 4)


def test_manifest_contains_direct_and_conversion_statuses():
    manifest = deployment_manifest("awq", 4, 128)

    assert manifest["producer"] == "complementary-lattice-correction"
    assert manifest["engines"]["vllm"]["status"] == "direct"
    assert manifest["engines"]["tgi"]["status"] == "direct"
    assert manifest["engines"]["tensorrt-llm"]["status"] == "conversion"
    assert manifest["engines"]["llama.cpp"]["status"] == "unsupported"


def test_manifest_is_written_next_to_checkpoint(tmp_path):
    path = write_deployment_manifest(tmp_path, "gptq", 4, 128)
    payload = json.loads(path.read_text())

    assert path.name == "deployment.json"
    assert payload["format"] == "gptq"
    assert payload["bits"] == 4


def test_engine_table_is_human_readable():
    table = format_engine_table(fmt="awq", bits=4)

    assert "vLLM" in table
    assert "Hugging Face TGI" in table
    assert "d = direct checkpoint load" in table
    assert "4:d" in table
