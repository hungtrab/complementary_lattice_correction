"""Interoperability checks for the compressed-tensors WNA16 writer."""

import json

import pytest
import torch

from clc.export.checkpoint import export_checkpoint, pack_layer, verify_layer
from clc.export.compressed_tensors_checkpoint import pack_signed
from clc.export.legacy_convert import convert_legacy_awq_checkpoint
from clc.lattice import LatticeState


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_compressed_tensors_round_trip_matches_lattice(bits, symmetric):
    torch.manual_seed(bits + int(symmetric))
    state = LatticeState.from_weight(
        torch.randn(16, 64), bits=bits, group_size=32, symmetric=symmetric
    )
    packed = pack_layer(state, "compressed-tensors")
    verify_layer(state, packed, "compressed-tensors", symmetric=symmetric)


@pytest.mark.parametrize("bits", [4, 8])
def test_compressed_tensors_packing_matches_reference_library(bits):
    reference = pytest.importorskip(
        "compressed_tensors.compressors.quantized_compressors.pack_quantized"
    )
    torch.manual_seed(bits)
    values = torch.randint(
        -(1 << (bits - 1)), 1 << (bits - 1), (16, 64), dtype=torch.int8
    )
    assert torch.equal(
        pack_signed(values, bits, packed_dim=1),
        reference.pack_to_int32(values, bits, packed_dim=1),
    )
    assert torch.equal(
        pack_signed(values, bits, packed_dim=0),
        reference.pack_to_int32(values, bits, packed_dim=0),
    )


@pytest.mark.parametrize("symmetric", [False, True])
def test_compressed_tensors_external_decompressor_reads_export(symmetric):
    ct = pytest.importorskip("compressed_tensors.quantization")
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        PackedQuantizationCompressor,
    )

    state = LatticeState.from_weight(
        torch.randn(16, 64), bits=4, group_size=32, symmetric=symmetric
    )
    packed = pack_layer(state, "compressed-tensors").tensors
    args = ct.QuantizationArgs(
        num_bits=4,
        symmetric=symmetric,
        strategy=ct.QuantizationStrategy.GROUP,
        group_size=32,
    )
    decompressed = PackedQuantizationCompressor().decompress_weight(packed, args)
    assert torch.allclose(decompressed.float(), state.weight().float(), atol=2e-3, rtol=0)


def test_export_preserves_huggingface_config_and_biases(tmp_path):
    state = LatticeState.from_weight(torch.randn(8, 64), bits=4, group_size=32)
    output = export_checkpoint(
        {"model.layers.0.self_attn.q_proj": state},
        tmp_path / "model",
        fmt="compressed-tensors",
        extra_tensors={
            "model.layers.0.self_attn.q_proj.bias": torch.randn(8),
            "model.embed_tokens.weight": torch.randn(32, 64),
        },
        model_config={
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
        },
    )
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "llama"
    assert config["quantization_config"]["format"] == "pack-quantized"

    from safetensors.torch import load_file

    tensors = load_file(output / "model.safetensors")
    assert "model.layers.0.self_attn.q_proj.weight_packed" in tensors
    assert "model.layers.0.self_attn.q_proj.bias" in tensors


def test_export_accepts_tied_unquantized_tensors(tmp_path):
    from safetensors.torch import load_file

    state = LatticeState.from_weight(torch.randn(8, 64), bits=4, group_size=32)
    tied = torch.randn(32, 64)
    output = export_checkpoint(
        {"model.layers.0.self_attn.q_proj": state},
        tmp_path / "tied",
        fmt="awq",
        extra_tensors={"model.embed_tokens.weight": tied, "lm_head.weight": tied},
    )
    tensors = load_file(output / "model.safetensors")
    assert torch.equal(tensors["model.embed_tokens.weight"], tied)
    assert torch.equal(tensors["lm_head.weight"], tied)


def test_compressed_tensors_config_is_accepted_by_vllm_when_available():
    vllm_config_module = pytest.importorskip(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
    )
    from clc.export.checkpoint import quantization_config

    config = quantization_config("compressed-tensors", 4, 128, ["lm_head"])
    parsed = vllm_config_module.CompressedTensorsConfig.from_config(config)
    assert parsed.quant_format == "pack-quantized"
    assert parsed.target_scheme_map["Linear"]["weights"].group_size == 128


def test_legacy_fake_quantized_checkpoint_is_reprojected_and_packed(tmp_path):
    from safetensors.torch import load_file, save_file

    source = tmp_path / "legacy"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": 128})
    )
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": torch.randn(8, 128),
            "model.embed_tokens.weight": torch.randn(16, 128),
            "lm_head.weight": torch.randn(16, 128),
        },
        source / "model.safetensors",
    )

    output = convert_legacy_awq_checkpoint(source, tmp_path / "converted")
    tensors = load_file(output / "model.safetensors")
    assert "model.layers.0.mlp.down_proj.qweight" in tensors
    assert "model.layers.0.mlp.down_proj.weight" not in tensors
    assert "lm_head.weight" in tensors
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["clc_legacy_conversion"]["reprojected"] is True
    assert "lm_head" in metadata["quantization_config"]["modules_to_not_convert"]
