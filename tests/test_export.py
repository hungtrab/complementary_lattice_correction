"""Packing corrected codes must reproduce the lattice exactly, and match vLLM."""

import json

import pytest
import torch

from clc.correction import LatticeCorrection
from clc.export.checkpoint import (
    ExportError,
    export_checkpoint,
    pack_layer,
    quantization_config,
    unpack_layer,
    verify_layer,
)
from clc.export.packing import pack_awq, pack_gptq, unpack_awq, unpack_gptq
from clc.lattice import LatticeState


def corrected_state(out_features=128, in_features=512, bits=4, seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(out_features, in_features) * 0.05
    mean = torch.randn(in_features) * 0.4 + 0.2
    state = LatticeState.from_weight(weight, bits=bits, group_size=128)
    LatticeCorrection().apply(state, mean)
    return state


# -- bit packing -------------------------------------------------------------

@pytest.mark.parametrize("bits", [4, 8])
def test_awq_packing_round_trips(bits):
    torch.manual_seed(0)
    codes = torch.randint(0, 2**bits, (256, 64), dtype=torch.int32)
    assert torch.equal(unpack_awq(pack_awq(codes, bits), bits).to(torch.int32), codes)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_gptq_packing_round_trips(bits):
    torch.manual_seed(0)
    codes = torch.randint(0, 2**bits, (256, 64), dtype=torch.int32)
    packed = pack_gptq(codes, bits)
    assert packed.shape == (256 * bits // 32, 64)
    assert torch.equal(unpack_gptq(packed, bits).to(torch.int32), codes)


def test_awq_packing_matches_vllm():
    vllm_utils = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.quant_utils"
    )
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (256, 64), dtype=torch.int32)
    assert torch.equal(pack_awq(codes, 4), vllm_utils.awq_pack(codes.clone(), 4, 256, 64))


def test_gptq_three_bit_matches_the_autogptq_layout():
    """3-bit codes straddle int32 boundaries; the layout must match byte for byte."""
    torch.manual_seed(0)
    codes = torch.randint(0, 8, (256, 64), dtype=torch.int64)

    reference = torch.zeros((256 * 3) // 32, 64, dtype=torch.int64)
    i = row = 0
    while row < reference.shape[0]:
        for j in range(i, i + 10):
            reference[row] |= codes[j] << (3 * (j - i))
        i += 10
        reference[row] |= codes[i] << 30
        row += 1
        reference[row] |= (codes[i] >> 2) & 1
        i += 1
        for j in range(i, i + 10):
            reference[row] |= codes[j] << (3 * (j - i) + 1)
        i += 10
        reference[row] |= codes[i] << 31
        row += 1
        reference[row] |= (codes[i] >> 1) & 0x3
        i += 1
        for j in range(i, i + 10):
            reference[row] |= codes[j] << (3 * (j - i) + 2)
        i += 10
        row += 1
    reference = (reference - (reference >= 2**31).to(torch.int64) * 2**32).to(torch.int32)

    assert torch.equal(pack_gptq(codes.to(torch.int32), 3), reference)


# -- layer export ------------------------------------------------------------

@pytest.mark.parametrize("fmt,bits", [("awq", 4), ("gptq", 2), ("gptq", 3), ("gptq", 4), ("gptq", 8)])
def test_packed_layer_reproduces_the_lattice_exactly(fmt, bits):
    state = corrected_state(bits=bits)
    packed = pack_layer(state, fmt, name="test")
    verify_layer(state, packed, fmt)  # raises on any mismatch

    codes, _, _ = unpack_layer(packed, fmt, bits=state.bits, group_size=state.group_size)
    assert torch.equal(codes.to(torch.int32), state.codes_int().t())


def test_gptq_legacy_format_offsets_the_zero_point():
    state = corrected_state(bits=4)
    state.zero_groups = state.zero_groups.clamp(min=1)
    packed = pack_layer(state, "gptq", name="test", v2=False)
    verify_layer(state, packed, "gptq", v2=False)


def test_gptq_legacy_format_rejects_a_zero_zero_point():
    """The v1 layout stores z - 1, so z = 0 is unrepresentable."""
    state = corrected_state(bits=4)
    state.zero_groups[0, 0] = 0
    with pytest.raises(ExportError, match="legacy GPTQ format"):
        pack_layer(state, "gptq", name="test", v2=False)


def test_awq_rejects_bit_widths_it_cannot_deploy():
    state = corrected_state(bits=3)
    with pytest.raises(ExportError, match="4-bit only"):
        pack_layer(state, "awq", name="test")


def test_padded_layers_are_refused():
    state = corrected_state(in_features=300)
    with pytest.raises(ExportError, match="no packed representation"):
        pack_layer(state, "awq", name="test")


def test_awq_rejects_out_features_that_do_not_pack():
    state = corrected_state(out_features=12)
    with pytest.raises(ExportError, match="multiple of 8"):
        pack_layer(state, "awq", name="test")


# -- whole-checkpoint export -------------------------------------------------

def test_export_writes_weights_config_and_is_reloadable(tmp_path):
    from safetensors.torch import load_file

    states = {f"model.layers.0.mlp.{n}": corrected_state(seed=i) for i, n in enumerate(("gate_proj", "up_proj"))}
    out = export_checkpoint(states, tmp_path / "packed", fmt="awq")

    tensors = load_file(out / "model.safetensors")
    for name in states:
        assert {f"{name}.qweight", f"{name}.qzeros", f"{name}.scales"} <= set(tensors)

    config = json.loads((out / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "awq"
    assert config["quantization_config"]["bits"] == 4
    assert config["quantization_config"]["group_size"] == 128
    deployment = json.loads((out / "deployment.json").read_text())
    assert deployment["format"] == "awq"
    assert deployment["engines"]["vllm"]["status"] == "direct"


def test_export_carries_unquantized_tensors_through(tmp_path):
    from safetensors.torch import load_file

    states = {"model.layers.0.mlp.gate_proj": corrected_state()}
    extra = {"lm_head.weight": torch.randn(32, 16)}
    out = export_checkpoint(states, tmp_path / "packed", fmt="gptq", extra_tensors=extra)

    tensors = load_file(out / "model.safetensors")
    assert torch.equal(tensors["lm_head.weight"], extra["lm_head.weight"])


def test_export_refuses_an_empty_model(tmp_path):
    with pytest.raises(ExportError, match="no quantized layers"):
        export_checkpoint({}, tmp_path / "packed", fmt="awq")


def test_quantization_config_marks_gptq_v2():
    assert quantization_config("gptq", 3, 128, [])["checkpoint_format"] == "gptq_v2"
    assert quantization_config("gptq", 4, 128, [], v2=False)["checkpoint_format"] == "gptq"


def test_gptq_config_declares_the_positive_quantized_module_list():
    config = quantization_config(
        "gptq", 4, 128, ["lm_head"], quantized_modules=["model.layers.0.mlp.gate_proj"]
    )
    assert config["lm_head"] is False
    assert config["modules_in_block_to_quantize"] == [
        "model.layers.0.mlp.gate_proj"
    ]


# -- the deployment kernel itself --------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_vllm_kernel_dequantizes_to_the_simulated_weight_bit_for_bit():
    """The measured model and the shipped model must be the same model.

    ``LatticeState`` stores the step at float16, the precision the checkpoint
    keeps, so ``(q - z) * s`` evaluated in the kernel and in simulation agree
    exactly.  Without that rounding the two differ by roughly one float16 ULP
    per weight.
    """
    ops = pytest.importorskip("vllm._custom_ops")

    state = corrected_state()
    packed = pack_layer(state, "awq", name="test")

    dequantized = ops.awq_dequantize(
        packed.tensors["qweight"].cuda(),
        packed.tensors["scales"].cuda(),
        packed.tensors["qzeros"].cuda(),
        0,
        0,
        0,
    )
    simulated = state.weight().t().to(torch.float16).cuda()

    assert torch.equal(dequantized, simulated)
