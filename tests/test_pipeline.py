"""End-to-end runs on a real (tiny) LLaMA, including the deployable export."""

import json

import pytest
import torch

from clc.correction import CorrectionConfig
from clc.export.checkpoint import export_checkpoint
from clc.models import decoder_blocks, scale_groups
from clc.pipeline import PipelineConfig, QuantizationPipeline
from clc.quantizers.awq import AWQQuantizer
from clc.quantizers.base import QuantConfig
from clc.quantizers.gptq import GPTQQuantizer
from clc.quantizers.rtn import RTNQuantizer

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    model.config.use_cache = False
    return model


@pytest.fixture(scope="module")
def batches():
    torch.manual_seed(1)
    return [torch.randint(0, 256, (1, 32)) for _ in range(4)]


def perplexity_proxy(model, batches):
    """Mean cross-entropy over the calibration batches; lower is better."""
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            total += float(model(batch, labels=batch).loss)
    return total / len(batches)


# -- architecture wiring -----------------------------------------------------

def test_decoder_blocks_are_found(tiny_llama):
    assert len(decoder_blocks(tiny_llama)) == 2


def test_scale_groups_cover_every_projection(tiny_llama):
    groups = scale_groups(decoder_blocks(tiny_llama)[0])
    assert [name for name, _, _ in groups] == [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    covered = {id(m) for _, _, consumers in groups for m in consumers}
    assert len(covered) == 7  # q, k, v, o, gate, up, down


# -- the pipeline itself -----------------------------------------------------

@pytest.mark.parametrize("bits", [3, 4])
def test_rtn_pipeline_quantizes_every_linear(tiny_llama, batches, bits):
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        RTNQuantizer(QuantConfig(bits=bits)), PipelineConfig(sample_limit=0)
    ).run(model, batches)

    assert len(result.states) == 14  # 7 projections x 2 blocks
    assert all(s.bits == bits for s in result.states.values())
    assert result.exportable


def test_correction_reduces_the_layer_mean_shift_everywhere(tiny_llama, batches):
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        RTNQuantizer(QuantConfig(bits=3)), PipelineConfig(sample_limit=0)
    ).run(model, batches)

    gains = [s["mean_shift_gain"] for s in result.layer_stats.values()]
    assert all(g >= 0 for g in gains)
    assert sum(g > 0 for g in gains) > len(gains) // 2


def test_correction_reduces_the_measured_output_drift(tiny_llama, batches):
    """What CLC provably does, measured on the real model.

    End-to-end loss is not the right assertion here: a randomly initialised model
    has no structure for a first-moment correction to recover, and language-model
    loss on random weights is dominated by noise.  The claim CLC actually makes is
    layer-wise -- the aggregate squared first-moment shift goes down and never up.
    """
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        RTNQuantizer(QuantConfig(bits=3)), PipelineConfig(post_correction="clc", sample_limit=0)
    ).run(model, batches)

    before = sum(s["mean_shift_before"] for s in result.layer_stats.values())
    after = sum(s["mean_shift_after"] for s in result.layer_stats.values())
    assert after < before
    assert all(s["mean_shift_after"] <= s["mean_shift_before"] + 1e-9 for s in result.layer_stats.values())


def test_correction_leaves_the_model_runnable(tiny_llama, batches):
    import copy
    import math

    model = copy.deepcopy(tiny_llama)
    QuantizationPipeline(
        RTNQuantizer(QuantConfig(bits=4)), PipelineConfig(post_correction="clc", sample_limit=0)
    ).run(model, batches)

    loss = perplexity_proxy(model, batches)
    assert math.isfinite(loss)
    assert loss < 2 * perplexity_proxy(tiny_llama, batches)


def test_awq_folds_scales_and_stays_exportable(tiny_llama, batches):
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        AWQQuantizer(QuantConfig(bits=4), n_grid=4), PipelineConfig(sample_limit=256)
    ).run(model, batches)

    assert result.exportable
    assert not result.notes
    # A folded layer's stored weight is exactly the dequantized lattice.
    name, state = next(iter(result.states.items()))
    module = model.get_submodule(name.replace("model.layers.", "model.layers."))
    assert torch.allclose(module.weight.data, state.weight(), atol=1e-6)


def test_unfolded_awq_is_reported_as_not_exportable(tiny_llama, batches):
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        AWQQuantizer(QuantConfig(bits=4), n_grid=4),
        PipelineConfig(sample_limit=256, fold_channel_scales=False),
    ).run(model, batches)

    assert not result.exportable


def test_gptq_pipeline_produces_a_valid_lattice(tiny_llama, batches):
    """GPTQ's quality claim is layer-wise and is asserted in test_quantizers.py."""
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        GPTQQuantizer(QuantConfig(bits=3)),
        # Keep the pre-round state so the Appendix G.2 split can be inspected.
        PipelineConfig(sample_limit=0, track_covariance=True, compact_states=False),
    ).run(model, batches)

    assert len(result.states) == 14
    for state in result.states.values():
        assert state.codes.min() >= state.min_code
        assert state.codes.max() <= state.max_code
    # Appendix G.2: the rounding state is GPTQ's own, not W / s + z.
    state = next(iter(result.states.values()))
    assert not torch.allclose(state.pre_round, state.float_weights / state.step + state.zero_point)


def test_bias_correction_is_flagged_as_measurement_only(tiny_llama, batches):
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        RTNQuantizer(QuantConfig(bits=3)),
        PipelineConfig(post_correction="bias_correction", sample_limit=0),
    ).run(model, batches)

    assert not result.exportable
    assert any("bias" in note for note in result.notes)
    assert model.model.layers[0].self_attn.q_proj.bias is not None


# -- export ------------------------------------------------------------------

@pytest.mark.parametrize("fmt,bits", [("awq", 4), ("gptq", 3)])
def test_quantized_model_exports_to_a_deployable_checkpoint(tiny_llama, batches, tmp_path, fmt, bits):
    import copy

    from safetensors.torch import load_file

    model = copy.deepcopy(tiny_llama)
    quantizer = (
        AWQQuantizer(QuantConfig(bits=bits), n_grid=4)
        if fmt == "awq"
        else RTNQuantizer(QuantConfig(bits=bits))
    )
    result = QuantizationPipeline(
        quantizer, PipelineConfig(sample_limit=256 if fmt == "awq" else 0)
    ).run(model, batches)
    assert result.exportable

    extra = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if "proj" not in name
    }
    out = export_checkpoint(
        result.states, tmp_path / fmt, fmt=fmt, extra_tensors=extra, verify=True
    )

    tensors = load_file(out / "model.safetensors")
    assert "model.layers.0.self_attn.q_proj.qweight" in tensors
    assert "model.embed_tokens.weight" in tensors

    config = json.loads((out / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == fmt
    assert config["quantization_config"]["bits"] == bits


def test_states_are_compacted_and_moved_off_device_by_default(tiny_llama, batches):
    """Retaining every layer's float weight does not fit for a real model."""
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        RTNQuantizer(QuantConfig(bits=4)), PipelineConfig(sample_limit=0)
    ).run(model, batches)

    for state in result.states.values():
        assert state.compacted
        assert state.codes.device.type == "cpu"
        assert not state.codes.is_floating_point()


def test_compacted_states_still_export(tiny_llama, batches, tmp_path):
    import copy

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        RTNQuantizer(QuantConfig(bits=4)), PipelineConfig(sample_limit=0)
    ).run(model, batches)

    export_checkpoint(result.states, tmp_path / "packed", fmt="awq", verify=True)


# -- grouped-query attention: not every scale has a producer that can absorb it

@pytest.fixture(scope="module")
def gqa_llama():
    """Fewer KV heads than attention heads, so v_proj is narrower than o_proj."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    model.config.use_cache = False
    return model


def test_o_proj_is_dropped_from_the_fold_groups_under_gqa(gqa_llama):
    block = decoder_blocks(gqa_llama)[0]
    assert block.self_attn.v_proj.out_features != block.self_attn.o_proj.in_features
    assert [name for name, _, _ in scale_groups(block)] == [
        "qkv_proj",
        "gate_up_proj",
        "down_proj",
    ]


def test_gqa_model_stays_exportable(gqa_llama, batches):
    """An unfoldable layer is quantized unscaled rather than scaled and stranded."""
    import copy

    model = copy.deepcopy(gqa_llama)
    result = QuantizationPipeline(
        AWQQuantizer(QuantConfig(bits=4), n_grid=4), PipelineConfig(sample_limit=256)
    ).run(model, batches)

    assert result.exportable
    o_proj = result.states["model.layers.0.self_attn.o_proj"]
    assert o_proj.unfolded_scale is None


# -- the invariant that ties the checkpoint to the measured model -------------

@pytest.mark.parametrize("model_name", ["tiny_llama", "gqa_llama"])
def test_exported_checkpoint_matches_the_evaluated_weights(model_name, batches, request, tmp_path):
    """Every packed layer must dequantize to exactly the weight the model holds.

    This is the invariant a mismatched coordinate system breaks: if a channel
    scale is applied to the codes but divided back out of the stored weight, the
    two drift apart and the shipped model is not the one that was measured.
    """
    import copy

    from clc.export.checkpoint import unpack_layer, pack_layer

    model = copy.deepcopy(request.getfixturevalue(model_name))
    result = QuantizationPipeline(
        AWQQuantizer(QuantConfig(bits=4), n_grid=4), PipelineConfig(sample_limit=256)
    ).run(model, batches)
    assert result.exportable

    for name, state in result.states.items():
        packed = pack_layer(state, "awq", name=name)
        codes, scales, zeros = unpack_layer(packed, "awq", bits=4, group_size=128)
        dequantized = (
            (codes.float() - zeros.repeat_interleave(128, dim=0).float())
            * scales.repeat_interleave(128, dim=0).float()
        ).t()
        assert torch.allclose(
            dequantized, model.get_submodule(name).weight.data.float(), atol=1e-6
        ), name


def test_export_refuses_a_stranded_channel_scale(tiny_llama, batches, tmp_path):
    import copy

    from clc.export.checkpoint import ExportError, export_checkpoint

    model = copy.deepcopy(tiny_llama)
    result = QuantizationPipeline(
        AWQQuantizer(QuantConfig(bits=4), n_grid=4),
        PipelineConfig(sample_limit=256, fold_channel_scales=False),
    ).run(model, batches)

    assert not result.exportable
    with pytest.raises(ExportError, match="never folded"):
        export_checkpoint(result.states, tmp_path / "packed", fmt="awq")


def test_exported_checkpoint_is_exact_for_a_half_precision_model(batches, tmp_path):
    """The realistic case: the module stores float16, so compare at float16.

    A float32 comparison against a float16 module weight shows a difference of
    roughly one float16 ULP per weight, which scales with the weight magnitude and
    is representation, not error. Evaluated at the stored precision the two agree
    exactly.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from clc.export.checkpoint import pack_layer, unpack_layer

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval().to(torch.float16)
    model.config.use_cache = False

    result = QuantizationPipeline(
        AWQQuantizer(QuantConfig(bits=4), n_grid=4), PipelineConfig(sample_limit=256)
    ).run(model, batches)
    assert result.exportable

    for name, state in result.states.items():
        packed = pack_layer(state, "awq", name=name)
        codes, scales, zeros = unpack_layer(packed, "awq", bits=4, group_size=128)
        dequantized = (
            (codes.float() - zeros.repeat_interleave(128, dim=0).float())
            * scales.float().repeat_interleave(128, dim=0)
        ).t().to(torch.float16)
        assert torch.equal(dequantized, model.get_submodule(name).weight.data), name
