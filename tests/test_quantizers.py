"""Layer-wise behaviour of each base quantizer -- the objective each one optimizes."""

import pytest
import torch
import torch.nn as nn

from clc.correction import LatticeCorrection
from clc.quantizers.awq import AWQQuantizer, fold_channel_scale
from clc.quantizers.base import QuantConfig, writeback
from clc.quantizers.gptq import GPTQQuantizer
from clc.quantizers.rtn import RTNQuantizer
from clc.statistics import ActivationStatistics


@pytest.fixture(scope="module")
def layer_and_activations():
    torch.manual_seed(0)
    module = nn.Linear(512, 128, bias=False)
    activations = torch.randn(4000, 512) * 0.5 + 0.2
    activations[:, :8] *= 8  # salient input channels, as AWQ assumes
    stats = ActivationStatistics(512, track_covariance=True, sample_limit=1024)
    stats.update(activations)
    return module, activations, stats


def layer_error(module_weight, weight, activations):
    return (activations @ module_weight.t() - activations @ weight.t()).pow(2).mean().item()


# -- RTN ---------------------------------------------------------------------

def test_rtn_is_the_nearest_level_projection(layer_and_activations):
    module, _, stats = layer_and_activations
    layer = RTNQuantizer(QuantConfig(bits=4)).quantize(module, stats)
    assert layer.state.normalized_residual().abs().max() <= 0.5 + 1e-6
    assert layer.channel_scale is None


def test_rtn_reports_the_untransformed_mean(layer_and_activations):
    module, _, stats = layer_and_activations
    layer = RTNQuantizer(QuantConfig(bits=4)).quantize(module, stats)
    assert torch.allclose(layer.activation_mean, stats.mean, atol=1e-6)


# -- AWQ ---------------------------------------------------------------------

def test_awq_beats_rtn_on_the_layer_objective(layer_and_activations):
    """Protecting salient channels is the whole point of the scale search."""
    module, activations, stats = layer_and_activations
    reference = module.weight.data

    rtn = RTNQuantizer(QuantConfig(bits=3)).quantize(module, stats)
    awq = AWQQuantizer(QuantConfig(bits=3)).quantize(module, stats)

    rtn_error = layer_error(reference, rtn.state.weight(), activations)
    awq_error = layer_error(
        reference, awq.state.weight() / awq.channel_scale.unsqueeze(0), activations
    )
    assert awq_error < rtn_error


def test_awq_transforms_the_mean_as_appendix_g1_requires(layer_and_activations):
    """mu' = S^-1 mu, in the coordinate system the lattice lives in."""
    module, _, stats = layer_and_activations
    layer = AWQQuantizer(QuantConfig(bits=4)).quantize(module, stats)
    assert torch.allclose(layer.activation_mean, stats.mean / layer.channel_scale, atol=1e-5)


def test_folding_preserves_the_composition(layer_and_activations):
    """The producer absorbs 1/alpha, so the two-layer function is unchanged."""
    _, activations, _ = layer_and_activations
    torch.manual_seed(1)
    producer = nn.Linear(512, 512, bias=False)
    consumer = nn.Linear(512, 128, bias=False)
    scale = torch.rand(512) + 0.5

    reference = (activations @ producer.weight.data.t()) @ consumer.weight.data.t()
    fold_channel_scale(producer, [consumer], scale)
    folded = (activations @ producer.weight.data.t()) @ consumer.weight.data.t()

    assert torch.allclose(reference, folded, atol=1e-3)


def test_folding_into_a_norm_scales_its_elementwise_weight():
    norm = nn.LayerNorm(64)
    consumer = nn.Linear(64, 16, bias=False)
    original = norm.weight.data.clone()
    scale = torch.rand(64) + 0.5

    fold_channel_scale(norm, [consumer], scale)
    assert torch.allclose(norm.weight.data, original / scale, atol=1e-6)


def test_folding_rejects_a_shape_mismatch():
    norm = nn.LayerNorm(64)
    with pytest.raises(ValueError, match="channels"):
        fold_channel_scale(norm, [nn.Linear(32, 8)], torch.ones(32))


# -- GPTQ --------------------------------------------------------------------

def test_gptq_beats_rtn_on_the_layer_objective(layer_and_activations):
    """Hessian-aware error compensation is what GPTQ buys over plain rounding."""
    module, activations, stats = layer_and_activations
    reference = module.weight.data

    rtn = RTNQuantizer(QuantConfig(bits=3)).quantize(module, stats)
    gptq = GPTQQuantizer(QuantConfig(bits=3)).quantize(module, stats)

    assert layer_error(reference, gptq.state.weight(), activations) < layer_error(
        reference, rtn.state.weight(), activations
    )


def test_gptq_scores_the_shift_against_the_original_weight(layer_and_activations):
    """Appendix G.2: b_j is measured on W, while the flip direction follows w_tilde."""
    module, _, stats = layer_and_activations
    layer = GPTQQuantizer(QuantConfig(bits=3)).quantize(module, stats)

    assert torch.allclose(layer.state.float_weights, module.weight.data.float(), atol=1e-6)
    assert not torch.allclose(
        layer.state.pre_round,
        layer.state.float_weights / layer.state.step + layer.state.zero_point,
    )


def test_gptq_stays_on_the_same_lattice_as_rtn(layer_and_activations):
    """GPTQ changes which level a weight lands on, not the grid itself."""
    module, _, stats = layer_and_activations
    rtn = RTNQuantizer(QuantConfig(bits=3)).quantize(module, stats)
    gptq = GPTQQuantizer(QuantConfig(bits=3)).quantize(module, stats)

    assert torch.equal(rtn.state.step_groups, gptq.state.step_groups)
    assert torch.equal(rtn.state.zero_groups, gptq.state.zero_groups)
    assert not torch.equal(rtn.state.codes, gptq.state.codes)


# -- correction on top of each base ------------------------------------------

@pytest.mark.parametrize("bits", [3, 4])
def test_correction_descends_on_every_base_quantizer(layer_and_activations, bits):
    module, _, stats = layer_and_activations
    for quantizer in (
        RTNQuantizer(QuantConfig(bits=bits)),
        AWQQuantizer(QuantConfig(bits=bits)),
        GPTQQuantizer(QuantConfig(bits=bits)),
    ):
        layer = quantizer.quantize(module, stats)
        correction = LatticeCorrection().apply(
            layer.state, layer.activation_mean, layer.pooled_variance
        )
        assert correction.mean_shift_gain >= 0, quantizer.name


def test_writeback_unscales_an_unfolded_transform(layer_and_activations):
    module, activations, stats = layer_and_activations
    original = module.weight.data.clone()
    try:
        layer = AWQQuantizer(QuantConfig(bits=4)).quantize(module, stats)
        writeback(module, layer)
        # The stored weight is back in the original coordinate system.
        assert layer_error(original, module.weight.data, activations) < 1e-2
    finally:
        module.weight.data = original


# -- AdaRound ----------------------------------------------------------------

def test_adaround_beats_nearest_level_rounding(layer_and_activations):
    from clc.quantizers.adaround import AdaRoundQuantizer

    module, activations, stats = layer_and_activations
    reference = module.weight.data

    rtn = RTNQuantizer(QuantConfig(bits=3)).quantize(module, stats)
    learned = AdaRoundQuantizer(QuantConfig(bits=3), iterations=400).quantize(module, stats)

    assert layer_error(reference, learned.state.weight(), activations) < layer_error(
        reference, rtn.state.weight(), activations
    )


def test_adaround_may_choose_the_farther_level(layer_and_activations):
    """Appendix G.4: the residual is no longer bounded by s / 2.

    That is why Theorem 1's certificate does not transfer to AdaRound, even though
    the first-moment descent of Proposition 1 still holds.
    """
    from clc.quantizers.adaround import AdaRoundQuantizer

    module, _, stats = layer_and_activations
    learned = AdaRoundQuantizer(QuantConfig(bits=3), iterations=400).quantize(module, stats)
    assert learned.state.normalized_residual().abs().max() > 0.5


def test_adaround_flip_reopens_the_learned_choice(layer_and_activations):
    """sigma must send floor(u) + h to floor(u) + (1 - h), not to the nearest level."""
    from clc.quantizers.adaround import AdaRoundQuantizer

    module, _, stats = layer_and_activations
    learned = AdaRoundQuantizer(QuantConfig(bits=3), iterations=200).quantize(module, stats)
    state = learned.state

    continuous = state.pre_round
    floor_codes = torch.floor(continuous)
    chose_up = (state.codes - floor_codes) == 1
    interior = (state.codes > state.min_code) & (state.codes < state.max_code)

    direction = state.flip_direction()
    assert torch.all(direction[chose_up & interior] == -1)
    assert torch.all(direction[~chose_up & interior] == 1)


def test_adaround_keeps_the_same_lattice(layer_and_activations):
    from clc.quantizers.adaround import AdaRoundQuantizer

    module, _, stats = layer_and_activations
    rtn = RTNQuantizer(QuantConfig(bits=3)).quantize(module, stats)
    learned = AdaRoundQuantizer(QuantConfig(bits=3), iterations=200).quantize(module, stats)
    assert torch.equal(rtn.state.step_groups, learned.state.step_groups)


def test_correction_still_descends_on_adaround(layer_and_activations):
    from clc.quantizers.adaround import AdaRoundQuantizer

    module, _, stats = layer_and_activations
    learned = AdaRoundQuantizer(QuantConfig(bits=3), iterations=200).quantize(module, stats)
    stats_out = LatticeCorrection().apply(
        learned.state, learned.activation_mean, learned.pooled_variance
    )
    assert stats_out.mean_shift_gain >= 0
