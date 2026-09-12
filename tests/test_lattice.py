"""Invariants of the integer lattice a corrected layer must never leave."""

import pytest
import torch

from clc.lattice import LatticeState


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(16, 300) * 0.05


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("symmetric", [False, True])
def test_codes_stay_in_representable_range(weight, bits, symmetric):
    state = LatticeState.from_weight(weight, bits=bits, group_size=128, symmetric=symmetric)
    assert state.codes.min() >= state.min_code
    assert state.codes.max() <= state.max_code
    assert torch.equal(state.codes, state.codes.round())


def test_padding_is_transparent(weight):
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    assert state.in_features == 300
    assert state.padded_in_features == 384
    assert state.n_groups == 3
    assert state.weight().shape == weight.shape
    # Padded columns hold no signal.
    assert torch.count_nonzero(state.float_weights[:, 300:]) == 0


def test_residual_matches_normalized_residual(weight):
    """e = -r * s  ties the paper's residual to the rounding state."""
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    assert torch.allclose(state.residual(), -state.normalized_residual() * state.step, atol=1e-6)


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_nearest_level_projection_is_within_half_a_step(bits):
    """Lemma 1(ii) needs |e_kj| <= s_j / 2 at every coordinate, not just typically.

    The step is stored at float16, and rounding it down would shrink the group's
    range enough for its extreme weight to clamp and exceed the bound; the step is
    therefore rounded up.
    """
    torch.manual_seed(0)
    weight = torch.randn(64, 512)
    state = LatticeState.from_weight(weight, bits=bits, group_size=128)
    assert state.normalized_residual().abs().max() <= 0.5 + 1e-6


def test_step_is_stored_at_checkpoint_precision(weight):
    """Simulation and deployment must evaluate (q - z) * s at the same width."""
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    assert torch.equal(state.step_groups, state.step_groups.to(torch.float16).to(state.step_groups.dtype))


def test_full_precision_steps_can_be_requested(weight):
    state = LatticeState.from_weight(weight, bits=4, group_size=128, scale_dtype=None)
    assert not torch.equal(
        state.step_groups, state.step_groups.to(torch.float16).to(state.step_groups.dtype)
    )


def test_flip_direction_is_anti_aligned_with_the_residual(weight):
    """Assumption A1: sigma = -sign(e), with ties resolved to +1."""
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    sigma = state.flip_direction()
    expected = torch.sign(-state.residual())
    expected = torch.where(expected == 0, torch.ones_like(expected), expected)
    assert torch.equal(sigma, expected)


def test_group_step_is_constant_within_a_group_and_varies_across_them(weight):
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    step = state.step
    for g in range(state.n_groups):
        block = step[:, g * 128 : (g + 1) * 128]
        assert torch.allclose(block, block[:, :1].expand_as(block))
    # Assumption A2 is per-channel-per-group, not one step for the whole row.
    assert not torch.allclose(step[:, 0], step[:, 128])


def test_dequantization_follows_the_packed_checkpoint_convention(weight):
    """w = (q - z) * s is exactly what AWQ / GPTQ kernels compute."""
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    manual = (state.codes - state.zero_point) * state.step
    assert torch.allclose(state.dequantize(), manual)


def test_apply_flips_clamps_at_the_range_boundary(weight):
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    huge = torch.full_like(state.codes, 100.0)
    state.apply_flips(huge)
    assert state.codes.max() <= state.max_code


def test_rejects_non_positive_group_size(weight):
    with pytest.raises(ValueError):
        LatticeState.from_weight(weight, group_size=0)


# -- releasing what only the correction needed -------------------------------

def test_compact_preserves_the_quantized_weight(weight):
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    expected = state.weight().clone()
    state.compact()
    assert torch.equal(state.weight(), expected)


def test_compact_shrinks_retained_memory(weight):
    def retained(state):
        tensors = [state.codes, state.step_groups, state.zero_groups]
        if not state.compacted:
            tensors += [state.float_weights, state.pre_round]
        return sum(t.numel() * t.element_size() for t in tensors)

    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    before = retained(state)
    state.compact()
    assert retained(state) < before / 5


@pytest.mark.parametrize("method", ["residual", "normalized_residual", "flip_direction"])
def test_compacted_state_refuses_correction_quantities(weight, method):
    """Failing loudly beats silently correcting against a released weight."""
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    state.compact()
    with pytest.raises(RuntimeError, match="compact"):
        getattr(state, method)()


def test_compacted_codes_are_still_exact(weight):
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    expected = state.codes_int().clone()
    state.compact()
    assert torch.equal(state.codes_int(), expected)
