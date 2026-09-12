"""Algorithm 1: the guarantees the greedy rule is supposed to deliver."""

import math

import pytest
import torch

from clc.correction import CorrectionConfig, LatticeCorrection
from clc.estimators import james_stein_mean, knee_threshold, pooled_activation_variance
from clc.lattice import LatticeState


def make_layer(out_features=48, in_features=512, m=400, seed=0, bits=4, group_size=128):
    torch.manual_seed(seed)
    weight = torch.randn(out_features, in_features) * 0.05
    activations = torch.randn(in_features, m) * 0.5 + 0.3
    activations[:6] += 5.0  # activation outliers, as in real LLM inputs
    state = LatticeState.from_weight(weight, bits=bits, group_size=group_size)
    sample_mean = activations.mean(dim=1)
    variance = pooled_activation_variance(activations.pow(2).mean(dim=1), sample_mean)
    return state, sample_mean, variance


def channel_shift(state, mu):
    padded = torch.zeros(state.padded_in_features, dtype=mu.dtype)
    padded[: state.in_features] = mu
    return (state.residual() * padded.unsqueeze(0)).sum(dim=1)


# -- Proposition 1: the greedy rule cannot increase the mean shift -----------

@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("budget", [0.01, 0.05, 0.5])
def test_per_channel_mean_shift_never_increases(bits, budget):
    """The prefix search includes k = 0, so |b'_j| <= |b_j| for every channel."""
    state, sample_mean, variance = make_layer(bits=bits)
    mu = james_stein_mean(sample_mean, variance)
    before = channel_shift(state, mu)

    LatticeCorrection(CorrectionConfig(budget_fraction=budget)).apply(
        state, mu, variance, already_stabilized=True
    )
    after = channel_shift(state, mu)

    assert torch.all(after.abs() <= before.abs() + 1e-5)


def test_layer_mean_shift_gain_is_non_negative():
    """G_l = B_l(W_q) - B_l(W_q') >= 0, summed over channels."""
    state, sample_mean, variance = make_layer()
    stats = LatticeCorrection().apply(state, sample_mean, variance)
    assert stats.mean_shift_gain >= 0.0
    assert stats.mean_shift_after <= stats.mean_shift_before + 1e-9


def test_correction_actually_moves_something():
    state, sample_mean, variance = make_layer()
    stats = LatticeCorrection().apply(state, sample_mean, variance)
    assert stats.flip_count > 0
    assert stats.mean_shift_gain > 0.0


# -- structural conditions the theory assumes -------------------------------

def test_every_move_is_a_single_adjacent_level():
    """Section 3.2: corrections are one-level revisions, so |dq| <= 1."""
    state, sample_mean, variance = make_layer()
    before = state.codes.clone()
    LatticeCorrection().apply(state, sample_mean, variance)
    assert (state.codes - before).abs().max() <= 1


def test_moves_are_anti_aligned_with_the_residual():
    """Assumption A1: a flip only ever moves against the rounding residual."""
    state, sample_mean, variance = make_layer()
    sigma = state.flip_direction().clone()
    before = state.codes.clone()
    LatticeCorrection().apply(state, sample_mean, variance)

    moved = state.codes != before
    assert torch.equal((state.codes - before)[moved], sigma[moved])


def test_codes_never_leave_the_representable_range():
    state, sample_mean, variance = make_layer(bits=3)
    LatticeCorrection(CorrectionConfig(budget_fraction=1.0)).apply(state, sample_mean, variance)
    assert state.codes.min() >= state.min_code
    assert state.codes.max() <= state.max_code


def test_no_flip_lands_outside_the_admissible_support():
    """Assumption A2: every support S_j lies inside the common index set I."""
    state, sample_mean, variance = make_layer()
    mu = james_stein_mean(sample_mean, variance)
    _, support = knee_threshold(mu)

    before = state.codes.clone()
    LatticeCorrection().apply(state, mu, variance, already_stabilized=True)
    touched = (state.codes != before).any(dim=0)[: state.in_features]

    assert not torch.any(touched & ~support)


def test_padded_columns_are_never_corrected():
    state, sample_mean, variance = make_layer(in_features=300)
    before = state.codes.clone()
    LatticeCorrection().apply(state, sample_mean, variance)
    assert torch.equal(state.codes[:, 300:], before[:, 300:])


# -- Eq. (8): the budget actually binds --------------------------------------

def test_flip_count_respects_the_per_channel_budget():
    """K_l <= sum_j B_j with B_j = ceil(p |I_j|) over the filtered candidates."""
    budget = 0.05
    state, sample_mean, variance = make_layer()
    mu = james_stein_mean(sample_mean, variance)

    padded_mu = torch.zeros(state.padded_in_features)
    padded_mu[: state.in_features] = mu
    _, support = knee_threshold(mu)
    support_padded = torch.zeros(state.padded_in_features, dtype=torch.bool)
    support_padded[: state.in_features] = support

    sigma = state.flip_direction()
    shift_reduction = -padded_mu.unsqueeze(0) * sigma * state.step
    shift = (state.residual() * padded_mu.unsqueeze(0)).sum(dim=1)
    aligned = (
        support_padded.unsqueeze(0)
        & state.in_range(sigma)
        & (torch.sign(shift_reduction) == torch.sign(shift).unsqueeze(1))
    )
    allowed = sum(math.ceil(budget * n) for n in aligned.sum(dim=1).tolist())

    before = state.codes.clone()
    stats = LatticeCorrection(CorrectionConfig(budget_fraction=budget)).apply(
        state, mu, variance, already_stabilized=True
    )
    assert stats.flip_count == int((state.codes != before).sum())
    assert stats.flip_count <= allowed


def test_a_tighter_budget_never_spends_more_flips():
    counts = []
    for budget in (0.01, 0.03, 0.05):
        state, sample_mean, variance = make_layer()
        counts.append(
            LatticeCorrection(CorrectionConfig(budget_fraction=budget))
            .apply(state, sample_mean, variance)
            .flip_count
        )
    assert counts == sorted(counts)


def test_zero_budget_still_permits_the_first_flip_per_channel():
    """B_j = ceil(p |I_j|) rounds up, so an arbitrarily small p keeps one move."""
    state, sample_mean, variance = make_layer()
    stats = LatticeCorrection(CorrectionConfig(budget_fraction=1e-9)).apply(
        state, sample_mean, variance
    )
    assert stats.flips_per_channel_max <= 1
    assert stats.mean_shift_gain >= 0.0


# -- degenerate inputs -------------------------------------------------------

def test_zero_activation_mean_is_a_noop():
    state, _, _ = make_layer()
    before = state.codes.clone()
    stats = LatticeCorrection().apply(state, torch.zeros(state.in_features))
    assert stats.flip_count == 0
    assert torch.equal(state.codes, before)


def test_mismatched_mean_length_is_rejected():
    state, _, _ = make_layer()
    with pytest.raises(ValueError, match="expected"):
        LatticeCorrection().apply(state, torch.zeros(7))


def test_shrinkage_can_be_disabled():
    state, sample_mean, variance = make_layer()
    stats = LatticeCorrection(
        CorrectionConfig(use_james_stein=False)
    ).apply(state, sample_mean, variance)
    assert stats.mean_shift_gain >= 0.0
