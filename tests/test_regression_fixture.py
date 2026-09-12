"""Regression on a real AWQ-scaled layer column.

``awq_layer3_gate_proj_channel0.csv`` holds one output channel (2304 input
coordinates, exactly 18 groups of 128) exported from MiniCPM-2B layer 3
``gate_proj``: the AWQ per-input-channel scale, the scaled weight ``W_scaled``
and the scaled activation mean ``E[Xs]``.  Appendix G.1 says CLC operates in
exactly this transformed coordinate system, so it is a faithful single-channel
instance of the real problem.
"""

import csv
from pathlib import Path

import pytest
import torch

from clc.correction import CorrectionConfig, LatticeCorrection
from clc.lattice import LatticeState

FIXTURE = Path(__file__).parent / "fixtures" / "awq_layer3_gate_proj_channel0.csv"


@pytest.fixture(scope="module")
def layer_column():
    with FIXTURE.open() as handle:
        rows = list(csv.DictReader(handle))
    activation_mean = torch.tensor([float(r["E[Xs]"]) for r in rows], dtype=torch.float32)
    weight = torch.tensor([float(r["W_scaled"]) for r in rows], dtype=torch.float32)
    return weight.unsqueeze(0), activation_mean


def shift(state, mu):
    return float((state.residual() * mu.unsqueeze(0)).sum())


def test_fixture_is_an_exact_number_of_groups(layer_column):
    weight, _ = layer_column
    assert weight.shape == (1, 2304)
    assert 2304 % 128 == 0


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("order", ["residual", "normalized"])
def test_correction_reduces_the_first_moment_shift(layer_column, bits, order):
    weight, mu = layer_column
    state = LatticeState.from_weight(weight, bits=bits, group_size=128)
    before = abs(shift(state, mu))

    LatticeCorrection(CorrectionConfig(candidate_order=order)).apply(state, mu)

    assert abs(shift(state, mu)) < before


def test_reported_reduction_on_the_reference_channel(layer_column):
    """Pins the 4-bit numbers so an accidental behaviour change is visible."""
    weight, mu = layer_column
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    before = abs(shift(state, mu))
    stats = LatticeCorrection(CorrectionConfig(budget_fraction=0.05)).apply(state, mu)
    after = abs(shift(state, mu))

    assert before == pytest.approx(3.955e-3, rel=1e-3)
    assert after == pytest.approx(2.378e-4, rel=1e-2)
    assert stats.flip_count == 2
    assert stats.mean_shift_gain > 0


def test_weight_ordering_beats_code_ordering_at_four_bits(layer_column):
    """|e| ordering reaches a smaller residual with fewer flips here."""
    weight, mu = layer_column
    results = {}
    for order in ("residual", "normalized"):
        state = LatticeState.from_weight(weight, bits=4, group_size=128)
        stats = LatticeCorrection(CorrectionConfig(candidate_order=order)).apply(state, mu)
        results[order] = (abs(shift(state, mu)), stats.flip_count)

    assert results["residual"][0] < results["normalized"][0]
    assert results["residual"][1] <= results["normalized"][1]
