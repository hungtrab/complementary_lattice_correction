"""`legacy=True` must reproduce the pre-refactor smart-flip result exactly.

The refactor changes the sign convention (the paper's ``e = W_q - W`` rather than
``W - W_q``), the candidate ordering, the budget rule and the shrinkage variance.
Only the last three change behaviour; the sign flip cancels out.  This test pins
that claim against the original implementation so previously reported numbers
stay reproducible.
"""

import importlib
import sys
from pathlib import Path

import pytest
import torch

from clc.correction import CorrectionConfig, LatticeCorrection
from clc.lattice import LatticeState

SMART_FLIP = Path(__file__).resolve().parents[2] / "smart-flip"


def _load_legacy():
    """Import the original CLCCorrection from the smart-flip checkout."""
    if not (SMART_FLIP / "src" / "post_correction" / "clc.py").exists():
        pytest.skip("smart-flip reference checkout not available")
    if str(SMART_FLIP) not in sys.path:
        sys.path.insert(0, str(SMART_FLIP))
    return importlib.import_module("src.post_correction.clc")


def _legacy_state(state: LatticeState):
    """Rebuild the legacy IntegerQuantizedTensorState from a LatticeState."""
    from src.quantization.state import IntegerQuantizedTensorState

    return IntegerQuantizedTensorState(
        float_weights=state.float_weights.clone(),
        pre_round=state.pre_round.clone(),
        integer_weights=state.codes.clone(),
        scale=state.step.clone(),
        zero_point=state.zero_point.clone(),
        max_int=state.max_code,
        min_int=state.min_code,
        in_features=state.in_features,
        padded_in_features=state.padded_in_features,
        original_dtype=state.original_dtype,
    )


@pytest.mark.parametrize("in_features", [512, 300])
@pytest.mark.parametrize("budget", [0.02, 0.05])
def test_legacy_mode_matches_the_original_implementation(in_features, budget):
    legacy_module = _load_legacy()

    torch.manual_seed(3)
    weight = torch.randn(24, in_features) * 0.05
    sample_mean = torch.randn(in_features) * 0.4 + 0.2
    sample_mean[:5] += 4.0

    refactored = LatticeState.from_weight(weight, bits=4, group_size=128)
    original = _legacy_state(LatticeState.from_weight(weight, bits=4, group_size=128))

    LatticeCorrection(
        CorrectionConfig(budget_fraction=budget, legacy=True)
    ).apply(refactored, sample_mean)

    legacy_correction = legacy_module.CLCCorrection(
        legacy_module.CLCConfig(max_flip_percent=budget, use_james_stein=True)
    )
    legacy_means = legacy_correction.prepare_activation_means(sample_mean)
    legacy_correction.apply(original, legacy_means)

    assert torch.equal(refactored.codes, original.integer_weights)


def test_paper_mode_is_more_conservative_than_legacy():
    """Eq. (8) caps the search at p|I_j|; legacy truncates afterwards at p*d."""
    torch.manual_seed(3)
    weight = torch.randn(24, 512) * 0.05
    sample_mean = torch.randn(512) * 0.4 + 0.2

    counts = {}
    for legacy in (False, True):
        state = LatticeState.from_weight(weight, bits=4, group_size=128)
        counts[legacy] = (
            LatticeCorrection(CorrectionConfig(budget_fraction=0.05, legacy=legacy))
            .apply(state, sample_mean)
            .flip_count
        )
    assert counts[False] < counts[True]
