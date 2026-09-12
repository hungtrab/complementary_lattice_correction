"""The Appendix E.1 verification, and proof that it can actually fail."""

import pytest
import torch

from clc.correction import CorrectionConfig, LatticeCorrection
from clc.lattice import LatticeState
from clc.theory import selftest
from clc.theory.verify import aggregate, format_report, verify_layer


@pytest.fixture(scope="module")
def layer():
    return selftest.synthetic_layer(seed=0)


def test_every_check_holds_on_synthetic_layers():
    results = selftest.run(layers=4)
    for result in results:
        failed = [name for name, ok in result.checks.items() if not ok]
        assert not failed, f"{result.name} violated {failed}"


def test_equation_five_is_an_exact_identity(layer):
    """Not an approximation: the -G + 2<Delta, Sigma e> + tr split is algebraic."""
    weight, mean, covariance = layer
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    result = verify_layer(state, mean, covariance)
    assert result.magnitudes["eq5_relative_residual"] < 1e-12


def test_the_diagonal_piece_dominates_the_off_diagonal_one(layer):
    """Lemma 1(i) is the mechanism that actually drives descent in practice."""
    weight, mean, covariance = layer
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    result = verify_layer(state, mean, covariance)
    assert result.magnitudes["diagonal_piece"] < 0
    assert abs(result.magnitudes["diagonal_piece"]) > abs(result.magnitudes["offdiagonal_piece"])


def test_the_worst_case_bounds_are_loose(layer):
    """Expected, and stated as such in the paper: they are adversarial."""
    weight, mean, covariance = layer
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    result = verify_layer(state, mean, covariance)
    assert abs(result.magnitudes["offdiagonal_piece"]) < result.magnitudes["offdiagonal_bound"]
    assert result.magnitudes["trace_term"] < result.magnitudes["trace_bound"]


def test_descent_is_realized_even_though_the_bound_only_certifies_non_increase(layer):
    weight, mean, covariance = layer
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    result = verify_layer(state, mean, covariance)
    assert result.magnitudes["layer_error_change"] < 0
    assert result.magnitudes["layer_error_change"] < result.magnitudes["master_bound"]


# -- the verifier must be able to fail ---------------------------------------

class _AlignedFlips(LatticeCorrection):
    """Deliberately breaks the one-level rule by moving *with* the residual.

    Moving along ``+sign(e)`` instead of against it increases the residual, so the
    structural check and the descent guarantee should both go red.
    """

    @torch.no_grad()
    def apply(self, state, activation_mean, pooled_variance=None, already_stabilized=False):
        stats = super().apply(state, activation_mean, pooled_variance, already_stabilized)
        # Undo the correct move and take the opposite one, twice over.
        state.codes = (state.codes + 2 * state.flip_direction()).clamp(
            state.min_code, state.max_code
        )
        return stats


def test_a_wrong_correction_is_detected(layer):
    weight, mean, covariance = layer
    state = LatticeState.from_weight(weight, bits=4, group_size=128)
    result = verify_layer(state, mean, covariance, correction=_AlignedFlips())

    assert not result.checks["one_level_rule"]
    assert not result.passed


def test_aggregate_reports_pass_fractions():
    results = selftest.run(layers=3)
    summary = aggregate(results)
    assert summary["checks"]["eq5_identity"] == 1.0
    assert "aggregate_error_change_pct" in summary["magnitudes"]
    assert summary["magnitudes"]["aggregate_error_change_pct"] < 0


def test_report_renders_every_check():
    report = format_report(selftest.run(layers=2))
    assert "theorem1_master_bound" in report
    assert "aggregate_error_change_pct" in report





# -- calibration drift (Verification 5) --------------------------------------

def test_guarantees_survive_calibration_drift():
    """Flips chosen on one draw of activations, error measured on another."""
    results = selftest.run(layers=4, cross_eval=True)
    for result in results:
        failed = [name for name, ok in result.checks.items() if not ok]
        assert not failed, f"{result.name} violated {failed} under cross-eval"


def test_cross_eval_descent_is_comparable_to_in_sample():
    """The mechanism is structural, so it should not depend on the split."""
    in_sample = aggregate(selftest.run(layers=4))["magnitudes"]
    cross_eval = aggregate(selftest.run(layers=4, cross_eval=True))["magnitudes"]

    assert cross_eval["layer_error_change"] < 0
    assert cross_eval["layer_error_change"] == pytest.approx(
        in_sample["layer_error_change"], rel=0.25
    )


def test_selftest_reports_both_regimes(capsys):
    assert selftest.main() == 0
    out = capsys.readouterr().out
    assert "in-sample" in out and "cross-eval" in out
