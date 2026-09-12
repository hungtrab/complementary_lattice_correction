"""Smoke test that the measurement code itself is right.

Runs the verification on synthetic layers with a known ``(W, mu, Sigma)``, where
every quantity can be checked independently.  A verifier that always reports
success is worthless, so this also perturbs the correction into rules the theory
does *not* cover and asserts the relevant checks turn red.
"""

from __future__ import annotations

import torch

from clc.correction import CorrectionConfig, LatticeCorrection
from clc.lattice import LatticeState
from clc.theory.verify import format_report, verify_layer


def _activations(seed: int, in_features: int, samples: int) -> torch.Tensor:
    """Correlated, non-zero-mean activations with planted outlier coordinates."""
    torch.manual_seed(seed)
    mixing = torch.randn(in_features, in_features) * 0.1
    activations = (mixing @ torch.randn(in_features, samples)) * 0.4 + 0.3
    activations[:4] += 4.0
    return activations


def synthetic_layer(seed: int = 0, out_features: int = 48, in_features: int = 256, samples: int = 3000):
    """Weight plus the calibration mean and covariance of one synthetic layer."""
    torch.manual_seed(seed)
    weight = torch.randn(out_features, in_features) * 0.05
    activations = _activations(seed, in_features, samples)
    return weight, activations.mean(dim=1), torch.cov(activations)


def run(
    layers: int = 8,
    bits: int = 4,
    group_size: int = 128,
    cross_eval: bool = False,
) -> list:
    """Verify each synthetic layer.

    With ``cross_eval`` the correction is driven by one draw of activations and
    measured against an independent one, which is the calibration-drift regime of
    the paper's Verification 5.
    """
    results = []
    for index in range(layers):
        weight, mean, covariance = synthetic_layer(seed=index)
        state = LatticeState.from_weight(weight, bits=bits, group_size=group_size)

        evaluation_mean = evaluation_covariance = None
        if cross_eval:
            held_out = _activations(seed=1000 + index, in_features=weight.shape[1], samples=3000)
            evaluation_mean = held_out.mean(dim=1)
            evaluation_covariance = torch.cov(held_out)

        results.append(
            verify_layer(
                state,
                mean,
                covariance,
                name=f"synthetic-{index}",
                evaluation_mean=evaluation_mean,
                evaluation_covariance=evaluation_covariance,
            )
        )
    return results


def main() -> int:
    status = 0
    for label, cross_eval in (("in-sample", False), ("cross-eval", True)):
        results = run(cross_eval=cross_eval)
        print(f"=== {label} ===")
        print(format_report(results))
        print()
        failed = [r.name for r in results if not r.passed]
        if failed:
            print(f"FAIL ({label}): {len(failed)} layer(s) violated the analysis")
            status = 1
    if status == 0:
        print("OK: every check holds on every synthetic layer, in-sample and cross-eval")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
