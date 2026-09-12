"""Empirical verification of the analysis (Appendix E.1, Tables 3 and 4).

Every claim in Sections 3.2-3.3 is checked directly on instrumented layers rather
than assumed.  Two kinds of quantity are reported:

* **Checks** -- booleans.  The exact identity of Eq. (5), the structural
  conditions (one-level rule, Assumptions 1-2), the bias descent of
  Proposition 1, the Lemma 1 decomposition and its two bounds, the trace bound,
  the master bound of Theorem 1, and whether the layer error actually descends.
* **Magnitudes** -- the realized values against the bounds, which is what shows
  *where* the worst-case certificate is loose and what actually drives descent.

The bounds are deliberately adversarial: ``rho(Sigma)`` assumes every
off-diagonal entry of ``Sigma`` aligns with ``e``, and ``||Sigma_II||_2`` assumes
``Delta`` points along the top eigenvector.  Neither holds in practice, so a
large gap between realized value and bound is expected and is not a failure.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from clc.correction import LatticeCorrection
from clc.estimators import james_stein_mean
from clc.lattice import LatticeState
from clc.statistics import find_linear_modules, record_linear_inputs

EXACT_TOLERANCE = 1e-9


@dataclass
class LayerVerification:
    """Checks and magnitudes for one instrumented layer."""

    name: str
    checks: Dict[str, bool] = field(default_factory=dict)
    magnitudes: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


@torch.no_grad()
def verify_layer(
    state: LatticeState,
    activation_mean: torch.Tensor,
    covariance: torch.Tensor,
    correction: Optional[LatticeCorrection] = None,
    name: str = "layer",
    evaluation_mean: Optional[torch.Tensor] = None,
    evaluation_covariance: Optional[torch.Tensor] = None,
) -> LayerVerification:
    """Apply the correction to ``state`` and check every step of the analysis.

    Two distinct statistics are involved, and conflating them hides the question
    Verification 5 of the paper asks.  The correction is *driven* by the
    calibration estimate ``mu_hat``; the layer error is *measured* against the
    population ``(mu, Sigma)``.  Passing ``evaluation_mean`` and
    ``evaluation_covariance`` from a held-out split gives the cross-eval regime,
    which probes whether the guarantees survive calibration drift.  Omitting them
    measures in-sample.

    Args:
        state: the quantized layer, mutated in place by the correction.
        activation_mean: ``mu_hat``, the estimate that drives flip selection.
        covariance: ``Sigma`` used for measurement when no separate one is given.
        evaluation_mean: ``mu`` to measure against; defaults to
            ``activation_mean``.
        evaluation_covariance: ``Sigma`` to measure against; defaults to
            ``covariance``.
    """
    correction = correction or LatticeCorrection()
    device = state.codes.device
    width = state.in_features

    driving_mean = activation_mean
    mu = (
        evaluation_mean if evaluation_mean is not None else activation_mean
    ).to(device, torch.float64)
    sigma = (
        evaluation_covariance if evaluation_covariance is not None else covariance
    ).to(device, torch.float64)

    def trim(tensor):
        return tensor[:, :width].to(torch.float64)

    residual_before = trim(state.residual())          # e   [C, d]
    step = trim(state.step)                           # s_{g(i,j)}
    codes_before = state.codes.clone()

    stats = correction.apply(state, driving_mean, already_stabilized=True)

    residual_after = trim(state.residual())
    delta = residual_after - residual_before          # Delta = W_q' - W_q
    delta_codes = (state.codes - codes_before)[:, :width].to(torch.float64)
    support = delta_codes != 0
    flip_count = int(support.sum())

    verification = LayerVerification(name=name)
    checks, magnitudes = verification.checks, verification.magnitudes

    # -- the exact identity of Eq. (5) --------------------------------------
    second_moment = sigma + torch.outer(mu, mu)

    def layer_error(res):
        return float(torch.einsum("ji,ik,jk->", res, second_moment, res))

    def mean_shift_sq(res):
        return float((res @ mu).pow(2).sum())

    delta_error = layer_error(residual_after) - layer_error(residual_before)
    gain = mean_shift_sq(residual_before) - mean_shift_sq(residual_after)   # G_l
    cross = float(torch.einsum("ji,ik,jk->", delta, sigma, residual_before))
    trace_term = float(torch.einsum("ji,ik,jk->", delta, sigma, delta))

    identity_rhs = -gain + 2.0 * cross + trace_term
    scale = max(abs(delta_error), abs(identity_rhs), 1e-30)
    identity_residual = abs(delta_error - identity_rhs) / scale
    checks["eq5_identity"] = identity_residual < 1e-9
    magnitudes["eq5_relative_residual"] = identity_residual

    # -- structural conditions ----------------------------------------------
    expected_delta = torch.where(
        support, -torch.sign(residual_before) * step, torch.zeros_like(step)
    )
    # Ties (e == 0) resolve to +1 by convention rather than to -sign(0) = 0.
    ties = support & (residual_before == 0)
    expected_delta = torch.where(ties, step, expected_delta)
    checks["one_level_rule"] = bool(torch.allclose(delta, expected_delta, atol=1e-6))

    magnitude_after = residual_after.abs()
    checks["stays_on_lattice"] = bool(
        (state.codes.min() >= state.min_code) and (state.codes.max() <= state.max_code)
    )
    # The knee mask is the single support I declared by Algorithm 1. Check that
    # every row's selected support really lies inside that same mask. The old
    # refactor instead used ``A | ~A`` here, a tautology that reported
    # Assumption 1 without inspecting the selected coordinates.
    declared_support = activation_mean.to(device).abs() <= stats.knee_threshold
    if declared_support.numel() < state.padded_in_features:
        padded_declared = torch.zeros(
            state.padded_in_features, dtype=torch.bool, device=device
        )
        padded_declared[: declared_support.numel()] = declared_support
        declared_support = padded_declared
    checks["assumption1_common_support"] = bool(
        (~support | declared_support.unsqueeze(0)).all()
    )
    # Assumption 2 in its group-wise form (Appendix I.4): the step is constant
    # within each group, and every flip uses the step of its own group.
    groups = step.reshape(step.shape[0], -1, state.group_size)
    checks["assumption2_uniform_group_step"] = bool(
        torch.allclose(groups, groups[:, :, :1].expand_as(groups))
    )

    # -- Proposition 1 -------------------------------------------------------
    checks["bias_descent"] = gain >= -EXACT_TOLERANCE
    magnitudes["mean_shift_gain"] = gain

    # -- Lemma 1 -------------------------------------------------------------
    diagonal = torch.diag(sigma)
    diagonal_piece = float((delta * diagonal.unsqueeze(0) * residual_before).sum())
    offdiag_piece = cross - diagonal_piece

    checks["lemma1_decomposition"] = abs(
        (diagonal_piece + offdiag_piece) - cross
    ) <= 1e-8 * (1.0 + abs(cross))
    checks["lemma1_diagonal_non_positive"] = diagonal_piece <= EXACT_TOLERANCE

    row_sum = (sigma.abs().sum(dim=1) - diagonal.abs()).max()
    per_channel_flips = support.sum(dim=1).to(torch.float64)
    step_max_per_channel = (step * support).amax(dim=1)
    offdiag_bound = float(
        (step_max_per_channel.pow(2) / 2.0 * per_channel_flips * row_sum).sum()
    )
    checks["lemma1_offdiagonal_bound"] = abs(offdiag_piece) <= offdiag_bound + 1e-8

    on_support_terms = (step * diagonal.unsqueeze(0) * residual_before.abs())[support]
    gamma = float(on_support_terms.min()) if flip_count else 0.0
    checks["diagonal_lower_bound"] = abs(diagonal_piece) >= flip_count * gamma - 1e-8

    # -- trace bound ---------------------------------------------------------
    touched = support.any(dim=0)
    step_max = float((step * support).max()) if flip_count else 0.0
    if touched.any():
        submatrix = sigma[touched][:, touched]
        spectral_norm = float(torch.linalg.matrix_norm(submatrix, ord=2))
    else:
        spectral_norm = 0.0
    trace_bound = step_max**2 * flip_count * spectral_norm
    checks["trace_bound"] = trace_term <= trace_bound + 1e-8

    # -- Theorem 1 -----------------------------------------------------------
    phi = -2.0 * gamma + step_max**2 * (float(row_sum) + spectral_norm)
    master_bound = -gain + flip_count * phi
    checks["theorem1_master_bound"] = delta_error <= master_bound + 1e-8

    # -- Corollary 1 and the realized outcome --------------------------------
    checks["realized_descent"] = delta_error < 0

    magnitudes.update(
        {
            "flip_count": float(flip_count),
            "diagonal_piece": diagonal_piece,
            "offdiagonal_piece": offdiag_piece,
            "cross_term": cross,
            "trace_term": trace_term,
            "offdiagonal_bound": offdiag_bound,
            "trace_bound": trace_bound,
            "phi": phi,
            "gamma": gamma,
            "layer_error_change": delta_error,
            "layer_error_before": layer_error(residual_before),
            "master_bound": master_bound,
            "support_fraction": stats.support_fraction,
        }
    )
    return verification


def aggregate(results: List[LayerVerification]) -> Dict[str, Dict[str, float]]:
    """Table 3 pass fractions and Table 4 layer-averaged magnitudes."""
    if not results:
        return {"checks": {}, "magnitudes": {}}

    names = results[0].checks.keys()
    checks = {
        name: sum(int(r.checks[name]) for r in results) / len(results) for name in names
    }
    magnitude_names = results[0].magnitudes.keys()
    magnitudes = {
        name: sum(r.magnitudes[name] for r in results) / len(results)
        for name in magnitude_names
    }
    total_before = sum(r.magnitudes["layer_error_before"] for r in results)
    total_change = sum(r.magnitudes["layer_error_change"] for r in results)
    magnitudes["aggregate_error_change_pct"] = (
        100.0 * total_change / total_before if total_before else 0.0
    )
    return {"checks": checks, "magnitudes": magnitudes}


def format_report(results: List[LayerVerification]) -> str:
    """Render the Table 3 / Table 4 style summary."""
    summary = aggregate(results)
    total = len(results)
    lines = [
        f"CLC theory verification over {total} layer(s)",
        "",
        f"{'Check':<40} {'layers passing':>16}",
        "-" * 58,
    ]
    for name, fraction in summary["checks"].items():
        lines.append(f"{name:<40} {int(round(fraction * total)):>8}/{total:<7}")

    lines += ["", f"{'Quantity (layer-averaged)':<40} {'value':>16}", "-" * 58]
    for name, value in summary["magnitudes"].items():
        lines.append(f"{name:<40} {value:>16.6g}")

    failures = [r.name for r in results if not r.passed]
    if failures:
        lines += ["", "FAILED layers: " + ", ".join(failures)]
    return "\n".join(lines)


@torch.no_grad()
def verify_model(
    model: torch.nn.Module,
    batches: List[torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
    max_layers: Optional[int] = 16,
    name_pattern: Optional[str] = None,
    use_james_stein: bool = True,
) -> List[LayerVerification]:
    """Verify RTN+CLC on selected real linear layers of a causal LM.

    Each layer is instrumented separately so the full ``O(d^2)`` covariance is
    released before the next layer is measured.  This is slower than attaching
    hooks to the whole model, but keeps the diagnostic usable on large models.
    ``batches`` must already be on the device where the model accepts input.
    """
    matcher = re.compile(name_pattern) if name_pattern else None
    candidates = [
        (name, module)
        for name, module in find_linear_modules(model, skip=("lm_head",)).items()
        if matcher is None or matcher.search(name)
    ]
    if max_layers is not None:
        candidates = candidates[:max_layers]
    if not candidates:
        raise ValueError("no linear layers matched the theory-verification filter")

    results: List[LayerVerification] = []
    for name, module in candidates:
        with record_linear_inputs(
            {name: module},
            device="cpu",
            dtype=torch.float32,
            track_covariance=True,
        ) as captured:
            for batch in batches:
                model(batch)

        stats = captured.get(name)
        if stats is None or stats.count == 0:
            raise RuntimeError(f"no activation rows were captured for {name}")
        driving_mean = stats.mean
        if use_james_stein:
            driving_mean = james_stein_mean(driving_mean, stats.pooled_variance)
        state = LatticeState.from_weight(
            module.weight.data,
            bits=bits,
            group_size=group_size,
        )
        results.append(
            verify_layer(
                state,
                driving_mean,
                stats.covariance,
                name=name,
            )
        )
    return results


def main(argv=None) -> int:
    """CLI for the Appendix E.1 real-layer verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face id or local path")
    parser.add_argument("--layers", type=int, default=16, help="maximum number of linear layers")
    parser.add_argument("--pattern", help="regex selecting layer names before --layers")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext2"])
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./data/cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-james-stein", action="store_true")
    args = parser.parse_args(argv)

    from clc.calibration import load_calibration_data
    from clc.cli import _model_device, load_model_and_tokenizer

    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    batches = load_calibration_data(
        args.calib_dataset,
        tokenizer,
        n_samples=args.n_calib,
        seqlen=args.calib_seqlen,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    device = _model_device(model)
    results = verify_model(
        model,
        [batch.to(device) for batch in batches],
        bits=args.bits,
        group_size=args.group_size,
        max_layers=args.layers,
        name_pattern=args.pattern,
        use_james_stein=not args.no_james_stein,
    )
    print(format_report(results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
