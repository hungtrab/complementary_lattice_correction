"""Complementary Lattice Correction -- Algorithm 1.

Given a quantized layer sitting on an integer lattice and an estimate of the
calibration activation mean, CLC selects a budgeted set of adjacent-level moves
that reduce the per-channel first-moment output shift

    b_j := mu_hat^T e_j,        e_j := (W_q - W)_{:,j}

without leaving the lattice.  Because the prefix search of Eq. (8) always
includes the empty selection ``k = 0``, the correction can never increase
``|b_j|``; summing over channels gives the non-negative mean-shift gain
``G_l = B_l(W_q) - B_l(W_q') >= 0`` (Proposition 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from clc.estimators import james_stein_mean, knee_threshold
from clc.lattice import LatticeState


@dataclass
class CorrectionConfig:
    """Hyperparameters of Algorithm 1.

    Attributes:
        budget_fraction: ``p`` in Eq. (8).  The per-channel cap is
            ``B_j = ceil(p * |I_j|)`` over the *filtered* candidate set, so ``p``
            limits how aggressively the greedy rule spends flips inside the
            reliable support.  It defines neither the flip direction nor the
            support itself.
        knee_tolerance: relaxation of the knee threshold ``tau``.
        use_james_stein: apply Eq. (7) shrinkage to the calibration mean.
        candidate_order: how to rank candidates within a channel (Section 3.4).
            ``"residual"`` ranks by ``delta_ij = |e_ij|`` in weight units, the
            literal reading of the paper.  ``"normalized"`` ranks by the
            code-space residual ``|r| = |e| / s``.  Under Assumption 2 -- one
            step per output channel -- the two orders are identical, which is
            the setting the paper's rationale is stated in ("post-flip residual
            ``s_j - |e_ij|`` smallest").  They diverge only under the group-wise
            extension of Appendix I.4, where a channel spans groups with
            different steps and the paper does not restate the ordering.  Since
            ``|e|`` correlates with the group step, ``"residual"`` front-loads
            large-impact flips and tends to stop the prefix search earlier.
        legacy: reproduce the pre-refactor smart-flip behaviour bit-for-bit
            (unconstrained prefix search truncated afterwards against ``p * d``,
            candidates ordered by the normalized residual ``|r|`` instead of
            ``|e|``, knee computed on the zero-padded mean).  Kept so previously
            reported numbers stay reproducible; not what the paper specifies.
    """

    budget_fraction: float = 0.05
    knee_tolerance: float = 0.0
    use_james_stein: bool = True
    candidate_order: str = "residual"
    legacy: bool = False

    def __post_init__(self):
        if self.candidate_order not in ("residual", "normalized"):
            raise ValueError(
                f"candidate_order must be 'residual' or 'normalized', got {self.candidate_order!r}"
            )


@dataclass
class CorrectionStats:
    """Per-layer diagnostics, including the quantities Theorem 1 is stated in."""

    knee_threshold: float = 0.0
    support_fraction: float = 0.0
    flip_count: int = 0                 # K_l
    mean_shift_before: float = 0.0      # B_l(W_q)  = sum_j b_j^2
    mean_shift_after: float = 0.0       # B_l(W_q')
    mean_shift_gain: float = 0.0        # G_l >= 0
    flips_per_channel_mean: float = 0.0
    flips_per_channel_max: float = 0.0
    channels_untouched_pct: float = 100.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class LatticeCorrection:
    """Applies Algorithm 1 to a :class:`~clc.lattice.LatticeState` in place."""

    def __init__(self, config: Optional[CorrectionConfig] = None):
        self.config = config or CorrectionConfig()

    def stabilize_mean(
        self,
        sample_mean: torch.Tensor,
        pooled_variance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Algorithm 1, lines 2-3: ``xbar -> mu_hat``."""
        if not self.config.use_james_stein:
            return sample_mean
        if self.config.legacy:
            return james_stein_mean(sample_mean, pooled_variance=None)
        return james_stein_mean(sample_mean, pooled_variance=pooled_variance)

    @torch.no_grad()
    def apply(
        self,
        state: LatticeState,
        activation_mean: torch.Tensor,
        pooled_variance: Optional[torch.Tensor] = None,
        already_stabilized: bool = False,
    ) -> CorrectionStats:
        """Correct ``state`` in place and return per-layer statistics.

        Args:
            state: the quantized layer.  Its ``codes`` are mutated.
            activation_mean: ``xbar`` over the calibration set, shape
                ``[in_features]``, in the coordinate system of ``state`` (for a
                transformation-based quantizer this is the transformed mean; see
                Appendix G).
            pooled_variance: ``sigma^2`` of Eq. (7); see
                :func:`clc.estimators.pooled_activation_variance`.
            already_stabilized: set when ``activation_mean`` is already
                ``mu_hat``, to avoid shrinking twice.
        """
        cfg = self.config
        device = state.codes.device
        dtype = state.codes.dtype

        mu = activation_mean.to(device=device, dtype=dtype)
        if mu.numel() != state.in_features:
            raise ValueError(
                f"activation mean has {mu.numel()} entries, expected {state.in_features}"
            )
        if not already_stabilized:
            mu = self.stabilize_mean(mu, pooled_variance)

        # Zero-pad the mean so padded columns can never contribute a flip.
        if state.padded_in_features > state.in_features:
            padded = torch.zeros(state.padded_in_features, device=device, dtype=dtype)
            padded[: state.in_features] = mu
            mu_padded = padded
        else:
            mu_padded = mu

        # -- lines 4-7: admissible support I = {i : |mu_hat_i| <= tau} --------
        knee_input = mu_padded if cfg.legacy else mu
        tau, support = knee_threshold(knee_input, tolerance=cfg.knee_tolerance)
        if support.numel() < state.padded_in_features:
            full = torch.zeros(state.padded_in_features, device=device, dtype=torch.bool)
            full[: state.in_features] = support
            support = full
        # Padded columns carry mu = 0 and are inert; keep them out of |I_j|.
        if state.padded_in_features > state.in_features:
            support = support.clone()
            support[state.in_features :] = False

        # -- line 10: e_j and b_j --------------------------------------------
        residual = state.residual()                                   # e   [C, d]
        mean_shift = (residual * mu_padded.unsqueeze(0)).sum(dim=1)   # b   [C]

        # -- lines 11-14: sigma and the per-flip shift reduction v ------------
        sigma = state.flip_direction()                                # [C, d]
        step = state.step                                             # s_{g(i,j)}
        shift_reduction = -mu_padded.unsqueeze(0) * sigma * step      # v   [C, d]

        # -- line 15: I_j = {i in I : sign(v_ij) = sign(b_j)} -----------------
        admissible = support.unsqueeze(0) & state.in_range(sigma)
        aligned = admissible & (torch.sign(shift_reduction) == torch.sign(mean_shift).unsqueeze(1))

        # -- line 18: order candidates by descending residual magnitude -------
        # See CorrectionConfig.candidate_order for why the two keys differ only
        # under group-wise quantization.
        if cfg.legacy or cfg.candidate_order == "normalized":
            order_key = state.normalized_residual().abs()
        else:
            order_key = residual.abs()
        order_key = order_key.masked_fill(~aligned, -1.0)
        order = torch.argsort(order_key, dim=1, descending=True)

        sorted_valid = torch.gather(aligned, 1, order)
        sorted_v = torch.gather(shift_reduction, 1, order) * sorted_valid.to(dtype)

        # -- lines 19-20: k* = argmin_{0<=k<=B_j} |b_j - sum_{t<=k} v| --------
        running = torch.cumsum(sorted_v, dim=1)
        residual_after = (mean_shift.unsqueeze(1) - running).abs()
        candidates = torch.cat([mean_shift.abs().unsqueeze(1), residual_after], dim=1)

        n_candidates = aligned.sum(dim=1)                             # |I_j|
        positions = torch.arange(candidates.shape[1], device=device).unsqueeze(0)

        if cfg.legacy:
            # Legacy: search the full range, then truncate against p * d.
            best_k = torch.argmin(candidates, dim=1)
        else:
            # Eq. (8): the search itself is capped at B_j = ceil(p * |I_j|).
            budget = torch.ceil(cfg.budget_fraction * n_candidates.to(dtype)).to(torch.long)
            candidates = candidates.masked_fill(positions > budget.unsqueeze(1), float("inf"))
            best_k = torch.argmin(candidates, dim=1)

        selected = (positions[:, : order.shape[1]] < best_k.unsqueeze(1)) & sorted_valid

        if cfg.legacy:
            max_flips = int(cfg.budget_fraction * state.in_features)
            within = selected.long().cumsum(dim=1) <= max_flips
            selected = selected & within

        # -- lines 21-23: apply the selected adjacent-level moves -------------
        sorted_sigma = torch.gather(sigma, 1, order) * selected.to(dtype)
        delta_codes = torch.zeros_like(state.codes)
        delta_codes.scatter_(1, order, sorted_sigma)
        state.apply_flips(delta_codes)

        return self._collect_stats(state, mu_padded, mean_shift, delta_codes, tau, support)

    @staticmethod
    def _collect_stats(
        state: LatticeState,
        mu_padded: torch.Tensor,
        mean_shift_before: torch.Tensor,
        delta_codes: torch.Tensor,
        tau: float,
        support: torch.Tensor,
    ) -> CorrectionStats:
        mean_shift_after = (state.residual() * mu_padded.unsqueeze(0)).sum(dim=1)
        before_sq = float(mean_shift_before.pow(2).sum().item())
        after_sq = float(mean_shift_after.pow(2).sum().item())

        flipped = delta_codes != 0
        per_channel = flipped.sum(dim=1).float()

        return CorrectionStats(
            knee_threshold=tau,
            support_fraction=float(support.float().mean().item()),
            flip_count=int(flipped.sum().item()),
            mean_shift_before=before_sq,
            mean_shift_after=after_sq,
            mean_shift_gain=before_sq - after_sq,
            flips_per_channel_mean=float(per_channel.mean().item()),
            flips_per_channel_max=float(per_channel.max().item()),
            channels_untouched_pct=float((per_channel == 0).float().mean().item() * 100.0),
        )
