"""AdaRound: learned per-weight rounding decisions (Appendix G.4).

AdaRound replaces nearest-level rounding with a learned binary choice per weight,

    q = floor(u) + h,     u = w / s + z,     h in {0, 1},

optimizing ``h`` through a continuous relaxation against a local reconstruction
loss.  The stored weights still lie on the same per-group lattice, so CLC still
applies -- but two things it normally relies on no longer hold.

First, the rounding residual is no longer bounded by ``s / 2``: AdaRound will
deliberately pick the farther of the two neighbouring levels when that improves
its objective.  Lemma 1(ii) assumes that bound, so the worst-case MSE
certificate of Theorem 1 does not transfer here.  The first-moment descent of
Proposition 1 is unaffected, because it only needs the prefix search to include
the no-flip option.

Second, "move toward the nearest level" is not the right adjacent move.  The
meaningful one is to *reopen the learned choice*, ``q' = floor(u) + (1 - h)``.
That falls out for free by reporting the continuous ``u`` as the pre-round state:
``sign(u - q)`` is ``+1`` exactly when ``h = 0`` and ``-1`` when ``h = 1``.  So a
CLC flip here reverses one of AdaRound's decisions rather than correcting a naive
one, and the mean-shift gain it predicts has to be weighed against the
reconstruction loss AdaRound had already optimized.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from clc.quantizers.base import LayerQuantizer, QuantConfig, QuantizedLayer
from clc.statistics import ActivationStatistics

ZETA, GAMMA = 1.1, -0.1
"""Stretch parameters of the rectified sigmoid, from the AdaRound paper."""


def _rectified_sigmoid(v: torch.Tensor) -> torch.Tensor:
    """Continuous relaxation of the binary rounding choice, clipped to [0, 1]."""
    return torch.clamp(torch.sigmoid(v) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


class AdaRoundQuantizer(LayerQuantizer):
    """Learns each weight's rounding direction against a layer reconstruction loss."""

    def __init__(
        self,
        config: Optional[QuantConfig] = None,
        iterations: int = 2000,
        learning_rate: float = 1e-3,
        regularization: float = 0.01,
        batch_size: int = 32,
        beta_range: tuple[float, float] = (20.0, 2.0),
        warmup: float = 0.2,
    ):
        super().__init__(config)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.beta_range = beta_range
        self.warmup = warmup

    @property
    def name(self) -> str:
        return "adaround"

    def _beta(self, step: int) -> float:
        """Anneal the rounding regularizer from flat to sharply bimodal."""
        start = int(self.warmup * self.iterations)
        if step < start:
            return self.beta_range[0]
        progress = (step - start) / max(1, self.iterations - start)
        high, low = self.beta_range
        return low + (high - low) * max(0.0, 1.0 - progress)

    def quantize(self, module: nn.Linear, stats: Optional[ActivationStatistics]) -> QuantizedLayer:
        weight = module.weight.data
        device, in_features = weight.device, weight.shape[1]

        lattice = self.base_lattice(weight)
        if stats is None or stats.count == 0:
            return QuantizedLayer(
                state=lattice, activation_mean=torch.zeros(in_features, device=device)
            )

        rows = stats.samples.to(device, torch.float32)
        step, zero = lattice.step, lattice.zero_point
        continuous = lattice.pre_round                      # u = w / s + z
        floor_codes = torch.floor(continuous)
        fraction = (continuous - floor_codes).clamp(1e-4, 1 - 1e-4)

        # Initialise so the relaxed choice starts at AdaRound's own initialisation,
        # h ~ frac(u), i.e. the nearest-level decision in expectation.
        alpha = torch.log(
            (fraction - GAMMA) / (ZETA - fraction)
        ).detach().requires_grad_(True)

        padded_rows = rows
        if lattice.padded_in_features > in_features:
            padded_rows = torch.zeros(
                rows.shape[0], lattice.padded_in_features, device=device, dtype=rows.dtype
            )
            padded_rows[:, :in_features] = rows

        reference = padded_rows @ lattice.float_weights.t()
        optimizer = torch.optim.Adam([alpha], lr=self.learning_rate)

        for step_index in range(self.iterations):
            indices = torch.randint(0, padded_rows.shape[0], (self.batch_size,), device=device)
            batch, target = padded_rows[indices], reference[indices]

            soft = _rectified_sigmoid(alpha)
            codes = (floor_codes + soft).clamp(lattice.min_code, lattice.max_code)
            reconstruction = ((codes - zero) * step)
            loss = (batch @ reconstruction.t() - target).pow(2).mean()

            beta = self._beta(step_index)
            loss = loss + self.regularization * (1 - (2 * soft - 1).abs().pow(beta)).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            hard = (alpha >= 0).to(continuous.dtype)
            lattice.codes = (floor_codes + hard).clamp(lattice.min_code, lattice.max_code)
            # pre_round stays the continuous u, so sign(u - q) reopens the learned
            # choice rather than pointing at the nearest level.

        return QuantizedLayer(
            state=lattice,
            activation_mean=stats.mean.to(device, weight.dtype),
            pooled_variance=stats.pooled_variance.to(device, weight.dtype),
            info={
                "learned_up_fraction": float(hard.mean().item()),
                "iterations": self.iterations,
            },
        )
