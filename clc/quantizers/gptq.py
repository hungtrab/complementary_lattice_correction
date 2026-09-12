"""GPTQ: Hessian-aware sequential updates before lattice assignment (Appendix G.2).

GPTQ quantizes columns left to right and, after each one, pushes the resulting
error into the remaining full-precision columns using ``H^-1``.  The value that
actually gets rounded is therefore not the original weight but a compensated
pre-round value ``w_tilde = w + c``.  Crucially GPTQ changes *what* is rounded,
not the lattice: the stored codes still lie on the same uniform per-group grid,
so CLC still applies.

That split is what this module encodes, and it is the reason
:class:`~clc.lattice.LatticeState` keeps ``float_weights`` and ``pre_round``
separately:

* the mean-shift target is measured against the **original** weight,
  ``b_j = mu_hat^T (w_q,j - w_j)``, because that is the error the deployed model
  actually makes;
* the admissible flip direction follows GPTQ's **own** rounding residual
  ``r = w_tilde / s + z - q``, because that is the local geometry its
  second-order update established.

Which guarantee survives: the first-moment descent of Proposition 1 still holds,
since the prefix search keeps the no-flip option.  The worst-case MSE bound of
Theorem 1 does not transfer verbatim -- it assumes the nearest-level residual
bound ``|e| <= s/2``, which GPTQ's compensated weights may violate.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from clc.lattice import LatticeState
from clc.quantizers.base import LayerQuantizer, QuantConfig, QuantizedLayer
from clc.statistics import ActivationStatistics


class GPTQQuantizer(LayerQuantizer):
    """Sequential Hessian-compensated quantization onto a group-wise lattice."""

    def __init__(
        self,
        config: Optional[QuantConfig] = None,
        damping: float = 0.01,
        block_size: int = 128,
    ):
        super().__init__(config)
        self.damping = damping
        self.block_size = block_size

    @property
    def name(self) -> str:
        return "gptq"

    @torch.no_grad()
    def _inverse_hessian(self, hessian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Damped ``H`` and the upper Cholesky factor of ``H^-1``.

        Columns with no activation energy are dead: they cannot be compensated
        against, so they are pinned to an identity row and their weights zeroed.
        """
        hessian = hessian.clone().to(torch.float32)
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1.0

        damp = self.damping * torch.mean(torch.diag(hessian))
        hessian += torch.eye(hessian.shape[0], device=hessian.device) * damp

        inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
        return dead, torch.linalg.cholesky(inverse, upper=True)

    @torch.no_grad()
    def quantize(self, module: nn.Linear, stats: Optional[ActivationStatistics]) -> QuantizedLayer:
        weight = module.weight.data
        out_features, in_features = weight.shape
        device, original_dtype = weight.device, weight.dtype

        if stats is None or stats.count == 0:
            state = self.base_lattice(weight)
            zeros = torch.zeros(in_features, device=device)
            return QuantizedLayer(state=state, activation_mean=zeros)

        # The lattice geometry (steps, zero points, padding) comes from the
        # original weight; GPTQ only changes which level each weight lands on.
        lattice = self.base_lattice(weight)
        step, zero = lattice.step, lattice.zero_point
        padded_in = lattice.padded_in_features

        working = torch.zeros(out_features, padded_in, dtype=torch.float32, device=device)
        working[:, :in_features] = weight.float()

        hessian = stats.gram.to(device, torch.float32)
        if hessian.shape[0] != padded_in:
            padded_hessian = torch.zeros(padded_in, padded_in, dtype=torch.float32, device=device)
            padded_hessian[:in_features, :in_features] = hessian
            hessian = padded_hessian

        dead, cholesky = self._inverse_hessian(hessian)
        working[:, dead] = 0.0

        codes = torch.zeros_like(working)
        pre_round = torch.zeros_like(working)

        for start in range(0, padded_in, self.block_size):
            end = min(start + self.block_size, padded_in)
            block = working[:, start:end].clone()
            block_error = torch.zeros_like(block)
            block_cholesky = cholesky[start:end, start:end]

            for offset in range(end - start):
                column = start + offset
                compensated = block[:, offset]
                diagonal = block_cholesky[offset, offset]

                normalized = compensated / step[:, column] + zero[:, column]
                code = normalized.round().clamp(lattice.min_code, lattice.max_code)

                codes[:, column] = code
                pre_round[:, column] = normalized

                dequantized = (code - zero[:, column]) * step[:, column]
                error = (compensated - dequantized) / diagonal
                block[:, offset:] -= error.unsqueeze(1) * block_cholesky[offset, offset:].unsqueeze(0)
                block_error[:, offset] = error

            working[:, end:] -= block_error @ cholesky[start:end, end:]

        # float_weights stays the ORIGINAL weight: that is what b_j is measured
        # against (Appendix G.2), even though pre_round is GPTQ's own state.
        lattice.codes = codes
        lattice.pre_round = pre_round

        mean = stats.mean.to(device, original_dtype)
        variance = stats.pooled_variance.to(device, original_dtype)
        return QuantizedLayer(
            state=lattice,
            activation_mean=mean,
            pooled_variance=variance,
            info={"damping": self.damping, "dead_columns": int(dead.sum())},
        )
