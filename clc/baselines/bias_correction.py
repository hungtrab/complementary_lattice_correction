"""Classical bias correction (Nagel et al., 2019) -- the comparison baseline.

BC removes the same first-moment shift CLC targets, but from the other side: it
leaves the quantized weights untouched and absorbs the shift into an additive
output bias,

    bias_j  <-  bias_j - mu^T e_j  =  bias_j - b_j.

Per channel that cancels ``b_j`` exactly, which is strictly better than what a
budgeted lattice correction can achieve on the mean-shift term alone.  The cost
is structural rather than numerical:

* modern LLMs are predominantly bias-free (LLaMA, Mistral), so BC has to *create*
  parameters that the architecture does not have, changing the state dict and the
  deployment graph;
* it leaves the covariance-weighted term ``V_l`` of Eq. (2) completely unchanged,
  because the weights are the same -- it compensates the measured shift without
  touching the quantized representation that produced it.

Empirically this is why BC improves perplexity inconsistently and can hurt
downstream accuracy while CLC does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from clc.lattice import LatticeState


@dataclass
class BiasCorrectionStats:
    bias_delta_norm: float = 0.0
    mean_shift_before: float = 0.0
    mean_shift_after: float = 0.0
    created_bias: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class BiasCorrection:
    """Cancels the per-channel first-moment shift with an output-side bias."""

    @torch.no_grad()
    def apply(
        self,
        module: nn.Linear,
        state: LatticeState,
        activation_mean: torch.Tensor,
    ) -> BiasCorrectionStats:
        """Add ``-b_j`` to ``module.bias``, creating the parameter if absent."""
        device = state.codes.device
        dtype = state.codes.dtype

        mean = activation_mean.to(device=device, dtype=dtype)
        if mean.numel() != state.in_features:
            raise ValueError(
                f"activation mean has {mean.numel()} entries, expected {state.in_features}"
            )
        if state.padded_in_features > state.in_features:
            padded = torch.zeros(state.padded_in_features, device=device, dtype=dtype)
            padded[: state.in_features] = mean
            mean = padded

        shift = (state.residual() * mean.unsqueeze(0)).sum(dim=1)

        created = module.bias is None
        if created:
            module.bias = nn.Parameter(
                torch.zeros(
                    module.out_features, device=module.weight.device, dtype=module.weight.dtype
                )
            )
        module.bias.data -= shift.to(module.bias.dtype).to(module.bias.device)

        return BiasCorrectionStats(
            bias_delta_norm=float(shift.norm().item()),
            mean_shift_before=float(shift.pow(2).sum().item()),
            # The weights did not move, so the residual -- and hence the shift the
            # weights themselves produce -- is unchanged; the bias merely offsets it.
            mean_shift_after=float(shift.pow(2).sum().item()),
            created_bias=created,
        )
