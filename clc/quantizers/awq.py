"""AWQ: activation-aware per-input-channel pre-scaling (Appendix G.1).

AWQ rescales the layer before quantization by a diagonal ``S = diag(alpha)``, so
the object that actually gets discretized is ``S W`` and the relevant activations
are ``S^-1 X``.  CLC is applied in that transformed coordinate system: the
lattice is the one of ``Q(SW)`` and the mean-shift is measured against
``mu' = S^-1 mu``.

Where the ``1/alpha`` factor lives at inference decides whether the result can be
deployed.  Folding it into the preceding op -- what upstream AWQ does -- leaves
the layer weight equal to ``dequant(Q(SW))``, which sits on a uniform group
lattice and packs directly.  Dividing it back out of the weight instead yields
an algebraically identical model whose stored weight has a per-column effective
step ``s / alpha_i``; that is off-lattice and cannot be packed.  Both are
supported; only the folded one is exportable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from clc.quantizers.base import LayerQuantizer, QuantConfig, QuantizedLayer
from clc.statistics import ActivationStatistics


class AWQQuantizer(LayerQuantizer):
    """Grid search over ``alpha = salience^a`` followed by lattice projection."""

    def __init__(
        self,
        config: Optional[QuantConfig] = None,
        n_grid: int = 20,
        max_search_rows: int = 2048,
    ):
        super().__init__(config)
        self.n_grid = n_grid
        self.max_search_rows = max_search_rows

    @property
    def name(self) -> str:
        return "awq"

    @torch.no_grad()
    def search_channel_scale(
        self, weight: torch.Tensor, stats: ActivationStatistics
    ) -> tuple[torch.Tensor, float, float]:
        """Pick ``alpha = E[x^2]^a`` minimising the layer reconstruction error.

        The objective is the true AWQ one, ``|| x W^T - (x S^-1)(Q(SW))^T ||^2``.
        Note ``(x S^-1) Q(SW)^T = x (Q(SW) S^-1)^T``, so it is the same number
        whether or not the scale ends up folded into the preceding op.
        """
        device = weight.device
        salience = stats.second_moment.to(device, weight.dtype).clamp(min=1e-5)

        rows = stats.samples.to(device, weight.dtype)
        if rows.shape[0] > self.max_search_rows:
            rows = rows[: self.max_search_rows]
        reference = rows @ weight.t()

        best = (torch.ones_like(salience), 0.0, float("inf"))
        for grid_index in range(self.n_grid + 1):
            exponent = grid_index / self.n_grid
            scale = salience.pow(exponent)
            state = self.base_lattice(weight * scale.unsqueeze(0))
            reconstructed = state.weight() / scale.unsqueeze(0)
            error = (reference - rows @ reconstructed.t()).pow(2).mean().item()
            if error < best[2]:
                best = (scale, exponent, error)
        return best

    @torch.no_grad()
    def quantize(self, module: nn.Linear, stats: Optional[ActivationStatistics]) -> QuantizedLayer:
        weight = module.weight.data
        if stats is None or stats.count == 0:
            state = self.base_lattice(weight)
            zeros = torch.zeros(weight.shape[1], device=weight.device)
            return QuantizedLayer(state=state, activation_mean=zeros, info={"alpha": 0.0})

        scale, exponent, error = self.search_channel_scale(weight, stats)
        state = self.base_lattice(weight * scale.unsqueeze(0))

        # Appendix G.1: replace (W, mu, Sigma) with (SW, S^-1 mu, S^-1 Sigma S^-1).
        device = weight.device
        transformed_mean = stats.mean.to(device, weight.dtype) / scale
        per_coordinate_variance = (
            stats.second_moment.to(device, weight.dtype) - stats.mean.to(device, weight.dtype).pow(2)
        ).clamp(min=0.0)
        transformed_variance = (per_coordinate_variance / scale.pow(2)).mean()

        return QuantizedLayer(
            state=state,
            activation_mean=transformed_mean,
            pooled_variance=transformed_variance,
            channel_scale=scale,
            info={"alpha": exponent, "search_error": error, "scale_folded": False},
        )


@torch.no_grad()
def fold_channel_scale(
    previous: nn.Module,
    scaled_linears: list[nn.Linear],
    scale: torch.Tensor,
) -> None:
    """Absorb ``1/alpha`` into ``previous`` so the scaled weights stay on-lattice.

    ``previous`` must produce the input of every module in ``scaled_linears``
    along the channel dimension ``alpha`` indexes -- an ``RMSNorm``/``LayerNorm``
    (elementwise weight) or a preceding ``nn.Linear`` (output rows).  After
    folding, the composition is unchanged while each scaled layer's stored weight
    equals ``dequant(Q(SW))``.
    """
    scale = scale.to(previous.weight.device, previous.weight.dtype)

    if previous.weight.dim() == 1:  # RMSNorm / LayerNorm elementwise affine
        if previous.weight.shape[0] != scale.shape[0]:
            raise ValueError(
                f"norm has {previous.weight.shape[0]} channels, scale has {scale.shape[0]}"
            )
        previous.weight.data = previous.weight.data / scale
        bias = getattr(previous, "bias", None)
        if bias is not None:
            bias.data = bias.data / scale
    elif previous.weight.dim() == 2:  # preceding nn.Linear: divide output rows
        if previous.weight.shape[0] != scale.shape[0]:
            raise ValueError(
                f"previous linear has {previous.weight.shape[0]} outputs, scale has {scale.shape[0]}"
            )
        previous.weight.data = previous.weight.data / scale.unsqueeze(1)
        bias = getattr(previous, "bias", None)
        if bias is not None:
            bias.data = bias.data / scale
    else:
        raise TypeError(f"cannot fold a channel scale into {type(previous).__name__}")

    for linear in scaled_linears:
        linear.weight.data = linear.weight.data * scale.unsqueeze(0).to(linear.weight.dtype)
