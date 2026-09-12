"""Round-to-nearest: the plain integer-lattice baseline.

RTN is the setting the main analysis is stated in -- the stored codes are the
nearest-level projection of the original weights, so the residual bound
``|e_ij| <= s_j / 2`` that Lemma 1(ii) relies on holds by construction, and the
coordinate system is untransformed.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from clc.quantizers.base import LayerQuantizer, QuantizedLayer
from clc.statistics import ActivationStatistics


class RTNQuantizer(LayerQuantizer):
    """Nearest-level projection with per-channel, per-group affine scaling."""

    @property
    def name(self) -> str:
        return "rtn"

    def quantize(self, module: nn.Linear, stats: Optional[ActivationStatistics]) -> QuantizedLayer:
        state = self.base_lattice(module.weight.data)
        in_features = module.weight.shape[1]
        if stats is None:
            import torch

            mean = torch.zeros(in_features, device=module.weight.device)
            variance = None
        else:
            mean = stats.mean.to(module.weight.device)
            variance = stats.pooled_variance.to(module.weight.device)
        return QuantizedLayer(state=state, activation_mean=mean, pooled_variance=variance)
