"""Common interface: a base quantizer hands CLC a lattice plus its coordinates.

Appendix G is the reason this interface exists.  CLC always performs the same
adjacent-level move on an integer lattice, but different PTQ pipelines reach
that lattice through different coordinate systems and different pre-rounding
states.  A quantizer is therefore responsible for reporting three things:

* the lattice itself (codes, step, zero point),
* the activation mean *in the coordinate system that lattice lives in* -- the
  transformed mean ``mu' = S^-1 mu`` for AWQ (G.1), the plain mean for RTN,
* the pre-round state that defines which adjacent level is admissible, which for
  GPTQ (G.2) and AdaRound (G.4) is not derived from the original weight.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from clc.lattice import LatticeState
from clc.statistics import ActivationStatistics


@dataclass
class QuantConfig:
    """Weight-quantization geometry shared by every backend."""

    bits: int = 4
    group_size: int = 128
    symmetric: bool = False


@dataclass
class QuantizedLayer:
    """One quantized linear layer, ready for correction and for export.

    Attributes:
        state: the integer lattice.
        activation_mean: ``xbar`` in the lattice's coordinate system.
        pooled_variance: ``sigma^2`` of Eq. (7), same coordinate system.
        channel_scale: the per-input-channel transform ``alpha`` that was folded
            into the weight, or ``None`` when the coordinates are untransformed.
            Export needs this to know whether the scale still has to be absorbed
            elsewhere in the graph.
        info: free-form per-layer diagnostics for the run metadata.
    """

    state: LatticeState
    activation_mean: torch.Tensor
    pooled_variance: Optional[torch.Tensor] = None
    channel_scale: Optional[torch.Tensor] = None
    info: Dict = field(default_factory=dict)


class LayerQuantizer(ABC):
    """Builds a :class:`QuantizedLayer` from a linear module and its statistics."""

    def __init__(self, config: Optional[QuantConfig] = None):
        self.config = config or QuantConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in run names and metadata."""

    @abstractmethod
    def quantize(self, module: nn.Linear, stats: Optional[ActivationStatistics]) -> QuantizedLayer:
        """Project ``module``'s weight onto the integer lattice."""

    def base_lattice(self, weight: torch.Tensor) -> LatticeState:
        """Nearest-level group-wise projection using this quantizer's geometry."""
        return LatticeState.from_weight(
            weight,
            bits=self.config.bits,
            group_size=self.config.group_size,
            symmetric=self.config.symmetric,
        )


def writeback(module: nn.Linear, layer: QuantizedLayer) -> None:
    """Install the (possibly corrected) quantized weight back into the module.

    When a per-input-channel transform was applied but *not* folded into the
    preceding op, it has to be divided out here so the module still computes the
    original function.  That factorization is mathematically equivalent but
    leaves the stored weight off the uniform lattice, which is why such a layer
    cannot be packed -- see :mod:`clc.export`.
    """
    weight = layer.state.weight()
    if layer.channel_scale is not None and not layer.info.get("scale_folded", False):
        weight = (weight / layer.channel_scale.unsqueeze(0)).to(layer.state.original_dtype)
    module.weight.data = weight
