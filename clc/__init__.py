"""Complementary Lattice Correction (CLC).

A post-quantization correction that reduces the structured first-moment output
shift by flipping selected weights to adjacent integer levels, leaving the
lattice, the architecture and the inference graph unchanged.
"""

from clc.correction import CorrectionConfig, CorrectionStats, LatticeCorrection
from clc.deployment import ENGINE_SPECS, inspect_checkpoint
from clc.estimators import james_stein_mean, knee_threshold, pooled_activation_variance
from clc.lattice import LatticeState
from clc.pipeline import PipelineConfig, PipelineResult, QuantizationPipeline
from clc.quantizers.base import QuantConfig, QuantizedLayer
from clc.statistics import ActivationStatistics

__all__ = [
    "ActivationStatistics",
    "CorrectionConfig",
    "CorrectionStats",
    "ENGINE_SPECS",
    "LatticeCorrection",
    "LatticeState",
    "PipelineConfig",
    "PipelineResult",
    "QuantConfig",
    "QuantizationPipeline",
    "QuantizedLayer",
    "inspect_checkpoint",
    "james_stein_mean",
    "knee_threshold",
    "pooled_activation_variance",
]
