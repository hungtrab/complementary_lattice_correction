"""Block-by-block quantization with CLC applied as the final stage.

The model is walked one decoder block at a time.  For each block the pipeline
records the inputs of its linear layers, quantizes and corrects them, and then
re-runs the block so the next one calibrates on already-quantized activations.
Only one block's statistics are alive at a time, and activations are summarised
into ``O(d)`` moments rather than cached, so peak memory is set by the model
rather than by the calibration set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from clc.baselines.bias_correction import BiasCorrection
from clc.correction import CorrectionConfig, LatticeCorrection
from clc.lattice import LatticeState
from clc.models import decoder_blocks, scale_groups
from clc.quantizers.adaround import AdaRoundQuantizer
from clc.quantizers.awq import AWQQuantizer, fold_channel_scale
from clc.quantizers.base import LayerQuantizer, QuantizedLayer, writeback
from clc.statistics import ActivationStatistics, find_linear_modules, record_linear_inputs


@dataclass
class PipelineConfig:
    """How the pipeline walks the model.

    Attributes:
        post_correction: ``"clc"``, ``"bias_correction"`` or ``"none"``.
        fold_channel_scales: fold AWQ's per-input-channel scale into the
            producing module.  Required for export; without it the stored weight
            carries a per-column effective step and cannot be packed.
        sample_limit: activation rows retained per layer, for objectives that
            need samples (the AWQ scale search).  0 keeps only moments.
        track_covariance: accumulate the full ``O(d^2)`` Gram matrix.  Needed by
            GPTQ and by the theory verification; off otherwise.
        compact_states: release each layer's full-precision weight and pre-round
            state once its correction has run, keeping only what export needs.
            Retaining all three costs three times the model in float32, which
            does not fit for a 7B model.  Turn off only to inspect the residuals
            afterwards.
        state_device: where compacted lattices are kept.  Off-GPU by default so
            accumulated states do not compete with the model for device memory.
    """

    post_correction: str = "clc"
    fold_channel_scales: bool = True
    sample_limit: int = 2048
    track_covariance: bool = False
    stats_dtype: torch.dtype = torch.float32
    skip_modules: tuple = ("lm_head",)
    compact_states: bool = True
    state_device: str = "cpu"


@dataclass
class PipelineResult:
    states: Dict[str, LatticeState] = field(default_factory=dict)
    layer_stats: Dict[str, dict] = field(default_factory=dict)
    exportable: bool = True
    notes: List[str] = field(default_factory=list)


class QuantizationPipeline:
    """Runs a base quantizer plus a post-correction over a causal LM."""

    def __init__(
        self,
        quantizer: LayerQuantizer,
        config: Optional[PipelineConfig] = None,
        correction_config: Optional[CorrectionConfig] = None,
    ):
        self.quantizer = quantizer
        self.config = config or PipelineConfig()
        self.correction = LatticeCorrection(correction_config or CorrectionConfig())
        self.bias_correction = BiasCorrection()

    # -- per-block work -----------------------------------------------------

    def _collect(self, block: nn.Module, run_block: Callable[[], None]) -> Dict[str, ActivationStatistics]:
        modules = find_linear_modules(block, skip=self.config.skip_modules)
        needs_samples = isinstance(self.quantizer, (AWQQuantizer, AdaRoundQuantizer))
        with record_linear_inputs(
            modules,
            device="cpu",
            dtype=self.config.stats_dtype,
            track_covariance=self.config.track_covariance,
            sample_limit=self.config.sample_limit if needs_samples else 0,
        ) as stats:
            run_block()
        return stats

    def _fold_block_scales(
        self, block: nn.Module, stats: Dict[str, ActivationStatistics], result: PipelineResult
    ) -> Dict[int, torch.Tensor]:
        """Search and fold one AWQ scale per producer/consumer group.

        The scale is shared by every consumer in the group, because they share an
        input and the producer can only be divided once.  It is searched on the
        concatenation of their weights, which is what upstream AWQ does.
        """
        folded: Dict[int, torch.Tensor] = {}
        groups = scale_groups(block)
        if not groups:
            result.exportable = False
            result.notes.append(
                f"{type(block).__name__}: unknown layout, AWQ scales left unfolded"
            )
            return folded

        lookup = {id(module): name for name, module in find_linear_modules(block, skip=()).items()}

        for group_name, producer, consumers in groups:
            reference = next(
                (stats[lookup[id(c)]] for c in consumers if lookup.get(id(c)) in stats), None
            )
            if reference is None or reference.count == 0:
                continue
            stacked = torch.cat([c.weight.data for c in consumers], dim=0)
            scale, _, _ = self.quantizer.search_channel_scale(stacked, reference)
            fold_channel_scale(producer, list(consumers), scale)
            for consumer in consumers:
                folded[id(consumer)] = scale
        return folded

    def _quantize_block(
        self,
        block: nn.Module,
        prefix: str,
        stats: Dict[str, ActivationStatistics],
        result: PipelineResult,
    ) -> None:
        folding = isinstance(self.quantizer, AWQQuantizer) and self.config.fold_channel_scales
        folded: Dict[int, torch.Tensor] = {}
        if folding:
            folded = self._fold_block_scales(block, stats, result)

        for name, module in find_linear_modules(block, skip=self.config.skip_modules).items():
            layer_stats = stats.get(name)
            if id(module) in folded:
                layer = self._folded_layer(module, layer_stats, folded[id(module)])
            elif folding:
                # No producer can absorb this layer's scale -- under grouped-query
                # attention, for instance, v_proj is narrower than o_proj's input.
                # Upstream AWQ leaves such layers unscaled; scaling them anyway
                # would put the stored weight off-lattice for no deployable gain.
                layer = self._unscaled_layer(module, layer_stats)
            else:
                layer = self.quantizer.quantize(module, layer_stats)
                if layer.channel_scale is not None:
                    # The transform was never absorbed into the graph, so the codes
                    # and the stored weight describe different tensors.
                    layer.state.unfolded_scale = layer.channel_scale
                    result.exportable = False

            record = dict(layer.info)
            if self.config.post_correction == "clc":
                record.update(
                    self.correction.apply(
                        layer.state, layer.activation_mean, layer.pooled_variance
                    ).as_dict()
                )

            writeback(module, layer)

            if self.config.post_correction == "bias_correction":
                record.update(
                    self.bias_correction.apply(
                        module, layer.state, layer.activation_mean
                    ).as_dict()
                )

            if self.config.compact_states:
                layer.state.compact(device=self.config.state_device)

            result.layer_stats[f"{prefix}.{name}"] = record
            result.states[f"{prefix}.{name}"] = layer.state

    def _unscaled_layer(
        self, module: nn.Linear, stats: Optional[ActivationStatistics]
    ) -> QuantizedLayer:
        """Plain lattice projection, for a layer whose scale cannot be folded."""
        device = module.weight.device
        state = self.quantizer.base_lattice(module.weight.data)
        if stats is None or stats.count == 0:
            mean = torch.zeros(module.weight.shape[1], device=device)
            variance = None
        else:
            mean = stats.mean.to(device, module.weight.dtype)
            variance = stats.pooled_variance.to(device, module.weight.dtype)
        return QuantizedLayer(
            state=state,
            activation_mean=mean,
            pooled_variance=variance,
            info={"channel_scale": "not foldable, layer left unscaled"},
        )

    def _folded_layer(
        self,
        module: nn.Linear,
        stats: Optional[ActivationStatistics],
        scale: torch.Tensor,
    ) -> QuantizedLayer:
        """Build the lattice for a layer whose channel scale is already folded.

        The scale search already ran in :meth:`_fold_block_scales` and the result
        is baked into ``module.weight``, which now holds ``S W``.  Re-running the
        quantizer here would search a second time on the already-scaled weight and
        scale it again, so the lattice is taken directly.  Appendix G.1: the
        statistics must move to the same coordinate system, ``mu' = S^-1 mu`` and
        ``Sigma' = S^-1 Sigma S^-1``.
        """
        device = module.weight.device
        state = self.quantizer.base_lattice(module.weight.data)

        if stats is None or stats.count == 0:
            mean = torch.zeros(module.weight.shape[1], device=device)
            variance = None
        else:
            scale = scale.to(device, module.weight.dtype)
            raw_mean = stats.mean.to(device, module.weight.dtype)
            mean = raw_mean / scale
            per_coordinate = (
                stats.second_moment.to(device, module.weight.dtype) - raw_mean.pow(2)
            ).clamp(min=0.0)
            variance = (per_coordinate / scale.pow(2)).mean()

        return QuantizedLayer(
            state=state,
            activation_mean=mean,
            pooled_variance=variance,
            channel_scale=scale,
            info={"scale_folded": True},
        )

    # -- whole model --------------------------------------------------------

    @torch.no_grad()
    def run(self, model: nn.Module, batches: List[torch.Tensor]) -> PipelineResult:
        """Quantize every decoder block, calibrating on ``batches`` of token ids."""
        result = PipelineResult()
        blocks = decoder_blocks(model)
        inputs, kwargs = _capture_block_inputs(model, blocks[0], batches)

        for index, block in enumerate(blocks):
            prefix = f"model.layers.{index}"

            def run_block():
                for sample in inputs:
                    block(sample, **kwargs)

            stats = self._collect(block, run_block)
            self._quantize_block(block, prefix, stats, result)

            # Propagate: the next block sees this block's quantized outputs, so
            # later layers calibrate against accumulated quantization error.
            inputs = [_block_output(block(sample, **kwargs)) for sample in inputs]

        if self.config.post_correction == "bias_correction":
            result.exportable = False
            result.notes.append(
                "bias correction adds output-side bias parameters; the packed "
                "formats store none, so this variant is measurement-only"
            )
        return result


def _block_output(returned) -> torch.Tensor:
    """The hidden state a decoder block produced.

    Transformers has changed this shape over versions: decoder layers used to
    return a tuple ``(hidden_states, ...)`` and now return the tensor directly.
    Indexing blindly would silently slice the batch dimension instead.
    """
    if torch.is_tensor(returned):
        return returned
    return returned[0]


class _Catcher(nn.Module):
    """Intercepts the first decoder block to capture its inputs, then stops."""

    def __init__(self, block: nn.Module, sink: list, kwargs: dict):
        super().__init__()
        self.block = block
        self.sink = sink
        self.kwargs = kwargs

    def forward(self, hidden_states, **kwargs):
        self.sink.append(hidden_states.detach())
        self.kwargs.update({k: v for k, v in kwargs.items() if k != "past_key_value"})
        raise _StopForward


class _StopForward(Exception):
    """Raised to abort the forward pass once the inputs have been captured."""


@torch.no_grad()
def _capture_block_inputs(model: nn.Module, first_block: nn.Module, batches):
    """Run the embedding stack only, to get what the first decoder block sees."""
    blocks = decoder_blocks(model)
    sink, kwargs = [], {}
    blocks[0] = _Catcher(first_block, sink, kwargs)
    try:
        for batch in batches:
            try:
                model(batch)
            except _StopForward:
                pass
    finally:
        blocks[0] = first_block
    if not sink:
        raise RuntimeError("no calibration inputs were captured")
    return sink, kwargs
