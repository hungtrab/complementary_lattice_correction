"""Streaming activation statistics for the layers being corrected.

CLC needs the per-input-coordinate mean ``xbar`` and, for Eq. (7), the pooled
variance ``sigma^2``.  Both follow from running sums, so activations never have
to be retained: the accumulator holds ``O(d)`` state per layer instead of the
``O(m*d)`` of a full activation cache.  The full covariance ``Sigma`` is
``O(d^2)`` and is only needed by the theory verification (Appendix E.1), so it
is opt-in.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class ActivationStatistics:
    """Running mean / second moment (and optionally covariance) of layer inputs."""

    def __init__(
        self,
        in_features: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        track_covariance: bool = False,
        sample_limit: int = 0,
        seed: int = 0,
    ):
        """
        Args:
            sample_limit: keep at most this many activation rows by reservoir
                sampling.  Needed by objectives that cannot be written in terms
                of the moments, such as the AWQ scale search; leave at 0 when
                only the moments are used.
        """
        self.in_features = in_features
        self.dtype = dtype
        self.device = torch.device(device)
        self.count = 0
        self.sample_limit = sample_limit
        self._reservoir: Optional[torch.Tensor] = (
            torch.zeros(sample_limit, in_features, dtype=dtype, device=self.device)
            if sample_limit > 0
            else None
        )
        self._reservoir_filled = 0
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._sum = torch.zeros(in_features, dtype=dtype, device=self.device)
        self._sum_sq = torch.zeros(in_features, dtype=dtype, device=self.device)
        self._outer: Optional[torch.Tensor] = (
            torch.zeros(in_features, in_features, dtype=dtype, device=self.device)
            if track_covariance
            else None
        )

    @torch.no_grad()
    def update(self, activations: torch.Tensor) -> None:
        """Accumulate a batch of layer inputs, shape ``[..., in_features]``."""
        rows = activations.reshape(-1, activations.shape[-1]).to(self.device, self.dtype)
        self.count += rows.shape[0]
        self._sum += rows.sum(dim=0)
        self._sum_sq += rows.pow(2).sum(dim=0)
        if self._outer is not None:
            self._outer += rows.t() @ rows
        if self._reservoir is not None:
            self._update_reservoir(rows)

    @torch.no_grad()
    def _update_reservoir(self, rows: torch.Tensor) -> None:
        """Uniform reservoir sample of the rows seen so far."""
        seen_before = self.count - rows.shape[0]
        free = self.sample_limit - self._reservoir_filled
        if free > 0:
            take = min(free, rows.shape[0])
            self._reservoir[self._reservoir_filled : self._reservoir_filled + take] = rows[:take]
            self._reservoir_filled += take
            rows = rows[take:]
            seen_before += take
            if rows.shape[0] == 0:
                return

        # Row t (0-based, counting all rows ever seen) replaces a uniformly
        # chosen slot with probability sample_limit / (t + 1).
        indices = torch.arange(rows.shape[0], device="cpu") + seen_before
        keep = torch.rand(rows.shape[0], generator=self._generator) < (
            self.sample_limit / (indices + 1).float()
        )
        if not keep.any():
            return
        slots = torch.randint(
            0, self.sample_limit, (int(keep.sum()),), generator=self._generator
        )
        self._reservoir[slots.to(self.device)] = rows[keep.to(rows.device)]

    @property
    def samples(self) -> torch.Tensor:
        """The retained activation rows, shape ``[n, in_features]``."""
        if self._reservoir is None:
            raise RuntimeError("ActivationStatistics was built with sample_limit=0")
        return self._reservoir[: self._reservoir_filled]

    @property
    def mean(self) -> torch.Tensor:
        """``xbar``, the per-coordinate calibration mean."""
        if self.count == 0:
            return torch.zeros(self.in_features, dtype=self.dtype, device=self.device)
        return self._sum / self.count

    @property
    def second_moment(self) -> torch.Tensor:
        """``E[x_i^2]`` per coordinate."""
        if self.count == 0:
            return torch.zeros(self.in_features, dtype=self.dtype, device=self.device)
        return self._sum_sq / self.count

    @property
    def pooled_variance(self) -> torch.Tensor:
        """``sigma^2`` of Eq. (7)."""
        return (self.second_moment - self.mean.pow(2)).clamp(min=0.0).mean()

    @property
    def gram(self) -> torch.Tensor:
        """``E[x x^T]``, the uncentered second-moment matrix.

        This is the Hessian GPTQ's layer-wise objective uses (up to a constant
        factor that the relative damping makes irrelevant).
        """
        if self._outer is None:
            raise RuntimeError("ActivationStatistics was built with track_covariance=False")
        if self.count == 0:
            return torch.zeros_like(self._outer)
        return self._outer / self.count

    @property
    def covariance(self) -> torch.Tensor:
        """``Sigma = E[xx^T] - mu mu^T``.  Requires ``track_covariance=True``."""
        if self._outer is None:
            raise RuntimeError("ActivationStatistics was built with track_covariance=False")
        if self.count == 0:
            return torch.zeros_like(self._outer)
        mu = self.mean
        return self._outer / self.count - torch.outer(mu, mu)


@contextmanager
def record_linear_inputs(
    modules: Dict[str, nn.Module],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    track_covariance: bool = False,
    sample_limit: int = 0,
):
    """Attach input-recording hooks to named ``nn.Linear`` modules.

    Yields a dict mapping the same names to :class:`ActivationStatistics`.  Hooks
    are removed on exit, including when the body raises.
    """
    stats: Dict[str, ActivationStatistics] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, module: nn.Module):
        def hook(_module, inputs, _output):
            x = inputs[0]
            if name not in stats:
                stats[name] = ActivationStatistics(
                    x.shape[-1],
                    device=device,
                    dtype=dtype,
                    track_covariance=track_covariance,
                    sample_limit=sample_limit,
                )
            stats[name].update(x.detach())

        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(make_hook(name, module)))
        yield stats
    finally:
        for handle in handles:
            handle.remove()


def find_linear_modules(
    model: nn.Module,
    skip: Iterable[str] = ("lm_head",),
) -> Dict[str, nn.Module]:
    """All ``nn.Linear`` submodules, excluding names containing any ``skip`` token."""
    skip = tuple(skip)
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not any(token in name for token in skip)
    }
