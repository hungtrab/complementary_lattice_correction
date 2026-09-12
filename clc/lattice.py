"""The integer lattice a quantized layer lives on.

Notation follows the paper (Complementary Lattice Correction, Section 3.1).
The paper writes ``W in R^{d x C}`` with ``d`` input dimensions and ``C`` output
channels, so a layer output is ``x^T W_{:,j}``.  PyTorch stores ``nn.Linear``
weights transposed, as ``[out_features, in_features] = [C, d]``.  We keep the
PyTorch layout and use ``j`` for the row (output channel) and ``i`` for the
column (input dimension) throughout, so the paper's ``W_{i,j}`` is our
``weight[j, i]``.

Under group-wise quantization (Appendix I.4) each output channel is partitioned
into groups of ``group_size`` columns and weight ``(i, j)`` uses the step
``s_{g(i,j)}`` of its own group.  ``LatticeState`` therefore stores the steps
compactly per group and expands them on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


_DISCARDED = torch.empty(0)
"""Sentinel replacing tensors that :meth:`LatticeState.compact` released."""


def _expand_groups(compact: torch.Tensor, group_size: int, padded_in: int) -> torch.Tensor:
    """[C, n_groups] -> [C, padded_in], repeating each group value group_size times."""
    return compact.repeat_interleave(group_size, dim=1)[:, :padded_in]


def _round_step_up(step: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round the quantization step to ``dtype``, never downward.

    Storing the step at float16 is what the checkpoint formats do, but rounding
    it *down* shrinks the representable range so the group's extreme weight no
    longer fits: it clamps at the code boundary and its rounding residual creeps
    just past ``s / 2``.  Lemma 1(ii) needs ``|e_kj| <= s_j / 2`` at every
    coordinate, so the step is rounded toward positive infinity instead.
    """
    rounded = step.to(dtype)
    ceiling = torch.nextafter(
        rounded, torch.full_like(rounded, float("inf"))
    )
    return torch.where(rounded.to(step.dtype) < step, ceiling, rounded).to(step.dtype)


@dataclass
class LatticeState:
    """A quantized weight tensor together with the lattice it sits on.

    The dequantization convention is ``w = (q - z) * s`` for every backend, which
    is also the convention used by the AWQ / GPTQ / compressed-tensors packed
    checkpoint formats -- so corrected codes can be packed without a round trip
    through floating point.

    Attributes:
        codes: ``q``, the stored integer codes, shape ``[C, padded_in]``.  Held as
            a floating tensor for arithmetic convenience; integral by construction.
        step_groups: ``s``, shape ``[C, n_groups]``.
        zero_groups: ``z``, shape ``[C, n_groups]``.
        float_weights: ``W``, the full-precision weights the correction is scored
            against, shape ``[C, padded_in]``.
        pre_round: the un-rounded value in code space whose rounding produced
            ``codes``.  For nearest-level quantizers this is ``W / s + z``; for
            GPTQ (Appendix G.2) it is the Hessian-updated pre-round state, which
            is deliberately *not* derived from ``float_weights``.  It defines the
            admissible flip direction and the candidate ordering, while
            ``float_weights`` defines the mean-shift target.
    """

    codes: torch.Tensor
    step_groups: torch.Tensor
    zero_groups: torch.Tensor
    float_weights: torch.Tensor
    pre_round: torch.Tensor
    bits: int
    group_size: int
    symmetric: bool
    in_features: int
    padded_in_features: int
    original_dtype: torch.dtype
    unfolded_scale: Optional[torch.Tensor] = None
    """A per-input-channel transform that is *not* absorbed into the graph.

    When set, the module's stored weight is ``dequant(Q(SW)) / alpha`` while these
    codes describe ``Q(SW)``.  The two are different tensors, so packing the codes
    would ship a model that does not match the one that was measured.  Export
    refuses such a layer rather than silently producing a wrong checkpoint.
    """

    _step: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    _zero: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)

    # -- lattice geometry ---------------------------------------------------

    @property
    def min_code(self) -> int:
        return -(2 ** (self.bits - 1)) if self.symmetric else 0

    @property
    def max_code(self) -> int:
        return 2 ** (self.bits - 1) - 1 if self.symmetric else 2 ** self.bits - 1

    @property
    def n_groups(self) -> int:
        return self.step_groups.shape[1]

    @property
    def step(self) -> torch.Tensor:
        """``s_{g(i,j)}`` expanded to ``[C, padded_in]``."""
        if self._step is None:
            self._step = _expand_groups(self.step_groups, self.group_size, self.padded_in_features)
        return self._step

    @property
    def zero_point(self) -> torch.Tensor:
        """``z`` expanded to ``[C, padded_in]``."""
        if self._zero is None:
            self._zero = _expand_groups(self.zero_groups, self.group_size, self.padded_in_features)
        return self._zero

    # -- weights ------------------------------------------------------------

    def dequantize(self) -> torch.Tensor:
        """``W_q = (q - z) * s``, padded, in the working dtype."""
        codes = self.codes
        if not codes.is_floating_point():
            codes = codes.to(self.step_groups.dtype)
        return (codes - self.zero_point) * self.step

    def weight(self) -> torch.Tensor:
        """``W_q`` with padding removed and cast back to the module dtype."""
        w = self.dequantize()
        if self.padded_in_features > self.in_features:
            w = w[:, : self.in_features]
        return w.to(self.original_dtype)

    # -- quantities the correction is defined in terms of -------------------

    def _require_full(self, what: str) -> None:
        if self.compacted:
            raise RuntimeError(
                f"{what} needs the full-precision weight, which compact() released"
            )

    def residual(self) -> torch.Tensor:
        """``e = W_q - W`` (paper sign convention, Section 3.1).

        Note this is the *opposite* sign to the ``W - W_q`` used by the legacy
        smart-flip implementation; every downstream formula here assumes the
        paper's orientation.
        """
        self._require_full("residual()")
        return self.dequantize() - self.float_weights

    def normalized_residual(self) -> torch.Tensor:
        """``r = pre_round - q``, the rounding residual in code units.

        For nearest-level projection ``|r| <= 1/2`` and ``e = -r * s``.  Under
        GPTQ or AdaRound ``pre_round`` is method-specific, so this is the
        rounding state of that method rather than ``-e / s``.
        """
        self._require_full("normalized_residual()")
        return self.pre_round - self.codes

    def flip_direction(self) -> torch.Tensor:
        """``sigma = -sign(e)``, the one-level anti-residual move (Section 3.2).

        Computed from ``pre_round`` so that it follows the base quantizer's own
        rounding state (Appendix G.2/G.4).  Ties resolve to ``+1``.
        """
        sigma = torch.sign(self.normalized_residual())
        return torch.where(sigma == 0, torch.ones_like(sigma), sigma)

    def in_range(self, direction: torch.Tensor) -> torch.Tensor:
        """Whether ``q + direction`` stays inside the representable code range."""
        proposed = self.codes + direction
        return (proposed >= self.min_code) & (proposed <= self.max_code)

    # -- mutation -----------------------------------------------------------

    def apply_flips(self, delta_codes: torch.Tensor) -> None:
        """Add ``delta_codes`` to the stored codes in place, clamped to range."""
        self._require_full("apply_flips()")
        self.codes = (self.codes + delta_codes).clamp(self.min_code, self.max_code)

    def compact(self, device: torch.device | str = "cpu") -> "LatticeState":
        """Drop what only the correction needed, keeping what export needs.

        ``float_weights`` and ``pre_round`` are full-size float tensors used to
        score and direct the flips.  Once the correction has run only the codes,
        steps and zero points are needed, so retaining all three per layer costs
        three times the model in float32 -- prohibitive for a 7B model.  This
        frees them and stores the codes in the narrowest integer type that holds
        the code range, on ``device``.

        The state stays usable for :meth:`dequantize`, :meth:`weight` and export;
        :meth:`residual`, :meth:`normalized_residual` and :meth:`flip_direction`
        raise afterwards, since the quantities they need are gone.
        """
        fits_in_int8 = self.min_code >= -128 and self.max_code <= 127
        dtype = torch.int8 if fits_in_int8 else torch.int16
        self.codes = self.codes.round().to(dtype).to(device)
        self.step_groups = self.step_groups.to(device)
        self.zero_groups = self.zero_groups.to(device)
        self.float_weights = _DISCARDED
        self.pre_round = _DISCARDED
        self._step = None
        self._zero = None
        return self

    @property
    def compacted(self) -> bool:
        return self.float_weights is _DISCARDED

    def codes_int(self) -> torch.Tensor:
        """Unpadded integer codes, ready for packing into a checkpoint."""
        q = self.codes
        if self.padded_in_features > self.in_features:
            q = q[:, : self.in_features]
        return q.round().to(torch.int32)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = False,
        scale_dtype: Optional[torch.dtype] = torch.float16,
    ) -> "LatticeState":
        """Group-wise nearest-level projection of ``weight`` (the RTN lattice).

        ``weight`` is ``[C, d]`` in PyTorch layout.  Columns are zero-padded up to
        a multiple of ``group_size``; padding never contributes to a correction
        because its activation mean is zero.

        ``scale_dtype`` is the precision the step is *stored* at.  Every packed
        checkpoint format keeps scales in float16, so rounding the step to
        float16 before deriving the codes makes the in-process simulation
        bit-identical to what the deployment kernel computes -- otherwise the
        two differ by about one float16 ULP per weight and the perplexity you
        measure is not quite the perplexity you ship.  Pass ``None`` to keep full
        float32 steps.
        """
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")

        out_features, in_features = weight.shape
        n_groups = (in_features + group_size - 1) // group_size
        padded_in = n_groups * group_size

        original_dtype = weight.dtype
        w = weight.float()
        if padded_in > in_features:
            padded = torch.zeros(out_features, padded_in, dtype=w.dtype, device=w.device)
            padded[:, :in_features] = w
            w = padded

        grouped = w.reshape(out_features, n_groups, group_size)

        if symmetric:
            max_code = 2 ** (bits - 1) - 1
            min_code = -(2 ** (bits - 1))
            amax = grouped.abs().amax(dim=2)
            step = (amax / max_code).clamp(min=1e-8)
            zero = torch.zeros_like(step)
        else:
            max_code = 2 ** bits - 1
            min_code = 0
            w_min = grouped.amin(dim=2)
            w_max = grouped.amax(dim=2)
            step = ((w_max - w_min) / max_code).clamp(min=1e-8)
            zero = torch.round(-w_min / step).clamp(0, max_code)

        # Round the step to storage precision *before* deriving the codes, so the
        # lattice the correction operates on is the lattice that gets deployed.
        if scale_dtype is not None:
            step = _round_step_up(step, scale_dtype)

        step_flat = _expand_groups(step, group_size, padded_in)
        zero_flat = _expand_groups(zero, group_size, padded_in)

        pre_round = w / step_flat + zero_flat
        codes = torch.round(pre_round).clamp(min_code, max_code)

        return cls(
            codes=codes,
            step_groups=step,
            zero_groups=zero,
            float_weights=w,
            pre_round=pre_round,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            in_features=in_features,
            padded_in_features=padded_in,
            original_dtype=original_dtype,
        )
