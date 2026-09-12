"""Packing helpers for the ``compressed-tensors`` WNA16 format.

The compressed-tensors packer stores signed integer codes in int32 words.  An
asymmetric affine quantizer still has a zero point, but both codes and zero
points are shifted by ``2 ** (bits - 1)`` before packing so that the on-disk
representation is signed int8-like values.  This is the convention used by
the compressed-tensors runtime and is algebraically equivalent to our
unsigned affine lattice:

    (q - z) * scale == ((q - offset) - (z - offset)) * scale
"""

from __future__ import annotations

from typing import Tuple

import torch

from clc.lattice import LatticeState


SUPPORTED_BITS = (4, 8)


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BITS:
        raise ValueError(
            "compressed-tensors WNA16 export supports only 4- and 8-bit "
            f"weights, got {bits}"
        )


def _pack_unsigned(values: torch.Tensor, bits: int, packed_dim: int) -> torch.Tensor:
    """Pack unsigned ``bits``-wide values into signed int32 words."""

    _validate_bits(bits)
    if packed_dim not in (0, 1):
        raise ValueError(f"packed_dim must be 0 or 1, got {packed_dim}")
    values = values.to(torch.int64)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(values.shape)}")

    factor = 32 // bits
    if packed_dim == 0:
        values = values.transpose(0, 1)
    size = values.shape[1]
    padded = (size + factor - 1) // factor * factor
    if padded != size:
        values = torch.nn.functional.pad(values, (0, padded - size))

    values = values.reshape(values.shape[0], -1, factor)
    mask = (1 << bits) - 1
    shifts = torch.arange(factor, device=values.device, dtype=torch.int64) * bits
    packed = ((values & mask) << shifts).sum(dim=-1).to(torch.int32)
    if packed_dim == 0:
        packed = packed.transpose(0, 1)
    return packed.contiguous()


def _unpack_unsigned(
    packed: torch.Tensor,
    bits: int,
    shape: Tuple[int, int],
    packed_dim: int,
) -> torch.Tensor:
    """Inverse of :func:`_pack_unsigned`, trimmed to the original shape."""

    _validate_bits(bits)
    if packed_dim not in (0, 1):
        raise ValueError(f"packed_dim must be 0 or 1, got {packed_dim}")
    if packed.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(packed.shape)}")

    factor = 32 // bits
    work = packed.to(torch.int64)
    if packed_dim == 0:
        work = work.transpose(0, 1)
    mask = (1 << bits) - 1
    shifts = torch.arange(factor, device=work.device, dtype=torch.int64) * bits
    values = ((work.unsqueeze(-1) >> shifts) & mask).reshape(work.shape[0], -1)
    if packed_dim == 0:
        values = values.transpose(0, 1)
    return values[: shape[0], : shape[1]].to(torch.int32).contiguous()


def pack_signed(values: torch.Tensor, bits: int, packed_dim: int) -> torch.Tensor:
    """Pack signed values using the compressed-tensors offset convention."""

    _validate_bits(bits)
    offset = 1 << (bits - 1)
    return _pack_unsigned(values.to(torch.int64) + offset, bits, packed_dim)


def unpack_signed(
    packed: torch.Tensor,
    bits: int,
    shape: Tuple[int, int],
    packed_dim: int,
) -> torch.Tensor:
    """Unpack signed compressed-tensors values."""

    _validate_bits(bits)
    offset = 1 << (bits - 1)
    return _unpack_unsigned(packed, bits, shape, packed_dim).to(torch.int64) - offset


def pack_layer_compressed_tensors(state: LatticeState, name: str):
    """Convert one lattice state to compressed-tensors parameter names."""

    # Imported lazily to avoid a module cycle: checkpoint.py dispatches here
    # only after defining PackedLayer.
    from clc.export.checkpoint import ExportError, PackedLayer

    _validate_bits(state.bits)
    if state.in_features != state.padded_in_features:
        raise ExportError(
            f"{name}: compressed-tensors export does not support padded input "
            "dimensions"
        )
    if state.unfolded_scale is not None:
        raise ExportError(
            f"{name}: channel scales are not folded into the lattice weights; "
            "run the AWQ pipeline with fold_channel_scales=True"
        )
    if state.in_features % state.group_size != 0:
        raise ExportError(
            f"{name}: input dimension {state.in_features} must be divisible by "
            f"group size {state.group_size}"
        )

    codes = state.codes_int()
    # Symmetric states already use signed codes and do not carry a zero point.
    if state.symmetric:
        packed = pack_signed(codes, state.bits, packed_dim=1)
        tensors = {
            "weight_packed": packed,
            "weight_scale": state.step_groups.to(torch.float16).contiguous(),
            "weight_shape": torch.tensor(
                [state.codes.shape[0], state.in_features], dtype=torch.int32
            ),
        }
    else:
        offset = 1 << (state.bits - 1)
        packed = pack_signed(codes - offset, state.bits, packed_dim=1)
        zero = state.zero_groups.to(torch.int64) - offset
        tensors = {
            "weight_packed": packed,
            "weight_scale": state.step_groups.to(torch.float16).contiguous(),
            "weight_zero_point": pack_signed(
                zero, state.bits, packed_dim=0
            ),
            "weight_shape": torch.tensor(
                [state.codes.shape[0], state.in_features], dtype=torch.int32
            ),
        }
    return PackedLayer(tensors=tensors)


def unpack_layer_compressed_tensors(
    tensors: dict[str, torch.Tensor], bits: int, symmetric: bool
):
    """Return lattice-shaped unsigned codes, scales, and zero points.

    This is intentionally an exporter verification helper rather than a model
    loader.  It normalizes the signed runtime representation back to the
    representation used by :class:`LatticeState`.
    """

    from clc.export.checkpoint import ExportError

    shape_tensor = tensors.get("weight_shape")
    if shape_tensor is None:
        raise ExportError("compressed-tensors layer is missing weight_shape")
    shape_values = [int(value) for value in shape_tensor.reshape(-1).tolist()]
    if len(shape_values) != 2:
        raise ExportError(f"invalid compressed-tensors weight_shape: {shape_values}")
    shape = (shape_values[0], shape_values[1])

    signed_codes = unpack_signed(
        tensors["weight_packed"], bits, shape, packed_dim=1
    )
    offset = 1 << (bits - 1)
    scales = tensors["weight_scale"]
    if symmetric:
        codes = signed_codes
        zeros = torch.zeros(
            (shape[0], scales.shape[1]), dtype=torch.int32, device=scales.device
        )
    else:
        codes = signed_codes + offset
        packed_zero = tensors.get("weight_zero_point")
        if packed_zero is None:
            raise ExportError("asymmetric compressed-tensors layer is missing weight_zero_point")
        zero_shape = (shape[0], scales.shape[1])
        zeros = unpack_signed(packed_zero, bits, zero_shape, packed_dim=0) + offset
        zeros = zeros.to(torch.int32)
    return codes.to(torch.int32), scales, zeros


__all__ = [
    "SUPPORTED_BITS",
    "pack_signed",
    "unpack_signed",
    "pack_layer_compressed_tensors",
    "unpack_layer_compressed_tensors",
]
