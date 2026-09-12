"""Bit-packing corrected integer codes into deployable checkpoint layouts.

CLC only ever changes ``q``; the step ``s``, the zero point ``z`` and the lattice
geometry are exactly those of the base quantizer.  Since every packed weight-only
format stores precisely ``(q, s, z)`` under the same ``w = (q - z) * s``
convention used by :class:`~clc.lattice.LatticeState`, the corrected codes can be
written straight into a checkpoint -- there is no re-quantization step and
therefore no round-trip error.

Note the layout transpose: PyTorch stores a linear weight as ``[out, in]`` while
the packed formats index ``[in, out]``, i.e. ``[K, N]``.
"""

from __future__ import annotations

import torch

# The AWQ kernels read nibbles in this order within each int32 word.
AWQ_INTERLEAVE = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_INTERLEAVE_8BIT = (0, 2, 1, 3)


def pack_factor(bits: int) -> int:
    """How many codes fit in one int32 word."""
    if 32 % bits != 0:
        raise ValueError(f"bits must divide 32, got {bits}")
    return 32 // bits


def _pack_along_columns(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``[rows, cols]`` codes into ``[rows, cols // pack_factor]`` int32."""
    rows, cols = codes.shape
    factor = pack_factor(bits)
    if cols % factor != 0:
        raise ValueError(f"column count {cols} is not divisible by pack factor {factor}")

    values = codes.to(torch.int64) & ((1 << bits) - 1)
    packed = torch.zeros(rows, cols // factor, dtype=torch.int64, device=codes.device)
    for slot in range(factor):
        packed |= values[:, slot::factor] << (bits * slot)
    # Wrap into the signed int32 range the checkpoint formats use.
    return (packed - (packed >= 2**31).to(torch.int64) * 2**32).to(torch.int32).contiguous()


def pack_awq(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``[K, N]`` codes into the AWQ ``qweight`` layout ``[K, N // factor]``.

    AWQ interleaves the nibbles inside each word so its dequantization kernel can
    unpack with a single shift sequence; the interleave must match exactly or the
    weights come back permuted.
    """
    if bits == 4:
        interleave = AWQ_INTERLEAVE
    elif bits == 8:
        interleave = AWQ_INTERLEAVE_8BIT
    else:
        raise ValueError(f"AWQ packing supports 4 or 8 bits, got {bits}")

    _, cols = codes.shape
    index = torch.tensor(interleave, dtype=torch.long, device=codes.device)
    interleaved = codes.reshape(-1, len(interleave)).index_select(1, index).reshape(-1, cols)
    return _pack_along_columns(interleaved, bits)


def unpack_awq(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_awq`, for verifying a written checkpoint."""
    if bits == 4:
        interleave = AWQ_INTERLEAVE
    elif bits == 8:
        interleave = AWQ_INTERLEAVE_8BIT
    else:
        raise ValueError(f"AWQ packing supports 4 or 8 bits, got {bits}")

    factor = pack_factor(bits)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, bits, device=packed.device, dtype=torch.int64)
    values = (words.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    values = values.reshape(packed.shape[0], -1)

    cols = values.shape[1]
    inverse = torch.empty(len(interleave), dtype=torch.long, device=packed.device)
    inverse[torch.tensor(interleave, dtype=torch.long, device=packed.device)] = torch.arange(
        len(interleave), dtype=torch.long, device=packed.device
    )
    return values.reshape(-1, len(interleave)).index_select(1, inverse).reshape(-1, cols)


GPTQ_SUPPORTED_BITS = (2, 3, 4, 8)
"""Bit widths vLLM's GPTQ kernels accept.

This matters for CLC because the AWQ format is 4-bit only and compressed-tensors
WNA16 covers 4 and 8 bits, so the GPTQ layout is the only deployable route for
the 3-bit setting where the correction helps most.
"""


def _gptq_block_layout(bits: int) -> list[tuple[int, int, int]]:
    """Where each of 32 consecutive codes lands inside a ``bits``-word block.

    GPTQ packs 32 codes along the input dimension into ``bits`` int32 words,
    treated as one little-endian ``32 * bits``-bit stream in which code ``j``
    occupies bits ``[j * bits, (j + 1) * bits)``.  When ``bits`` does not divide
    32 a code straddles a word boundary, which is what the hand-written 3-bit
    packing loops in AutoGPTQ spell out case by case.

    Returns one ``(word, offset, spill)`` triple per code, where ``spill`` is the
    number of bits that continue into the next word (0 when aligned).
    """
    layout = []
    for code in range(32):
        start = code * bits
        word, offset = divmod(start, 32)
        spill = max(0, offset + bits - 32)
        layout.append((word, offset, spill))
    return layout


def pack_gptq(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``[K, N]`` codes into the GPTQ ``qweight`` layout.

    GPTQ packs along the input dimension, producing ``[K * bits // 32, N]``.
    """
    if bits not in GPTQ_SUPPORTED_BITS:
        raise ValueError(f"GPTQ packing supports {GPTQ_SUPPORTED_BITS} bits, got {bits}")

    size_k, size_n = codes.shape
    if size_k % 32 != 0:
        raise ValueError(f"input dimension {size_k} must be a multiple of 32 for GPTQ packing")

    blocks = size_k // 32
    values = (codes.to(torch.int64) & ((1 << bits) - 1)).reshape(blocks, 32, size_n)
    packed = torch.zeros(blocks, bits, size_n, dtype=torch.int64, device=codes.device)

    for code, (word, offset, spill) in enumerate(_gptq_block_layout(bits)):
        packed[:, word] |= values[:, code] << offset
        if spill:
            packed[:, word + 1] |= values[:, code] >> (bits - spill)

    packed = packed.reshape(blocks * bits, size_n) & 0xFFFFFFFF
    return (packed - (packed >= 2**31).to(torch.int64) * 2**32).to(torch.int32).contiguous()


def unpack_gptq(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_gptq`."""
    if bits not in GPTQ_SUPPORTED_BITS:
        raise ValueError(f"GPTQ packing supports {GPTQ_SUPPORTED_BITS} bits, got {bits}")

    rows, size_n = packed.shape
    blocks = rows // bits
    words = (packed.to(torch.int64) & 0xFFFFFFFF).reshape(blocks, bits, size_n)
    mask = (1 << bits) - 1

    values = torch.zeros(blocks, 32, size_n, dtype=torch.int64, device=packed.device)
    for code, (word, offset, spill) in enumerate(_gptq_block_layout(bits)):
        chunk = words[:, word] >> offset
        if spill:
            chunk |= words[:, word + 1] << (bits - spill)
        values[:, code] = chunk & mask

    return values.reshape(blocks * 32, size_n)
