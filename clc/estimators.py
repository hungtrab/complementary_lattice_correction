"""Stabilized activation-mean estimate and the admissible-support mask.

These are the two stabilization components CLC puts in front of the greedy rule
(paper Section 3.4): a James-Stein shrunk activation mean, and a knee-point mask
that removes activation-outlier coordinates from the admissible support.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


@torch.no_grad()
def pooled_activation_variance(
    second_moment: torch.Tensor,
    sample_mean: torch.Tensor,
) -> torch.Tensor:
    """``sigma^2`` of Eq. (7): the pooled per-coordinate activation variance.

    The paper defines ``sigma^2 = (d*m)^-1 sum_{i,t} (X_{i,t} - xbar_i)^2``, which
    is exactly the mean over coordinates of the per-coordinate variance.  Given
    ``E[x^2]`` and ``E[x]`` accumulated over the calibration set this is free:
    ``Var_i = E[x_i^2] - xbar_i^2``, then average over ``i``.

    Args:
        second_moment: ``E[x_i^2]`` per input coordinate, shape ``[d]``.
        sample_mean: ``xbar_i`` per input coordinate, shape ``[d]``.

    Returns:
        Scalar tensor ``sigma^2``, clamped to be non-negative.
    """
    per_coord = (second_moment - sample_mean.pow(2)).clamp(min=0.0)
    return per_coord.mean()


@torch.no_grad()
def james_stein_mean(
    sample_mean: torch.Tensor,
    pooled_variance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """James-Stein shrinkage of the calibration activation mean, Eq. (7).

        mu_hat = xbar_g * 1 + (1 - c) * (xbar - xbar_g * 1)
        c      = clip( (d - 2) * sigma^2 / ||xbar - xbar_g * 1||_2^2 , 0, 1 )

    Shrinking toward the common center ``xbar_g`` stabilizes the coordinate-wise
    extremes that both the per-channel shift ``b_j`` and the support mask are
    sensitive to.  Clipping prevents over-shrinkage and recovers the sample mean
    when the shrinkage estimate degenerates.

    Args:
        sample_mean: ``xbar``, shape ``[d]``.
        pooled_variance: ``sigma^2`` from :func:`pooled_activation_variance`.  If
            omitted, falls back to a spread estimate of the channel means
            themselves -- this is the legacy smart-flip behaviour and is *not*
            what Eq. (7) specifies; pass the pooled variance whenever the
            calibration second moment is available.

    Returns:
        ``mu_hat``, same shape and dtype as ``sample_mean``.
    """
    d = sample_mean.numel()
    if d < 3:
        return sample_mean.clone()

    grand = sample_mean.mean()
    deviation = sample_mean - grand
    sum_sq_deviation = deviation.pow(2).sum()
    if sum_sq_deviation < 1e-10:
        return sample_mean.clone()

    if pooled_variance is None:
        variance = (deviation.abs().mean() ** 2).clamp(min=1e-8)
    else:
        variance = pooled_variance.to(sample_mean.dtype).clamp(min=1e-12)

    # Eq. (7) uses sigma^2 directly rather than the sampling variance sigma^2/m
    # of the mean; kept literal so the estimator matches the paper.
    shrinkage = ((d - 2) * variance / sum_sq_deviation).clamp(0.0, 1.0)
    return grand + (1.0 - shrinkage) * deviation


def _knee_index(values_desc: np.ndarray) -> int:
    """Index of the maximum-distance-to-chord point of a descending curve."""
    n = values_desc.shape[0]
    if n < 3:
        return max(n - 1, 0)
    lo, hi = float(values_desc.min()), float(values_desc.max())
    if hi - lo < 1e-10:
        return n // 2
    y = (values_desc - lo) / (hi - lo)
    x = np.linspace(0.0, 1.0, n)
    chord = y[0] + (y[-1] - y[0]) * x
    return int(np.argmax(np.abs(y - chord)))


@torch.no_grad()
def knee_threshold(
    activation_mean: torch.Tensor,
    tolerance: float = 0.0,
) -> Tuple[float, torch.Tensor]:
    """Knee-point support mask on ``|mu_hat|`` (paper Section 3.4).

    Sort ``a_i = |mu_hat_i|`` in descending order and apply a Kneedle-style rule
    to the *upper half* of the sorted curve: after normalization, connect the
    largest value ``a_(1)`` to the mid-rank value ``a_(floor(d/2))`` and take the
    point furthest from that line.  Its value is the threshold ``tau``, and the
    admissible support is ``I = {i : |mu_hat_i| <= tau}``.

    Coordinates above ``tau`` are activation outliers.  Eq. (3) shows a flip
    there can overshoot, because the quadratic term ``mu_i^2 s_j^2`` grows faster
    than the linear mean-shift decrease ``2|b_j||mu_i|s_j``.  Excluding them also
    reduces the covariance penalty ``||Sigma_II||_2`` in Theorem 1, since
    large-mean coordinates in LLM activations tend to coincide with
    high-variance directions.

    Args:
        activation_mean: ``mu_hat``, shape ``[d]``.
        tolerance: shifts the knee ``tolerance * d`` ranks further down the
            descending curve.  Because the curve is descending this *lowers*
            ``tau`` and therefore *narrows* the support -- larger values mask
            more coordinates as outliers.  Retained as a sweepable knob from the
            reference implementation; the paper uses the bare knee.

    Returns:
        ``(tau, support_mask)`` where ``support_mask`` is a bool tensor ``[d]``
        that is True on the admissible support ``I``.
    """
    magnitude = activation_mean.abs()
    d = magnitude.numel()
    if d < 3:
        return float("inf"), torch.ones_like(magnitude, dtype=torch.bool)

    sorted_desc, _ = torch.sort(magnitude, descending=True)
    half = max(d // 2, 3)
    upper = sorted_desc[:half].detach().float().cpu().numpy()

    knee = _knee_index(upper)
    if tolerance > 0.0:
        knee = min(half - 1, knee + int(tolerance * d))

    tau = float(sorted_desc[knee].item())
    return tau, magnitude <= tau
