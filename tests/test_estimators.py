"""Eq. (7) shrinkage and the knee-point support mask."""

import torch

from clc.estimators import james_stein_mean, knee_threshold, pooled_activation_variance


def _activations(d=256, m=800, n_outliers=5, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(d, m) * 0.5
    X[:n_outliers] += 6.0
    return X


def test_pooled_variance_matches_the_definition():
    """sigma^2 = (d*m)^-1 sum_{i,t} (X_it - xbar_i)^2."""
    X = _activations()
    xbar = X.mean(dim=1)
    direct = (X - xbar.unsqueeze(1)).pow(2).mean()
    assert torch.allclose(pooled_activation_variance(X.pow(2).mean(dim=1), xbar), direct, atol=1e-5)


def test_shrinkage_pulls_toward_the_common_center():
    X = _activations()
    xbar = X.mean(dim=1)
    mu = james_stein_mean(xbar, pooled_activation_variance(X.pow(2).mean(dim=1), xbar))
    grand = xbar.mean()
    assert (mu - grand).norm() < (xbar - grand).norm()
    # Shrinkage is uniform, so the grand mean itself is preserved.
    assert torch.allclose(mu.mean(), grand, atol=1e-5)


def test_pooled_variance_shrinks_more_than_the_legacy_fallback():
    """The legacy estimator uses the spread of the channel means, not sigma^2."""
    X = _activations()
    xbar = X.mean(dim=1)
    grand = xbar.mean()
    paper = james_stein_mean(xbar, pooled_activation_variance(X.pow(2).mean(dim=1), xbar))
    legacy = james_stein_mean(xbar)
    assert (paper - grand).norm() < (legacy - grand).norm()


def test_shrinkage_is_a_noop_below_three_coordinates():
    x = torch.tensor([1.0, 2.0])
    assert torch.equal(james_stein_mean(x), x)


def test_degenerate_mean_is_returned_unchanged():
    x = torch.full((64,), 3.0)
    assert torch.allclose(james_stein_mean(x), x)


def test_knee_excludes_planted_activation_outliers():
    X = _activations(n_outliers=5)
    tau, support = knee_threshold(X.mean(dim=1))
    assert not support[:5].any()
    assert support[5:].sum() > 0.9 * (support.numel() - 5)
    assert tau > 0


def test_support_is_exactly_the_sub_threshold_set():
    X = _activations()
    mu = X.mean(dim=1)
    tau, support = knee_threshold(mu)
    assert torch.equal(support, mu.abs() <= tau)


def test_tolerance_narrows_the_support():
    """The curve is descending, so a later knee means a lower tau."""
    X = _activations()
    mu = X.mean(dim=1)
    tau_bare, bare = knee_threshold(mu, tolerance=0.0)
    tau_strict, strict = knee_threshold(mu, tolerance=0.05)
    assert tau_strict <= tau_bare
    assert strict.sum() <= bare.sum()
