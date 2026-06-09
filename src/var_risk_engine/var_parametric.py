"""Parametric (Variance-Covariance) Value-at-Risk estimation."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def parametric_var(
    portfolio_returns: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> float:
    """Estimate Value-at-Risk assuming normally distributed portfolio returns.

    Computes the sample mean and standard deviation of the return series,
    then applies the analytical normal-VaR formula:

        VaR = -(mu + z * sigma) * sqrt(holding_period)

    where *z* = ``norm.ppf(1 - confidence)`` is the negative z-score
    corresponding to the left tail.

    Args:
        portfolio_returns: 1-D array of daily portfolio returns (simple or
            log).  Shape ``(n_days,)``.
        confidence: One-sided confidence level (e.g. 0.95).  Must be in
            (0, 1).
        holding_period: Number of trading days to scale to via the
            square-root-of-time rule.

    Returns:
        A positive float representing the estimated VaR loss.

    Raises:
        ValueError: If *confidence* is not in (0, 1) or *holding_period* < 1.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if holding_period < 1:
        raise ValueError(f"holding_period must be >= 1, got {holding_period}")

    portfolio_returns = np.asarray(portfolio_returns, dtype=float).ravel()

    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns, ddof=1)

    # z is negative for confidence > 0.5.
    z = norm.ppf(1.0 - confidence)

    var_1d = -(mu + z * sigma)
    var_scaled = var_1d * np.sqrt(holding_period)

    return float(var_scaled)


def parametric_var_from_cov(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
    expected_returns: np.ndarray | None = None,
) -> float:
    """Compute parametric VaR directly from the asset covariance matrix.

    This implements the classic Variance-Covariance (a.k.a. delta-normal)
    approach:

        portfolio_variance = w^T @ Sigma @ w
        portfolio_sigma    = sqrt(portfolio_variance)
        portfolio_mu       = w^T @ mu   (if expected_returns provided, else 0)
        VaR = -(portfolio_mu + z * portfolio_sigma) * sqrt(holding_period)

    Args:
        weights: 1-D array of portfolio weights summing to 1.
            Shape ``(n_assets,)``.
        cov_matrix: Variance-covariance matrix of daily asset returns.
            Shape ``(n_assets, n_assets)``.  Must be symmetric and
            positive-semidefinite.
        confidence: One-sided confidence level.  Must be in (0, 1).
        holding_period: Number of trading days to scale to.
        expected_returns: Optional 1-D array of expected daily asset
            returns.  Shape ``(n_assets,)``.  If ``None``, the drift term
            is assumed to be zero (a common simplification for short
            horizons).

    Returns:
        A positive float representing the portfolio VaR loss.

    Raises:
        ValueError: If dimensions of *weights*, *cov_matrix*, and
            *expected_returns* are inconsistent, or if *cov_matrix* is not
            square.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if holding_period < 1:
        raise ValueError(f"holding_period must be >= 1, got {holding_period}")

    weights = np.asarray(weights, dtype=float).ravel()
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    n_assets = weights.shape[0]

    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"cov_matrix shape {cov_matrix.shape} is inconsistent with "
            f"weights length {n_assets}"
        )

    # Portfolio variance: w^T Sigma w
    portfolio_variance = weights @ cov_matrix @ weights
    portfolio_sigma = np.sqrt(portfolio_variance)

    # Portfolio expected return.
    if expected_returns is not None:
        expected_returns = np.asarray(expected_returns, dtype=float).ravel()
        if expected_returns.shape[0] != n_assets:
            raise ValueError(
                f"expected_returns length {expected_returns.shape[0]} does "
                f"not match n_assets {n_assets}"
            )
        portfolio_mu = weights @ expected_returns
    else:
        portfolio_mu = 0.0

    z = norm.ppf(1.0 - confidence)

    var_1d = -(portfolio_mu + z * portfolio_sigma)
    var_scaled = var_1d * np.sqrt(holding_period)

    return float(var_scaled)
