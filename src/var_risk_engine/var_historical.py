"""Historical Value-at-Risk estimation."""

from __future__ import annotations

import numpy as np


def historical_var(
    portfolio_returns: np.ndarray,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> float:
    """Estimate Value-at-Risk using the historical simulation method.

    Sorts the empirical return distribution and extracts the (1 - confidence)
    quantile as the VaR estimate, then scales to the requested holding period
    via the square-root-of-time rule.

    Args:
        portfolio_returns: 1-D array of daily portfolio log-returns or simple
            returns.  Shape ``(n_days,)``.
        confidence: One-sided confidence level (e.g. 0.95 means the 5th
            percentile of losses).  Must be in (0, 1).
        holding_period: Number of trading days to scale to.  Uses the
            square-root-of-time rule: ``VaR(T) = VaR(1) * sqrt(T)``.

    Returns:
        A positive float representing the estimated VaR loss at the given
        confidence level over the specified holding period.

    Raises:
        ValueError: If *confidence* is not in (0, 1) or *holding_period* < 1.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if holding_period < 1:
        raise ValueError(f"holding_period must be >= 1, got {holding_period}")

    portfolio_returns = np.asarray(portfolio_returns, dtype=float).ravel()

    # (1 - confidence) quantile of the return distribution (a negative number
    # for typical confidence levels).
    var_1d = -np.quantile(portfolio_returns, 1.0 - confidence)

    # Square-root-of-time scaling.
    var_scaled = var_1d * np.sqrt(holding_period)

    return float(var_scaled)


def historical_var_series(
    portfolio_returns: np.ndarray,
    confidence: float = 0.95,
    window: int = 250,
) -> np.ndarray:
    """Compute rolling-window Historical VaR (useful for backtesting).

    For each window of *window* consecutive returns the historical VaR is
    calculated, producing a time-series of VaR estimates that can be compared
    against actual P&L for backtesting purposes.

    Args:
        portfolio_returns: 1-D array of daily portfolio returns.
            Shape ``(n_days,)``.  Must have at least *window* observations.
        confidence: One-sided confidence level.  Must be in (0, 1).
        window: Length of the rolling look-back window in trading days.
            Must be >= 2.

    Returns:
        1-D ``np.ndarray`` of shape ``(n_days - window + 1,)`` containing the
        Historical VaR estimate for each window.  The first element
        corresponds to the window ending at index ``window - 1`` of the
        original series.

    Raises:
        ValueError: If *window* is larger than the length of
            *portfolio_returns* or is less than 2.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    portfolio_returns = np.asarray(portfolio_returns, dtype=float).ravel()

    if window > len(portfolio_returns):
        raise ValueError(
            f"window ({window}) exceeds the length of portfolio_returns "
            f"({len(portfolio_returns)})"
        )

    n_obs = len(portfolio_returns)
    n_windows = n_obs - window + 1
    var_series = np.empty(n_windows, dtype=float)

    for i in range(n_windows):
        window_slice = portfolio_returns[i : i + window]
        var_series[i] = -np.quantile(window_slice, 1.0 - confidence)

    return var_series
