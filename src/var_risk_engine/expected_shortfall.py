"""Expected Shortfall (Conditional VaR / CVaR) — coherent risk measure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from var_risk_engine.var_historical import historical_var
from var_risk_engine.var_parametric import parametric_var


def expected_shortfall(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute Expected Shortfall (CVaR) from a historical return series.

    Expected Shortfall is the conditional expectation of losses given that
    the loss exceeds the VaR threshold.  Unlike VaR it is a *coherent* risk
    measure (satisfies sub-additivity).

        ES = -E[R | R <= -VaR]

    Args:
        returns: 1-D array of daily portfolio returns (simple or log).
            Shape ``(n_days,)``.
        confidence: One-sided confidence level (e.g. 0.95).  Must be in
            (0, 1).

    Returns:
        A positive float representing the average loss in the worst
        ``(1 - confidence) * 100`` percent of observations.

    Raises:
        ValueError: If *confidence* is not in (0, 1).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    returns = np.asarray(returns, dtype=float).ravel()

    var_threshold = -np.quantile(returns, 1.0 - confidence)

    # Tail losses: returns that fall at or below -VaR.
    tail = returns[returns <= -var_threshold]

    if tail.size == 0:
        # Degenerate case — no observations in the tail; return VaR itself.
        return float(var_threshold)

    return float(-np.mean(tail))


def es_historical(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Alias / wrapper for :func:`expected_shortfall`.

    Provides a naming convention that mirrors the ``historical_var`` /
    ``parametric_var`` split, making it straightforward to call the
    historical Expected Shortfall alongside its parametric counterpart.

    Args:
        returns: 1-D array of daily portfolio returns.  Shape ``(n_days,)``.
        confidence: One-sided confidence level.  Must be in (0, 1).

    Returns:
        A positive float — the historical ES estimate.
    """
    return expected_shortfall(returns, confidence=confidence)


def es_parametric(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Analytical Expected Shortfall under the normal distribution assumption.

    For a normal portfolio with mean *mu* and standard deviation *sigma*:

        ES = sigma * phi(z) / (1 - alpha) - mu

    where:
        * ``alpha`` = confidence level
        * ``z`` = ``norm.ppf(alpha)`` (the standard-normal quantile)
        * ``phi(z)`` = ``norm.pdf(z)`` (the standard-normal density at *z*)

    Args:
        returns: 1-D array of daily portfolio returns used to estimate *mu*
            and *sigma*.  Shape ``(n_days,)``.
        confidence: One-sided confidence level.  Must be in (0, 1).

    Returns:
        A positive float representing the parametric (normal) ES.

    Raises:
        ValueError: If *confidence* is not in (0, 1).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    returns = np.asarray(returns, dtype=float).ravel()

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    z = norm.ppf(confidence)
    phi_z = norm.pdf(z)

    es = sigma * phi_z / (1.0 - confidence) - mu

    return float(es)


def compare_var_es(
    returns: np.ndarray,
    confidence_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Build a comparison table of VaR and ES across multiple confidence levels.

    For each confidence level the function computes:

    * Historical VaR (via :func:`historical_var`)
    * Historical ES (via :func:`expected_shortfall`)
    * Parametric (normal) VaR (via :func:`parametric_var`)
    * Parametric (normal) ES (via :func:`es_parametric`)

    Args:
        returns: 1-D array of daily portfolio returns.  Shape ``(n_days,)``.
        confidence_levels: List of confidence levels to evaluate.  Defaults
            to ``[0.90, 0.95, 0.975, 0.99]``.

    Returns:
        A :class:`pandas.DataFrame` with columns:

        * ``confidence``
        * ``var_historical``
        * ``es_historical``
        * ``var_parametric``
        * ``es_parametric``

        Each row corresponds to one confidence level.

    Raises:
        ValueError: If any element of *confidence_levels* is not in (0, 1).
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.975, 0.99]

    returns = np.asarray(returns, dtype=float).ravel()

    rows: list[dict[str, float]] = []

    for cl in confidence_levels:
        if not 0 < cl < 1:
            raise ValueError(
                f"All confidence levels must be in (0, 1); got {cl}"
            )

        rows.append(
            {
                "confidence": cl,
                "var_historical": historical_var(returns, confidence=cl),
                "es_historical": expected_shortfall(returns, confidence=cl),
                "var_parametric": parametric_var(returns, confidence=cl),
                "es_parametric": es_parametric(returns, confidence=cl),
            }
        )

    return pd.DataFrame(rows)
