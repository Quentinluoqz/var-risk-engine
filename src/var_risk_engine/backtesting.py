"""VaR backtesting: violation counting, Kupiec POF test, Christoffersen test, Basel traffic light."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import chi2


def count_violations(returns: np.ndarray, var_forecasts: np.ndarray) -> np.ndarray:
    """Identify observations where the actual loss exceeds the VaR forecast.

    A violation occurs when the return is more negative than the (negative)
    VaR forecast, i.e. ``returns < -var_forecasts`` (assuming VaR is reported
    as a positive number representing a loss).

    Parameters
    ----------
    returns : np.ndarray
        Array of realised returns.
    var_forecasts : np.ndarray
        Array of VaR forecasts (positive numbers representing potential loss).

    Returns
    -------
    np.ndarray
        Boolean array of the same length as *returns*, ``True`` where a
        violation occurred.
    """
    returns = np.asarray(returns, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    return returns < -var_forecasts


def kupiec_pof_test(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Kupiec Proportion-of-Failures (POF) test for VaR backtesting.

    Tests whether the observed violation rate is statistically consistent
    with the expected rate implied by the VaR confidence level.

    The test statistic is:

        LR = -2 * log( ((1-p0)^(n-x) * p0^x) / ((1-p1)^(n-x) * p1^x) )

    where *n* is the number of observations, *x* is the number of violations,
    *p0 = 1 - confidence* is the expected violation probability, and
    *p1 = x / n* is the realised violation probability.

    Under the null hypothesis the statistic follows a chi-squared distribution
    with 1 degree of freedom.

    Parameters
    ----------
    returns : np.ndarray
        Array of realised returns.
    var_forecasts : np.ndarray
        Array of VaR forecasts (positive numbers).
    confidence : float, optional
        VaR confidence level (default 0.95).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"n_violations"`` : int -- number of observed violations
        - ``"n_obs"`` : int -- total number of observations
        - ``"expected_violations"`` : float -- expected number of violations
        - ``"actual_rate"`` : float -- realised violation rate
        - ``"test_statistic"`` : float -- Kupiec LR statistic
        - ``"p_value"`` : float -- p-value from chi-squared(1)
        - ``"reject_h0"`` : bool -- whether to reject at the 5 % level
    """
    violations = count_violations(returns, var_forecasts)
    n = len(violations)
    x = int(np.sum(violations))

    p0 = 1.0 - confidence
    p1 = x / n if n > 0 else 0.0

    expected_violations = n * p0

    # Guard against degenerate cases (all or no violations)
    if x == 0:
        # No violations at all -- compute a finite statistic
        test_statistic = -2.0 * (n * np.log(1.0 - p0) - n * np.log(1.0))
        # Simplify: -2*n*log(1-p0) when p1=0 => numerator has (1-p0)^n, denominator has 1^n
        test_statistic = -2.0 * n * np.log(1.0 - p0)
    elif x == n:
        # Every observation is a violation -- extreme case
        test_statistic = -2.0 * n * np.log(p0)
    else:
        numerator = (n - x) * np.log(1.0 - p0) + x * np.log(p0)
        denominator = (n - x) * np.log(1.0 - p1) + x * np.log(p1)
        test_statistic = -2.0 * (numerator - denominator)

    p_value = 1.0 - chi2.cdf(test_statistic, df=1)

    return {
        "n_violations": x,
        "n_obs": n,
        "expected_violations": expected_violations,
        "actual_rate": p1,
        "test_statistic": float(test_statistic),
        "p_value": float(p_value),
        "reject_h0": bool(p_value < 0.05),
    }


def christoffersen_test(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Christoffersen conditional coverage test for VaR backtesting.

    Tests both the correct unconditional coverage rate **and** the
    independence of violations (i.e. violations should not cluster).

    The method counts transitions between violation states:

    - ``00``: no violation followed by no violation
    - ``01``: no violation followed by a violation
    - ``10``: violation followed by no violation
    - ``11``: violation followed by a violation

    The likelihood-ratio statistic is:

        LR = -2 * (log L0 - log L1)

    where L0 assumes a single transition probability and L1 allows
    state-dependent probabilities.  Under H0, LR ~ chi-squared(2).

    Parameters
    ----------
    returns : np.ndarray
        Array of realised returns.
    var_forecasts : np.ndarray
        Array of VaR forecasts (positive numbers).
    confidence : float, optional
        VaR confidence level (default 0.95).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"n_violations"`` : int
        - ``"n_obs"`` : int
        - ``"test_statistic"`` : float
        - ``"p_value"`` : float
        - ``"reject_h0"`` : bool (at 5 % significance)
        - ``"n00"``, ``"n01"``, ``"n10"``, ``"n11"`` : int -- transition counts
    """
    violations = count_violations(returns, var_forecasts).astype(int)
    n = len(violations)
    x = int(np.sum(violations))

    # Count transitions
    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for t in range(n - 1):
        i_t = violations[t]
        i_t1 = violations[t + 1]
        if i_t == 0 and i_t1 == 0:
            n00 += 1
        elif i_t == 0 and i_t1 == 1:
            n01 += 1
        elif i_t == 1 and i_t1 == 0:
            n10 += 1
        else:  # 1 -> 1
            n11 += 1

    # Unrestricted (Markov) transition probabilities
    # pi_01 = P(violation at t+1 | no violation at t)
    # pi_11 = P(violation at t+1 | violation at t)
    n0_total = n00 + n01
    n1_total = n10 + n11

    if n0_total > 0:
        pi_01 = n01 / n0_total
    else:
        pi_01 = 0.0

    if n1_total > 0:
        pi_11 = n11 / n1_total
    else:
        pi_11 = 0.0

    # Restricted (unconditional) probability
    pi = (n01 + n11) / (n - 1) if (n - 1) > 0 else 0.0

    # Log-likelihood under H0 (single probability)
    log_l0 = 0.0
    if pi > 0 and pi < 1:
        n_no_violation_next = n00 + n10  # transitions ending in 0
        n_violation_next = n01 + n11     # transitions ending in 1
        log_l0 = n_no_violation_next * np.log(1.0 - pi) + n_violation_next * np.log(pi)
    elif pi == 0:
        log_l0 = 0.0  # all transitions to 0, log(1) = 0
    else:  # pi == 1
        log_l0 = 0.0  # all transitions to 1, log(1) = 0

    # Log-likelihood under H1 (Markov transition probabilities)
    log_l1 = 0.0
    if n00 > 0 and pi_01 < 1:
        log_l1 += n00 * np.log(1.0 - pi_01)
    if n01 > 0 and pi_01 > 0:
        log_l1 += n01 * np.log(pi_01)
    if n10 > 0 and pi_11 < 1:
        log_l1 += n10 * np.log(1.0 - pi_11)
    if n11 > 0 and pi_11 > 0:
        log_l1 += n11 * np.log(pi_11)

    test_statistic = -2.0 * (log_l0 - log_l1)
    # Clamp to zero in case of floating-point noise
    test_statistic = max(test_statistic, 0.0)

    p_value = 1.0 - chi2.cdf(test_statistic, df=2)

    return {
        "n_violations": x,
        "n_obs": n,
        "test_statistic": float(test_statistic),
        "p_value": float(p_value),
        "reject_h0": bool(p_value < 0.05),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def basel_traffic_light(n_violations: int, n_obs: int = 250) -> dict:
    """Classify VaR backtesting results using the Basel traffic-light approach.

    Zones (for a 250-day window at 99 % confidence):

    - **Green** (0--4 violations): Model is acceptable.
    - **Yellow** (5--9 violations): Cause for concern; a scaling factor
      is applied that increases linearly with the number of excess
      violations.
    - **Red** (10+ violations): Model failure; a significant penalty
      scaling factor is applied.

    Parameters
    ----------
    n_violations : int
        Number of VaR violations observed.
    n_obs : int, optional
        Number of observations in the backtesting window (default 250).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"zone"`` : str -- ``"green"``, ``"yellow"``, or ``"red"``
        - ``"n_violations"`` : int
        - ``"scaling_factor"`` : float -- multiplicative factor applied to VaR
        - ``"interpretation"`` : str -- human-readable explanation
    """
    if n_violations <= 4:
        zone = "green"
        scaling_factor = 1.0
        interpretation = (
            f"Green zone: {n_violations} violation(s) out of {n_obs} observations. "
            "The VaR model is performing within acceptable bounds."
        )
    elif n_violations <= 9:
        zone = "yellow"
        scaling_factor = 1.0 + 0.05 * (n_violations - 4)
        interpretation = (
            f"Yellow zone: {n_violations} violation(s) out of {n_obs} observations. "
            "The model shows signs of underestimating risk. A scaling factor of "
            f"{scaling_factor:.2f} should be applied to increase the VaR estimate."
        )
    else:
        zone = "red"
        scaling_factor = 1.3 + 0.05 * (n_violations - 9)
        interpretation = (
            f"Red zone: {n_violations} violation(s) out of {n_obs} observations. "
            "The VaR model has failed and must be revised. A scaling factor of "
            f"{scaling_factor:.2f} is applied as a punitive measure."
        )

    return {
        "zone": zone,
        "n_violations": n_violations,
        "scaling_factor": float(scaling_factor),
        "interpretation": interpretation,
    }


def rolling_backtest(
    returns: np.ndarray,
    var_func: Callable[[np.ndarray, float], float],
    confidence: float = 0.95,
    window: int = 250,
    holding_period: int = 1,
) -> dict:
    """Perform a rolling T+1 VaR backtesting exercise.

    For each day after the initial estimation *window*, the function:

    1. Estimates VaR using the trailing ``window`` observations via
       ``var_func``.
    2. Compares the forecast to the realised return on the next day
       (scaled by ``holding_period``).

    Parameters
    ----------
    returns : np.ndarray
        Full array of daily returns.
    var_func : callable
        A function with signature ``var_func(returns_window, confidence)``
        that returns a single VaR value (positive number representing loss).
    confidence : float, optional
        VaR confidence level (default 0.95).
    window : int, optional
        Rolling estimation window in days (default 250).
    holding_period : int, optional
        Holding period in days for scaling the VaR forecast (default 1).
        The VaR is scaled by ``sqrt(holding_period)`` under the
        square-root-of-time rule.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"dates_idx"`` : np.ndarray -- integer indices of the test dates
        - ``"var_forecasts"`` : np.ndarray -- VaR forecasts for each test date
        - ``"actual_returns"`` : np.ndarray -- realised returns on test dates
        - ``"violations"`` : np.ndarray -- boolean violation flags
        - ``"n_violations"`` : int -- total number of violations
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < window + 1:
        raise ValueError(
            f"Not enough data: need at least {window + 1} observations, got {n}."
        )

    dates_idx = []
    var_forecasts_list = []
    actual_returns_list = []

    sqrt_hp = np.sqrt(holding_period)

    for t in range(window, n):
        trailing = returns[t - window : t]
        var_value = var_func(trailing, confidence) * sqrt_hp
        actual = returns[t]

        dates_idx.append(t)
        var_forecasts_list.append(var_value)
        actual_returns_list.append(actual)

    dates_idx_arr = np.array(dates_idx, dtype=int)
    var_forecasts_arr = np.array(var_forecasts_list, dtype=float)
    actual_returns_arr = np.array(actual_returns_list, dtype=float)

    violations = count_violations(actual_returns_arr, var_forecasts_arr)

    return {
        "dates_idx": dates_idx_arr,
        "var_forecasts": var_forecasts_arr,
        "actual_returns": actual_returns_arr,
        "violations": violations,
        "n_violations": int(np.sum(violations)),
    }
