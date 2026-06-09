"""Extreme Value Theory (EVT) — Peaks-over-Threshold with Generalised Pareto Distribution.

Provides more accurate tail risk estimates than Historical or Parametric VaR
at extreme confidence levels (99%+) by modelling only the tail of the
return distribution with the GPD.
"""
from __future__ import annotations

# ruff: noqa: E402

import os
import tempfile
import warnings
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "var-risk-engine-matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import genpareto, kstest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from var_risk_engine.expected_shortfall import es_parametric, expected_shortfall
from var_risk_engine.var_historical import historical_var
from var_risk_engine.var_parametric import parametric_var


# ---------------------------------------------------------------------------
# 1. Fit GPD to tail losses
# ---------------------------------------------------------------------------

def fit_gpd(
    losses: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Fit a Generalised Pareto Distribution to losses exceeding a threshold.

    Uses the Peaks-over-Threshold (POT) approach: only observations that
    exceed *threshold* are used to estimate the GPD shape (xi) and scale
    (beta) parameters via maximum likelihood (``scipy.stats.genpareto.fit``).

    When *threshold* is ``None`` the 90th percentile of *losses* is used as
    a common rule-of-thumb.

    Args:
        losses: 1-D array of **positive** loss values (e.g. negated returns).
            All values should be real numbers; NaN / Inf are silently dropped.
        threshold: The loss level above which observations are treated as
            tail exceedances.  If ``None``, the 90th percentile of *losses*
            is used.

    Returns:
        A dictionary with the following keys:

        * ``"xi"`` — GPD shape parameter.
        * ``"beta"`` — GPD scale parameter.
        * ``"threshold"`` — the threshold that was used.
        * ``"n_exceedances"`` — number of observations above the threshold.
        * ``"n_total"`` — total number of (clean) observations.
        * ``"exceedance_rate"`` — ``n_exceedances / n_total``.
        * ``"ks_statistic"`` — Kolmogorov-Smirnov test statistic.
        * ``"ks_pvalue"`` — KS test p-value (null: exceedances come from
          the fitted GPD).

    Raises:
        ValueError: If fewer than 2 exceedances are found (insufficient data
            to fit a two-parameter distribution).
    """
    losses = np.asarray(losses, dtype=float).ravel()
    # Drop NaN / Inf silently.
    losses = losses[np.isfinite(losses)]

    if losses.size == 0:
        raise ValueError("losses array is empty after removing NaN/Inf values.")

    # --- Auto-select threshold ------------------------------------------------
    if threshold is None:
        threshold = float(np.percentile(losses, 90))

    # --- Extract exceedances --------------------------------------------------
    exceedances = losses[losses > threshold]
    n_exceedances = exceedances.size
    n_total = losses.size

    if n_exceedances < 2:
        raise ValueError(
            f"Only {n_exceedances} exceedance(s) found above threshold "
            f"{threshold:.6g}.  Need at least 2 to fit the GPD.  "
            "Consider lowering the threshold."
        )

    # --- Fit GPD (location fixed at 0) ----------------------------------------
    # genpareto parameterisation: shape=c (xi), loc, scale (beta).
    # Fixing loc=0 is standard for POT since we model excess over threshold.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi, _loc, beta = genpareto.fit(exceedances - threshold, floc=0)

    # --- Goodness-of-fit: Kolmogorov-Smirnov test ----------------------------
    ks_stat, ks_pvalue = kstest(
        exceedances - threshold,
        "genpareto",
        args=(xi, 0, beta),
    )

    return {
        "xi": float(xi),
        "beta": float(beta),
        "threshold": float(threshold),
        "n_exceedances": int(n_exceedances),
        "n_total": int(n_total),
        "exceedance_rate": float(n_exceedances / n_total),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def mean_excess_table(
    losses: np.ndarray,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Compute the empirical mean excess function across candidate thresholds.

    A roughly linear mean-excess curve above a candidate threshold is a
    practical diagnostic supporting the Peaks-over-Threshold GPD assumption.
    """
    if quantiles is None:
        quantiles = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975]

    losses = np.asarray(losses, dtype=float).ravel()
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        raise ValueError("losses array is empty after removing NaN/Inf values.")

    rows: list[dict[str, float]] = []
    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"quantiles must be in (0, 1), got {q}")
        threshold = float(np.quantile(losses, q))
        exceedances = losses[losses > threshold] - threshold
        rows.append(
            {
                "quantile": q,
                "threshold": threshold,
                "n_exceedances": int(exceedances.size),
                "mean_excess": float(np.mean(exceedances)) if exceedances.size else np.nan,
            }
        )

    return pd.DataFrame(rows)


def threshold_stability_table(
    losses: np.ndarray,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Fit GPD across candidate thresholds and report parameter stability."""
    if quantiles is None:
        quantiles = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975]

    losses = np.asarray(losses, dtype=float).ravel()
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        raise ValueError("losses array is empty after removing NaN/Inf values.")

    rows: list[dict[str, float]] = []
    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"quantiles must be in (0, 1), got {q}")
        threshold = float(np.quantile(losses, q))
        try:
            params = fit_gpd(losses, threshold=threshold)
            rows.append(
                {
                    "quantile": q,
                    "threshold": threshold,
                    "n_exceedances": params["n_exceedances"],
                    "xi": params["xi"],
                    "beta": params["beta"],
                    "ks_pvalue": params["ks_pvalue"],
                }
            )
        except ValueError:
            rows.append(
                {
                    "quantile": q,
                    "threshold": threshold,
                    "n_exceedances": 0,
                    "xi": np.nan,
                    "beta": np.nan,
                    "ks_pvalue": np.nan,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. EVT Value-at-Risk
# ---------------------------------------------------------------------------

def evt_var(
    gpd_params: dict,
    confidence: float = 0.99,
) -> float:
    """Compute Value-at-Risk using the POT-EVT framework.

    The closed-form POT VaR estimator is:

    .. math::

        \\text{VaR}_p = u + \\frac{\\beta}{\\xi}
            \\left[ \\left( \\frac{n}{n_u}(1-p) \\right)^{-\\xi} - 1 \\right]

    where *u* is the threshold, *n* is the total number of observations,
    *n_u* is the number of exceedances, and *p* is the confidence level.

    When the shape parameter xi is effectively zero (|xi| < 1e-6) the
    exponential limit is used instead:

    .. math::

        \\text{VaR}_p = u - \\beta \\, \\ln\\!\\left(\\frac{n}{n_u}(1-p)\\right)

    Args:
        gpd_params: Dictionary returned by :func:`fit_gpd`.  Must contain
            keys ``"xi"``, ``"beta"``, ``"threshold"``, ``"n_total"``, and
            ``"n_exceedances"``.
        confidence: Confidence level *p*.  Must be in (0, 1).

    Returns:
        A positive float representing the EVT-based VaR loss estimate.

    Raises:
        ValueError: If *confidence* is not in (0, 1).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    xi: float = gpd_params["xi"]
    beta: float = gpd_params["beta"]
    u: float = gpd_params["threshold"]
    n: int = gpd_params["n_total"]
    n_u: int = gpd_params["n_exceedances"]

    # Probability weight: (n / n_u) * (1 - p)
    prob_weight = (n / n_u) * (1.0 - confidence)

    if abs(xi) < 1e-6:
        # Exponential limit (xi -> 0).
        var_p = u - beta * np.log(prob_weight)
    else:
        var_p = u + (beta / xi) * (prob_weight ** (-xi) - 1.0)

    return float(var_p)


# ---------------------------------------------------------------------------
# 3. EVT Expected Shortfall
# ---------------------------------------------------------------------------

def evt_es(
    gpd_params: dict,
    confidence: float = 0.99,
) -> float:
    """Compute Expected Shortfall using the POT-EVT framework.

    The closed-form POT ES estimator (for xi < 1) is:

    .. math::

        \\text{ES}_p = \\frac{\\text{VaR}_p}{1 - \\xi}
            + \\frac{\\beta - \\xi \\, u}{1 - \\xi}

    When xi >= 1 the mean of the tail distribution does not exist (the
    distribution is so heavy-tailed that ES is infinite).

    When |xi| < 1e-6 the exponential-limit form is used:

    .. math::

        \\text{ES}_p = \\text{VaR}_p + \\beta

    Args:
        gpd_params: Dictionary returned by :func:`fit_gpd`.
        confidence: Confidence level *p*.  Must be in (0, 1).

    Returns:
        A positive float representing the EVT-based Expected Shortfall.

    Raises:
        ValueError: If *confidence* is not in (0, 1).
        ValueError: If xi >= 1 (ES is undefined / infinite).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    xi: float = gpd_params["xi"]
    beta: float = gpd_params["beta"]
    u: float = gpd_params["threshold"]

    if xi >= 1.0:
        raise ValueError(
            f"GPD shape parameter xi = {xi:.4f} >= 1.  "
            "Expected Shortfall is infinite (tail mean does not exist) "
            "for distributions this heavy-tailed."
        )

    var_p = evt_var(gpd_params, confidence=confidence)

    if abs(xi) < 1e-6:
        # Exponential limit: ES = VaR + beta.
        es_p = var_p + beta
    else:
        es_p = var_p / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)

    return float(es_p)


# ---------------------------------------------------------------------------
# 4. Rolling-window EVT-VaR time series
# ---------------------------------------------------------------------------

def evt_var_series(
    returns: np.ndarray,
    threshold_quantile: float = 0.90,
    confidence: float = 0.99,
    window: int = 500,
) -> np.ndarray:
    """Compute a rolling-window EVT-VaR time series.

    For each window of *window* consecutive observations the function:

    1. Converts returns to losses (negates them).
    2. Fits a GPD above the *threshold_quantile* of losses within that window.
    3. Computes the EVT-VaR at the requested *confidence* level.

    This is useful for comparing how EVT-based tail risk evolves over time
    relative to Historical or Parametric VaR.

    Args:
        returns: 1-D array of daily portfolio returns.  Shape ``(n_days,)``.
            Must have at least *window* observations.
        threshold_quantile: Quantile of losses within each window used to
            set the GPD threshold (e.g. 0.90 means the 90th percentile of
            losses).  Must be in (0, 1).
        confidence: Confidence level for VaR.  Must be in (0, 1).
        window: Length of the rolling look-back window in trading days.
            Must be >= 2.

    Returns:
        1-D ``np.ndarray`` of shape ``(n_days - window + 1,)`` containing
        the EVT-VaR estimate for each window position.  If the GPD fit
        fails for a particular window (e.g. too few exceedances) the value
        is set to ``np.nan``.

    Raises:
        ValueError: If *window* exceeds the length of *returns* or is < 2,
            or if *confidence* / *threshold_quantile* are not in (0, 1).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if not 0 < threshold_quantile < 1:
        raise ValueError(
            f"threshold_quantile must be in (0, 1), got {threshold_quantile}"
        )
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    returns = np.asarray(returns, dtype=float).ravel()

    if window > len(returns):
        raise ValueError(
            f"window ({window}) exceeds the length of returns "
            f"({len(returns)})"
        )

    n_obs = len(returns)
    n_windows = n_obs - window + 1
    var_series = np.full(n_windows, np.nan, dtype=float)

    for i in range(n_windows):
        window_returns = returns[i : i + window]
        losses = -window_returns

        # Auto-threshold at the requested quantile of losses.
        threshold = float(np.percentile(losses, threshold_quantile * 100))
        exceedances = losses[losses > threshold]

        if exceedances.size < 2:
            # Not enough tail observations — skip this window.
            continue

        try:
            gpd_params = fit_gpd(losses, threshold=threshold)
            var_series[i] = evt_var(gpd_params, confidence=confidence)
        except (ValueError, RuntimeError):
            # Fitting can fail for degenerate windows; leave as NaN.
            continue

    return var_series


# ---------------------------------------------------------------------------
# 5. Compare EVT with Historical and Parametric methods
# ---------------------------------------------------------------------------

def compare_evt_methods(
    returns: np.ndarray,
    confidence_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Compare EVT-VaR/ES with Historical and Parametric methods.

    For each confidence level the function computes:

    * **Historical VaR** (via :func:`var_historical.historical_var`)
    * **Historical ES** (via :func:`expected_shortfall.expected_shortfall`)
    * **Parametric VaR** (via :func:`var_parametric.parametric_var`)
    * **Parametric ES** (via :func:`expected_shortfall.es_parametric`)
    * **EVT-VaR** (via :func:`evt_var`)
    * **EVT-ES** (via :func:`evt_es`)

    At high confidence levels (99%+), EVT-VaR should typically exceed
    Historical VaR because EVT extrapolates into the tail using the GPD
    rather than relying solely on observed extremes.

    Args:
        returns: 1-D array of daily portfolio returns.  Shape ``(n_days,)``.
        confidence_levels: List of confidence levels to evaluate.  Defaults
            to ``[0.95, 0.975, 0.99, 0.995, 0.999]``.

    Returns:
        A :class:`pandas.DataFrame` with columns:

        * ``confidence``
        * ``var_historical``
        * ``es_historical``
        * ``var_parametric``
        * ``es_parametric``
        * ``var_evt``
        * ``es_evt``

        Each row corresponds to one confidence level.

    Raises:
        ValueError: If any element of *confidence_levels* is not in (0, 1).
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.975, 0.99, 0.995, 0.999]

    returns = np.asarray(returns, dtype=float).ravel()
    losses = -returns

    # Fit GPD once on the full loss series.
    gpd_params = fit_gpd(losses)

    rows: list[dict[str, float]] = []

    for cl in confidence_levels:
        if not 0 < cl < 1:
            raise ValueError(
                f"All confidence levels must be in (0, 1); got {cl}"
            )

        # Historical methods.
        v_hist = historical_var(returns, confidence=cl)
        es_hist = expected_shortfall(returns, confidence=cl)

        # Parametric methods.
        v_para = parametric_var(returns, confidence=cl)
        es_para = es_parametric(returns, confidence=cl)

        # EVT methods.
        v_evt = evt_var(gpd_params, confidence=cl)
        try:
            es_evt = evt_es(gpd_params, confidence=cl)
        except ValueError:
            # xi >= 1: ES is infinite.
            es_evt = float("inf")

        rows.append(
            {
                "confidence": cl,
                "var_historical": v_hist,
                "es_historical": es_hist,
                "var_parametric": v_para,
                "es_parametric": es_para,
                "var_evt": v_evt,
                "es_evt": es_evt,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. GPD Q-Q plot
# ---------------------------------------------------------------------------

def gpd_qq_plot(
    losses: np.ndarray,
    gpd_params: dict,
) -> tuple:
    """Create a Q-Q plot comparing empirical exceedances against the fitted GPD.

    Plots the sorted empirical quantiles of exceedances (losses above
    threshold, shifted by threshold) against the theoretical quantiles of
    the fitted Generalised Pareto Distribution.  A 45-degree reference line
    is drawn for visual assessment of fit quality.

    Args:
        losses: 1-D array of positive loss values (same array that was
            passed to :func:`fit_gpd`).
        gpd_params: Dictionary returned by :func:`fit_gpd`.  Must contain
            ``"xi"``, ``"beta"``, and ``"threshold"``.

    Returns:
        A ``(fig, ax)`` tuple of :class:`matplotlib.figure.Figure` and
        :class:`matplotlib.axes.Axes` objects.  The caller is responsible
        for calling ``plt.show()`` or ``fig.savefig(...)``.

    Raises:
        ValueError: If no exceedances are found above the threshold stored
            in *gpd_params*.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    losses = losses[np.isfinite(losses)]

    xi: float = gpd_params["xi"]
    beta: float = gpd_params["beta"]
    threshold: float = gpd_params["threshold"]

    exceedances = losses[losses > threshold] - threshold
    n_exc = exceedances.size

    if n_exc == 0:
        raise ValueError(
            "No exceedances found above the threshold stored in gpd_params."
        )

    # Empirical quantiles (plotting positions): (i - 0.5) / n  for i = 1..n.
    sorted_exc = np.sort(exceedances)
    empirical_probs = (np.arange(1, n_exc + 1) - 0.5) / n_exc

    # Theoretical quantiles from the fitted GPD.
    theoretical_quantiles = genpareto.ppf(empirical_probs, xi, loc=0, scale=beta)

    # --- Build figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        theoretical_quantiles,
        sorted_exc,
        s=18,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
        color="steelblue",
        label="Empirical exceedances",
    )

    # 45-degree reference line.
    all_vals = np.concatenate([theoretical_quantiles, sorted_exc])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    margin = (hi - lo) * 0.05
    line_range = [lo - margin, hi + margin]
    ax.plot(
        line_range,
        line_range,
        "r--",
        linewidth=1.2,
        label="45\u00b0 reference line",
    )

    ax.set_xlabel("Theoretical GPD quantiles", fontsize=11)
    ax.set_ylabel("Empirical exceedance quantiles", fontsize=11)
    ax.set_title(
        f"GPD Q-Q Plot  (xi={xi:.3f}, beta={beta:.3f}, "
        f"n_u={n_exc})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    return fig, ax
