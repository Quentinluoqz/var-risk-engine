"""Stress testing: historical scenarios, hypothetical shocks, and reverse stress testing.

Evaluates portfolio risk under extreme but plausible market conditions,
going beyond the VaR framework which assumes "normal" market behaviour.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from var_risk_engine.covariance import sample_covariance
from var_risk_engine.expected_shortfall import expected_shortfall
from var_risk_engine.var_historical import historical_var
from var_risk_engine.var_parametric import parametric_var_from_cov

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR = 252

# Pre-defined historical stress periods (start, end) — both inclusive.
_HISTORICAL_STRESS_PERIODS: dict[str, tuple[str, str]] = {
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate Hike Cycle": ("2022-01-01", "2022-10-31"),
}

# Worst-rolling-window length (trading days).
_WORST_WINDOW_LENGTH = 20

# Default confidence level used throughout this module.
_DEFAULT_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Compute the daily portfolio return series as ``returns @ weights``.

    Args:
        returns: Multi-asset return DataFrame with a ``DatetimeIndex``.
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.

    Returns:
        A :class:`pandas.Series` indexed by date with the portfolio return
        for each observation.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    port_ret = returns.values @ weights
    return pd.Series(port_ret, index=returns.index, name="portfolio_return")


def _find_worst_window(
    portfolio_series: pd.Series,
    window: int = _WORST_WINDOW_LENGTH,
) -> tuple[int, int] | None:
    """Find the start/end indices of the worst *window*-day cumulative return.

    Args:
        portfolio_series: Daily portfolio return series.
        window: Number of consecutive trading days to evaluate.

    Returns:
        A ``(start, end)`` tuple of integer positions (inclusive on both
        ends) into *portfolio_series*, or ``None`` if the series is shorter
        than *window*.
    """
    n = len(portfolio_series)
    if n < window:
        return None

    ret_vals = portfolio_series.values
    # Cumulative log-return over each rolling window.
    cum_ret = np.convolve(ret_vals, np.ones(window), mode="valid")
    worst_idx = int(np.argmin(cum_ret))
    return worst_idx, worst_idx + window - 1


def _annualised_vol(portfolio_returns: np.ndarray) -> float:
    """Annualised volatility of a daily return series.

    Args:
        portfolio_returns: 1-D array of daily portfolio returns.

    Returns:
        Annualised standard deviation (assuming 252 trading days).
    """
    return float(np.std(portfolio_returns, ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------------------
# 1. Historical stress scenarios
# ---------------------------------------------------------------------------


def historical_stress_scenarios(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Evaluate portfolio performance during historical stress periods.

    Automatically identifies three stress episodes in the data:

    1. **COVID-19 Crash** (2020-02-19 to 2020-03-23)
    2. **2022 Rate Hike Cycle** (2022-01-01 to 2022-10-31)
    3. **Worst 20-day period** — the 20 consecutive trading days with the
       lowest cumulative portfolio return anywhere in the dataset.

    For each episode the following metrics are computed:

    * ``cumulative_return`` — compounded portfolio return over the period.
    * ``worst_daily_return`` — single worst daily portfolio return.
    * ``var_95`` — Historical VaR at 95 % confidence (period returns only).
    * ``es_95`` — Expected Shortfall at 95 % confidence (period returns only).
    * ``annualised_vol`` — annualised portfolio volatility during the period.

    Args:
        returns: Multi-asset daily return DataFrame with a
            :class:`pandas.DatetimeIndex` and one column per asset.
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
            Must be the same length as the number of columns in *returns*.

    Returns:
        A :class:`pandas.DataFrame` with scenario names as the index and
        the metrics listed above as columns.  If a stress period has no
        overlapping data (e.g. the dataset starts after 2020) the
        corresponding row will contain ``NaN`` values and a ``note`` column
        will carry a descriptive message.

    Raises:
        ValueError: If *weights* length does not match the number of asset
            columns in *returns*.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.shape[0] != returns.shape[1]:
        raise ValueError(
            f"weights length ({weights.shape[0]}) does not match the number "
            f"of asset columns ({returns.shape[1]}) in returns."
        )

    port_series = _portfolio_returns(returns, weights)
    results: list[dict[str, Any]] = []

    # --- Pre-defined historical periods ------------------------------------
    for name, (start, end) in _HISTORICAL_STRESS_PERIODS.items():
        mask = (returns.index >= pd.Timestamp(start)) & (
            returns.index <= pd.Timestamp(end)
        )
        period_data = port_series[mask]

        if period_data.empty:
            results.append(
                {
                    "scenario": name,
                    "cumulative_return": np.nan,
                    "worst_daily_return": np.nan,
                    "var_95": np.nan,
                    "es_95": np.nan,
                    "annualised_vol": np.nan,
                    "note": f"No data in range {start} to {end}.",
                }
            )
            continue

        period_vals = period_data.values
        cum_ret = float(np.prod(1.0 + period_vals) - 1.0)
        worst_day = float(np.min(period_vals))
        var_95 = historical_var(period_vals, confidence=_DEFAULT_CONFIDENCE)
        es_95 = expected_shortfall(period_vals, confidence=_DEFAULT_CONFIDENCE)
        ann_vol = _annualised_vol(period_vals)

        results.append(
            {
                "scenario": name,
                "cumulative_return": cum_ret,
                "worst_daily_return": worst_day,
                "var_95": var_95,
                "es_95": es_95,
                "annualised_vol": ann_vol,
                "note": "",
            }
        )

    # --- Worst 20-day rolling window --------------------------------------
    window_indices = _find_worst_window(port_series, window=_WORST_WINDOW_LENGTH)
    if window_indices is None:
        results.append(
            {
                "scenario": f"Worst {_WORST_WINDOW_LENGTH}-day period",
                "cumulative_return": np.nan,
                "worst_daily_return": np.nan,
                "var_95": np.nan,
                "es_95": np.nan,
                "annualised_vol": np.nan,
                "note": (
                    f"Insufficient data: need at least {_WORST_WINDOW_LENGTH} "
                    f"observations, have {len(port_series)}."
                ),
            }
        )
    else:
        start_idx, end_idx = window_indices
        period_data = port_series.iloc[start_idx : end_idx + 1]
        period_vals = period_data.values
        cum_ret = float(np.prod(1.0 + period_vals) - 1.0)
        worst_day = float(np.min(period_vals))
        var_95 = historical_var(period_vals, confidence=_DEFAULT_CONFIDENCE)
        es_95 = expected_shortfall(period_vals, confidence=_DEFAULT_CONFIDENCE)
        ann_vol = _annualised_vol(period_vals)

        period_start = period_data.index[0].strftime("%Y-%m-%d")
        period_end = period_data.index[-1].strftime("%Y-%m-%d")

        results.append(
            {
                "scenario": f"Worst {_WORST_WINDOW_LENGTH}-day period",
                "cumulative_return": cum_ret,
                "worst_daily_return": worst_day,
                "var_95": var_95,
                "es_95": es_95,
                "annualised_vol": ann_vol,
                "note": f"Period: {period_start} to {period_end}.",
            }
        )

    df = pd.DataFrame(results).set_index("scenario")
    return df


# ---------------------------------------------------------------------------
# 2. Hypothetical stress scenarios
# ---------------------------------------------------------------------------


def _apply_directional_shock(
    weights: np.ndarray,
    shock_vector: np.ndarray,
) -> float:
    """Compute the portfolio return under a deterministic shock vector.

    Args:
        weights: Portfolio weights, shape ``(n_assets,)``.
        shock_vector: Per-asset return shocks, shape ``(n_assets,)``.

    Returns:
        The scalar shocked portfolio return.
    """
    return float(np.dot(weights, shock_vector))


def _build_default_shocks(
    n_assets: int,
    asset_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the four default hypothetical stress scenarios.

    Each scenario dictionary contains:

    * ``type`` — one of ``"directional"`` or ``"covariance"``.
    * For ``"directional"``: a ``shock_vector`` of shape ``(n_assets,)``.
    * For ``"covariance"``: a flag describing the covariance transformation.

    The asset-class heuristic classifies columns by name substring:

    * **Equities**: names containing ``"EQ"``, ``"SP"``, ``"STOCK"``, or
      tickers that do not match bonds/gold.
    * **Bonds**: names containing ``"BOND"``, ``"TLT"``, ``"AGG"``,
      ``"BND"``, ``"FIXED"``.
    * **Gold**: names containing ``"GOLD"``, ``"GLD"``, ``"XAU"``.

    If asset names are not provided or classification is ambiguous, all
    assets are treated as equities.

    Args:
        n_assets: Number of assets in the portfolio.
        asset_names: Optional list of column names from the returns
            DataFrame, used to classify assets.

    Returns:
        A dictionary keyed by scenario name.
    """

    def _classify(name: str) -> str:
        upper = name.upper()
        if any(tok in upper for tok in ("BOND", "TLT", "AGG", "BND", "FIXED")):
            return "bond"
        if any(tok in upper for tok in ("GOLD", "GLD", "XAU")):
            return "gold"
        return "equity"

    if asset_names is None:
        classes = ["equity"] * n_assets
    else:
        classes = [_classify(n) for n in asset_names]

    def _shock_vec(equity: float, bond: float, gold: float) -> np.ndarray:
        mapping = {"equity": equity, "bond": bond, "gold": gold}
        return np.array([mapping[c] for c in classes])

    # Directional shocks.
    market_crash_shock = _shock_vec(equity=-0.25, bond=-0.05, gold=-0.10)
    flight_to_quality_shock = _shock_vec(equity=-0.20, bond=0.05, gold=0.10)

    return {
        "Market Crash": {
            "type": "directional",
            "shock_vector": market_crash_shock,
        },
        "Flight to Quality": {
            "type": "directional",
            "shock_vector": flight_to_quality_shock,
        },
        "Correlation Spike": {
            "type": "covariance",
            "method": "correlation_spike",
            "target_corr": 0.9,
        },
        "Volatility Doubles": {
            "type": "covariance",
            "method": "vol_doubles",
        },
    }


def hypothetical_stress(
    returns: pd.DataFrame,
    weights: np.ndarray,
    shocks: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Apply hypothetical stress scenarios to the portfolio.

    Evaluates the portfolio under four default scenarios (or user-supplied
    shocks) using the most recent observation window:

    * **Market Crash** — equities drop 25 %, bonds 5 %, gold 10 %.
    * **Correlation Spike** — all pairwise correlations set to 0.9, then
      parametric VaR is recomputed from the stressed covariance matrix.
    * **Volatility Doubles** — all asset volatilities double (covariance
      matrix multiplied by 4), then parametric VaR is recomputed.
    * **Flight to Quality** — equities drop 20 %, bonds rise 5 %, gold
      rises 10 %.

    Asset classification (equity / bond / gold) is inferred from column
    names using substring heuristics.  Unrecognised names default to equity.

    Args:
        returns: Multi-asset daily return DataFrame with a
            :class:`pandas.DatetimeIndex` and one column per asset.
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        shocks: Optional user-defined shocks.  If provided, each key is a
            scenario name and each value is a 1-D shock array of length
            ``n_assets`` applied as a directional shock.  When ``None`` the
            four built-in scenarios are used.

    Returns:
        A :class:`pandas.DataFrame` with scenario names as the index and
        the following columns:

        * ``portfolio_return`` — the stressed portfolio return.
        * ``loss_bps`` — portfolio loss in basis points (positive = loss).
        * ``var_95`` — stressed 95 % VaR.
        * ``es_95`` — stressed 95 % Expected Shortfall.
        * ``note`` — human-readable description of the scenario.

    Raises:
        ValueError: If *weights* length does not match the number of
            columns in *returns*.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    n_assets = returns.shape[1]

    if weights.shape[0] != n_assets:
        raise ValueError(
            f"weights length ({weights.shape[0]}) does not match the number "
            f"of asset columns ({n_assets}) in returns."
        )

    port_series = _portfolio_returns(returns, weights)
    asset_names = list(returns.columns)

    # Determine scenario definitions.
    if shocks is not None:
        # User-supplied shocks are treated as directional.
        scenario_defs: dict[str, dict[str, Any]] = {}
        for name, vec in shocks.items():
            scenario_defs[name] = {
                "type": "directional",
                "shock_vector": np.asarray(vec, dtype=float).ravel(),
            }
    else:
        scenario_defs = _build_default_shocks(n_assets, asset_names)

    # Pre-compute sample covariance and portfolio statistics for cov scenarios.
    cov = sample_covariance(returns)
    port_ret_vals = port_series.values

    results: list[dict[str, Any]] = []

    for name, spec in scenario_defs.items():
        if spec["type"] == "directional":
            shock_vec = np.asarray(spec["shock_vector"], dtype=float).ravel()
            if shock_vec.shape[0] != n_assets:
                warnings.warn(
                    f"Shock vector for '{name}' has length "
                    f"{shock_vec.shape[0]}, expected {n_assets}. Skipping.",
                    stacklevel=2,
                )
                continue

            shocked_port_ret = _apply_directional_shock(weights, shock_vec)
            loss_bps = -shocked_port_ret * 10_000.0

            # Stressed VaR / ES: shift the return distribution by the shock.
            # Each day's return is adjusted by the per-asset shock weighted
            # by portfolio weights — effectively overlaying the shock on the
            # empirical distribution.
            daily_shock = float(np.dot(weights, shock_vec))
            stressed_returns = port_ret_vals + daily_shock
            var_95 = historical_var(stressed_returns, confidence=_DEFAULT_CONFIDENCE)
            es_95 = expected_shortfall(stressed_returns, confidence=_DEFAULT_CONFIDENCE)

            results.append(
                {
                    "scenario": name,
                    "portfolio_return": shocked_port_ret,
                    "loss_bps": loss_bps,
                    "var_95": var_95,
                    "es_95": es_95,
                    "note": f"Directional shock: {shock_vec.round(4).tolist()}",
                }
            )

        elif spec["type"] == "covariance":
            method = spec.get("method", "")

            if method == "correlation_spike":
                target_corr = spec.get("target_corr", 0.9)
                # Build stressed covariance: keep diagonal (variances) but
                # set all off-diagonal elements to
                #   target_corr * sqrt(var_i * var_j).
                stressed_cov = cov.copy()
                stds = np.sqrt(np.diag(cov))
                for i in range(n_assets):
                    for j in range(n_assets):
                        if i != j:
                            stressed_cov[i, j] = (
                                target_corr * stds[i] * stds[j]
                            )

                var_95 = parametric_var_from_cov(
                    weights,
                    stressed_cov,
                    confidence=_DEFAULT_CONFIDENCE,
                )
                # ES approximation: for a normal distribution
                #   ES = sigma * phi(z) / (1 - alpha)
                port_var = float(weights @ stressed_cov @ weights)
                port_sigma = np.sqrt(port_var)
                z = norm.ppf(_DEFAULT_CONFIDENCE)
                phi_z = norm.pdf(z)
                es_95 = float(port_sigma * phi_z / (1.0 - _DEFAULT_CONFIDENCE))

                # Portfolio return under correlation spike is zero (the
                # shock affects risk, not the level).
                shocked_port_ret = 0.0
                loss_bps = 0.0

                results.append(
                    {
                        "scenario": name,
                        "portfolio_return": shocked_port_ret,
                        "loss_bps": loss_bps,
                        "var_95": var_95,
                        "es_95": es_95,
                        "note": (
                            f"All pairwise correlations set to "
                            f"{target_corr}."
                        ),
                    }
                )

            elif method == "vol_doubles":
                # Doubling volatilities => multiply covariance by 4.
                stressed_cov = cov * 4.0

                var_95 = parametric_var_from_cov(
                    weights,
                    stressed_cov,
                    confidence=_DEFAULT_CONFIDENCE,
                )
                port_var = float(weights @ stressed_cov @ weights)
                port_sigma = np.sqrt(port_var)
                z = norm.ppf(_DEFAULT_CONFIDENCE)
                phi_z = norm.pdf(z)
                es_95 = float(port_sigma * phi_z / (1.0 - _DEFAULT_CONFIDENCE))

                shocked_port_ret = 0.0
                loss_bps = 0.0

                results.append(
                    {
                        "scenario": name,
                        "portfolio_return": shocked_port_ret,
                        "loss_bps": loss_bps,
                        "var_95": var_95,
                        "es_95": es_95,
                        "note": "All asset volatilities doubled (cov * 4).",
                    }
                )

    df = pd.DataFrame(results).set_index("scenario")
    return df


# ---------------------------------------------------------------------------
# 3. Reverse stress testing
# ---------------------------------------------------------------------------


def reverse_stress_test(
    returns: pd.DataFrame,
    weights: np.ndarray,
    max_loss: float = 0.10,
) -> dict:
    """Work backwards from a maximum tolerable loss to identify causative conditions.

    Given a *max_loss* threshold (expressed as a positive fraction, e.g.
    0.10 for 10 %), this function determines:

    1. The **percentile** of the historical portfolio-return distribution at
       which the loss equals *max_loss*.
    2. The **uniform asset drop** (same percentage decline across all
       assets) that would produce *max_loss*.
    3. The **volatility multiplier** — the factor by which current
       volatility would need to increase so that the 95 % parametric VaR
       equals *max_loss*.

    Args:
        returns: Multi-asset daily return DataFrame with a
            :class:`pandas.DatetimeIndex` and one column per asset.
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        max_loss: Maximum tolerable portfolio loss expressed as a positive
            fraction (e.g. 0.10 for 10 %).  Must be > 0.

    Returns:
        A dictionary with the following keys:

        * ``max_loss`` — the input maximum loss.
        * ``percentile`` — the percentile (0--100) of the historical
          distribution corresponding to *max_loss*.
        * ``uniform_drop_required`` — the uniform per-asset decline that
          would produce *max_loss*.
        * ``vol_multiplier`` — the factor by which volatility must increase
          for the parametric 95 % VaR to equal *max_loss*.
        * ``interpretation`` — a human-readable summary string.

    Raises:
        ValueError: If *max_loss* is not positive, or *weights* length
            does not match the number of columns in *returns*.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    n_assets = returns.shape[1]

    if weights.shape[0] != n_assets:
        raise ValueError(
            f"weights length ({weights.shape[0]}) does not match the number "
            f"of asset columns ({n_assets}) in returns."
        )
    if max_loss <= 0:
        raise ValueError(f"max_loss must be positive, got {max_loss}")

    port_series = _portfolio_returns(returns, weights)
    port_vals = port_series.values

    # --- Percentile at which loss == max_loss ------------------------------
    # We want: P(R <= -max_loss) = quantile_fraction
    # i.e. what fraction of historical returns are worse than -max_loss?
    n_obs = len(port_vals)
    n_worse = int(np.sum(port_vals <= -max_loss))
    percentile = (n_worse / n_obs) * 100.0 if n_obs > 0 else 0.0

    # --- Uniform drop required --------------------------------------------
    # If every asset drops by d%, the portfolio drops by
    #   sum(w_i * d) = d * sum(w_i) = d  (weights sum to 1).
    # So d = max_loss / sum(w_i).
    weight_sum = float(np.sum(weights))
    if abs(weight_sum) < 1e-12:
        uniform_drop = float("inf")
    else:
        uniform_drop = max_loss / weight_sum

    # --- Volatility multiplier --------------------------------------------
    # Current parametric VaR (1-day, 95%):
    #   VaR = -(mu + z * sigma)
    # We want: -(mu + z * sigma* ) = max_loss
    #   => sigma* = (max_loss + mu) / (-z)
    #   => vol_multiplier = sigma* / sigma
    mu = float(np.mean(port_vals))
    sigma = float(np.std(port_vals, ddof=1))
    z = norm.ppf(1.0 - _DEFAULT_CONFIDENCE)  # negative

    if sigma < 1e-15:
        vol_multiplier = float("inf")
    else:
        # Solve: -(mu + z * sigma_star) = max_loss
        #   => sigma_star = (max_loss + mu) / (-z)
        sigma_star = (max_loss + mu) / (-z)
        vol_multiplier = sigma_star / sigma if sigma > 0 else float("inf")
        # If sigma_star is negative the loss is unreachable via vol alone
        # (the mean return already exceeds the max_loss in magnitude).
        if sigma_star < 0:
            vol_multiplier = float("nan")

    # --- Human-readable interpretation ------------------------------------
    interpretation_parts = [
        f"A portfolio loss of {max_loss:.1%} corresponds to the "
        f"{percentile:.1f}th percentile of the historical distribution.",
        f"A uniform asset decline of {uniform_drop:.1%} across all assets "
        f"would produce this loss.",
    ]
    if not np.isnan(vol_multiplier) and not np.isinf(vol_multiplier):
        interpretation_parts.append(
            f"Volatility would need to increase by a factor of "
            f"{vol_multiplier:.2f}x for the 95% parametric VaR to reach "
            f"this loss level."
        )
    elif np.isinf(vol_multiplier):
        interpretation_parts.append(
            "Volatility scaling is undefined (current volatility is near zero)."
        )
    else:
        interpretation_parts.append(
            "The target loss cannot be reached by increasing volatility "
            "alone given the current mean return."
        )

    return {
        "max_loss": max_loss,
        "percentile": percentile,
        "uniform_drop_required": uniform_drop,
        "vol_multiplier": vol_multiplier,
        "interpretation": " ".join(interpretation_parts),
    }


# ---------------------------------------------------------------------------
# 4. Stress comparison table
# ---------------------------------------------------------------------------


def stress_comparison_table(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Build a master comparison table combining historical and hypothetical stress results.

    Merges the outputs of :func:`historical_stress_scenarios` and
    :func:`hypothetical_stress` into a single DataFrame, sorted by
    ``portfolio_return`` (worst scenario first).

    Args:
        returns: Multi-asset daily return DataFrame with a
            :class:`pandas.DatetimeIndex` and one column per asset.
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.

    Returns:
        A :class:`pandas.DataFrame` with the following columns:

        * ``scenario`` — name of the stress scenario.
        * ``type`` — ``"historical"`` or ``"hypothetical"``.
        * ``portfolio_return`` — portfolio return under stress.
        * ``var_95`` — 95 % VaR under stress.
        * ``es_95`` — 95 % Expected Shortfall under stress.
        * ``max_daily_loss`` — worst single-day loss (from historical
          scenarios) or ``NaN`` for hypothetical scenarios.

        Rows are sorted by ``portfolio_return`` ascending (worst first).

    Raises:
        ValueError: If *weights* length does not match the number of
            columns in *returns*.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.shape[0] != returns.shape[1]:
        raise ValueError(
            f"weights length ({weights.shape[0]}) does not match the number "
            f"of asset columns ({returns.shape[1]}) in returns."
        )

    # --- Historical scenarios ---------------------------------------------
    hist_df = historical_stress_scenarios(returns, weights)
    hist_rows: list[dict[str, Any]] = []
    for scenario_name, row in hist_df.iterrows():
        hist_rows.append(
            {
                "scenario": scenario_name,
                "type": "historical",
                "portfolio_return": row.get("cumulative_return", np.nan),
                "var_95": row.get("var_95", np.nan),
                "es_95": row.get("es_95", np.nan),
                "max_daily_loss": row.get("worst_daily_return", np.nan),
            }
        )

    # --- Hypothetical scenarios -------------------------------------------
    hypo_df = hypothetical_stress(returns, weights)
    hypo_rows: list[dict[str, Any]] = []
    for scenario_name, row in hypo_df.iterrows():
        hypo_rows.append(
            {
                "scenario": scenario_name,
                "type": "hypothetical",
                "portfolio_return": row.get("portfolio_return", np.nan),
                "var_95": row.get("var_95", np.nan),
                "es_95": row.get("es_95", np.nan),
                "max_daily_loss": np.nan,
            }
        )

    combined = pd.DataFrame(hist_rows + hypo_rows)

    # Sort by portfolio_return ascending (worst first).  NaN values go last.
    combined = combined.sort_values(
        "portfolio_return", ascending=True, na_position="last"
    ).reset_index(drop=True)

    return combined
