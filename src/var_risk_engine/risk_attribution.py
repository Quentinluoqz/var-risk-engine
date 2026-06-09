"""Risk attribution -- decomposing portfolio VaR into per-asset contributions.

Answers the question: "Which assets contribute the most to portfolio risk?"
Computes Component VaR, Marginal VaR, and percentage risk contribution
for each asset in the portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from var_risk_engine.covariance import sample_covariance
from var_risk_engine.var_parametric import parametric_var_from_cov


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _z_score(confidence: float) -> float:
    """Return the positive z-score corresponding to the left-tail confidence.

    For confidence = 0.95, ``norm.ppf(1 - 0.95)`` is approximately -1.645.
    We negate it so that z is a positive number used throughout the
    attribution formulae.

    Args:
        confidence: One-sided confidence level in (0, 1).

    Returns:
        Positive z-score (e.g. ~1.645 for 95 %).
    """
    return -norm.ppf(1.0 - confidence)


def _portfolio_sigma(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio standard deviation sqrt(w^T Sigma w).

    Args:
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        cov_matrix: Covariance matrix, shape ``(n_assets, n_assets)``.

    Returns:
        Scalar portfolio standard deviation.
    """
    return float(np.sqrt(weights @ cov_matrix @ weights))


# ---------------------------------------------------------------------------
# Core attribution functions
# ---------------------------------------------------------------------------

def marginal_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
) -> np.ndarray:
    """Compute the Marginal VaR (M-VaR) for each asset.

    Marginal VaR measures the rate of change of portfolio VaR with respect
    to a small increase in the weight of asset *i*:

        M-VaR_i = dVaR / dw_i = z * (Sigma @ w)_i / sqrt(w^T Sigma w)

    where z = -norm.ppf(1 - confidence).

    Args:
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        cov_matrix: Variance-covariance matrix of asset returns,
            shape ``(n_assets, n_assets)``.  Must be symmetric and
            positive-semidefinite.
        confidence: One-sided confidence level (e.g. 0.95).  Must be in
            (0, 1).

    Returns:
        1-D array of marginal VaR contributions, shape ``(n_assets,)``.
        Each element tells you how much portfolio VaR would change per unit
        increase in the corresponding asset's weight.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    z = _z_score(confidence)
    sigma_w = cov_matrix @ weights  # Sigma @ w, shape (n,)
    port_sigma = _portfolio_sigma(weights, cov_matrix)

    mvar = z * sigma_w / port_sigma
    return mvar


def component_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
) -> np.ndarray:
    """Compute the Component VaR (C-VaR) for each asset.

    Component VaR is the Euler decomposition of portfolio VaR into
    additive per-asset contributions:

        C-VaR_i = w_i * M-VaR_i
                = w_i * z * (Sigma @ w)_i / sqrt(w^T Sigma w)

    Key property (Euler's theorem for homogeneous functions):

        sum(C-VaR_i) = Portfolio VaR

    This makes Component VaR the theoretically preferred decomposition
    for risk attribution.

    Args:
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        cov_matrix: Variance-covariance matrix of asset returns,
            shape ``(n_assets, n_assets)``.
        confidence: One-sided confidence level (e.g. 0.95).  Must be in
            (0, 1).

    Returns:
        1-D array of component VaR values, shape ``(n_assets,)``.
        The sum equals total parametric portfolio VaR.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    mvar = marginal_var(weights, cov_matrix, confidence)
    cvar = weights * mvar

    return cvar


def risk_contribution_pct(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
) -> np.ndarray:
    """Compute the percentage risk contribution of each asset.

    Defined as:

        pct_i = C-VaR_i / Portfolio_VaR * 100

    The percentages sum to 100 %.

    Args:
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        cov_matrix: Variance-covariance matrix of asset returns,
            shape ``(n_assets, n_assets)``.
        confidence: One-sided confidence level (e.g. 0.95).  Must be in
            (0, 1).

    Returns:
        1-D array of percentage contributions, shape ``(n_assets,)``.
        Values sum to 100.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    cvar = component_var(weights, cov_matrix, confidence)
    port_var = parametric_var_from_cov(weights, cov_matrix, confidence)
    return cvar / port_var * 100.0


# ---------------------------------------------------------------------------
# Tabular / reporting functions
# ---------------------------------------------------------------------------

def risk_attribution_table(
    tickers: list[str],
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Build a comprehensive risk-attribution summary table.

    The table contains one row per asset plus a ``TOTAL`` footer row with
    portfolio-level aggregates.  Rows are sorted by percentage risk
    contribution in descending order so that the largest risk drivers
    appear first.

    Columns:
        - ticker: asset identifier
        - weight: raw portfolio weight
        - weight_pct: weight expressed as a percentage (weight * 100)
        - component_var: Component VaR (additive decomposition)
        - marginal_var: Marginal VaR (derivative of VaR w.r.t. weight)
        - risk_contribution_pct: percentage of total VaR attributable to
          this asset
        - portfolio_var_total: total portfolio VaR (repeated on every row
          for convenience; summed in the TOTAL row)

    Args:
        tickers: List of asset ticker symbols, length ``n_assets``.
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        cov_matrix: Variance-covariance matrix of asset returns,
            shape ``(n_assets, n_assets)``.
        confidence: One-sided confidence level (e.g. 0.95).

    Returns:
        A :class:`~pandas.DataFrame` with ``n_assets + 1`` rows (including
        the TOTAL row) and the columns described above.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    n = len(tickers)

    cvar = component_var(weights, cov_matrix, confidence)
    mvar = marginal_var(weights, cov_matrix, confidence)
    rpct = risk_contribution_pct(weights, cov_matrix, confidence)
    port_var = parametric_var_from_cov(weights, cov_matrix, confidence)

    df = pd.DataFrame({
        "ticker": list(tickers),
        "weight": weights,
        "weight_pct": weights * 100.0,
        "component_var": cvar,
        "marginal_var": mvar,
        "risk_contribution_pct": rpct,
        "portfolio_var_total": np.full(n, port_var),
    })

    # Sort by risk contribution descending (highest risk driver first).
    df = df.sort_values("risk_contribution_pct", ascending=False).reset_index(drop=True)

    # Append TOTAL row.
    total_row = pd.DataFrame([{
        "ticker": "TOTAL",
        "weight": df["weight"].sum(),
        "weight_pct": df["weight_pct"].sum(),
        "component_var": df["component_var"].sum(),
        "marginal_var": np.nan,  # marginal VaR is not summable
        "risk_contribution_pct": df["risk_contribution_pct"].sum(),
        "portfolio_var_total": port_var,
    }])
    df = pd.concat([df, total_row], ignore_index=True)

    return df


def incremental_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.95,
) -> np.ndarray:
    """Compute the Incremental VaR (I-VaR) for each asset.

    Incremental VaR measures how much portfolio VaR decreases if asset *i*
    is removed from the portfolio:

        I-VaR_i = VaR(full portfolio) - VaR(portfolio without asset i)

    "Without asset i" means: set ``weight_i = 0`` and redistribute that
    weight equally among the remaining assets, then recompute parametric
    VaR.

    Unlike Component VaR, Incremental VaR is **not** additive -- the sum
    of all I-VaR values does *not* equal total portfolio VaR.

    Args:
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        cov_matrix: Variance-covariance matrix of asset returns,
            shape ``(n_assets, n_assets)``.
        confidence: One-sided confidence level (e.g. 0.95).

    Returns:
        1-D array of incremental VaR values, shape ``(n_assets,)``.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    n = weights.shape[0]

    port_var_full = parametric_var_from_cov(weights, cov_matrix, confidence)

    ivar = np.empty(n, dtype=float)
    for i in range(n):
        w_without = weights.copy()
        removed_weight = w_without[i]
        w_without[i] = 0.0

        # Redistribute removed weight equally among remaining assets.
        n_remaining = n - 1
        if n_remaining > 0:
            w_without += removed_weight / n_remaining

        var_without = parametric_var_from_cov(w_without, cov_matrix, confidence)
        ivar[i] = port_var_full - var_without

    return ivar


# ---------------------------------------------------------------------------
# Risk budget analysis
# ---------------------------------------------------------------------------

def risk_budget_analysis(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    target_contributions: np.ndarray | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compare current risk contributions against a target risk budget.

    By default the target is an *equal risk contribution* (ERC) budget
    where each asset should contribute ``100 / n_assets`` percent of total
    risk.  A custom target vector can be supplied instead.

    The output DataFrame contains:
        - ticker: asset index (integer label; use
          :func:`risk_attribution_table` for named tickers)
        - current_contribution_pct: actual risk contribution (%)
        - target_contribution_pct: desired risk contribution (%)
        - deviation_pct: ``current - target`` (positive = over-contributing)

    Args:
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        cov_matrix: Variance-covariance matrix of asset returns,
            shape ``(n_assets, n_assets)``.
        target_contributions: Optional 1-D array of target percentage
            contributions, shape ``(n_assets,)``.  Must sum to 100.  If
            ``None``, an equal-risk-contribution budget is used.
        confidence: One-sided confidence level (e.g. 0.95).

    Returns:
        A :class:`~pandas.DataFrame` with one row per asset and the
        columns described above.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    n = weights.shape[0]

    current_pct = risk_contribution_pct(weights, cov_matrix, confidence)

    if target_contributions is None:
        target_pct = np.full(n, 100.0 / n)
    else:
        target_pct = np.asarray(target_contributions, dtype=float).ravel()

    deviation = current_pct - target_pct

    df = pd.DataFrame({
        "ticker": [f"Asset_{i}" for i in range(n)],
        "current_contribution_pct": current_pct,
        "target_contribution_pct": target_pct,
        "deviation_pct": deviation,
    })

    return df


# ---------------------------------------------------------------------------
# Master summary
# ---------------------------------------------------------------------------

def risk_attribution_summary(
    tickers: list[str],
    weights: np.ndarray,
    returns: pd.DataFrame,
    confidence: float = 0.95,
) -> dict:
    """Compute a full risk-attribution report in one call.

    This is the main entry point for producing a comprehensive risk
    attribution analysis.  It estimates the covariance matrix from
    *returns*, then delegates to the individual attribution functions
    and packages everything into a single dictionary.

    Args:
        tickers: List of asset ticker symbols, length ``n_assets``.
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        returns: DataFrame of historical asset returns where each column
            corresponds to a ticker and each row is a time observation.
        confidence: One-sided confidence level (e.g. 0.95).

    Returns:
        A dictionary with the following keys:

        - ``"portfolio_var"`` (*float*): total parametric portfolio VaR.
        - ``"attribution_table"`` (*pd.DataFrame*): output of
          :func:`risk_attribution_table`.
        - ``"risk_budget"`` (*pd.DataFrame*): output of
          :func:`risk_budget_analysis` (with ticker labels aligned to
          the supplied *tickers* list).
        - ``"incremental_var"`` (*np.ndarray*): incremental VaR array,
          shape ``(n_assets,)``.
        - ``"concentration"`` (*float*): Herfindahl-Hirschman Index of
          risk contributions, computed as
          ``sum((pct_i / 100) ** 2)``.  A value of 1.0 means all risk
          is concentrated in a single asset; ``1/n`` means perfectly
          diversified.
    """
    weights = np.asarray(weights, dtype=float).ravel()

    # Estimate covariance from historical returns.
    cov_matrix = sample_covariance(returns)

    # Total portfolio VaR (parametric, zero-drift).
    port_var = parametric_var_from_cov(weights, cov_matrix, confidence)

    # Attribution table.
    attr_table = risk_attribution_table(tickers, weights, cov_matrix, confidence)

    # Risk budget analysis -- override generic asset labels with tickers.
    risk_budget = risk_budget_analysis(weights, cov_matrix, confidence=confidence)
    risk_budget["ticker"] = list(tickers)

    # Incremental VaR.
    ivar = incremental_var(weights, cov_matrix, confidence)

    # Concentration: Herfindahl-Hirschman Index of risk contributions.
    rpct = risk_contribution_pct(weights, cov_matrix, confidence)
    hhi = float(np.sum((rpct / 100.0) ** 2))

    return {
        "portfolio_var": port_var,
        "attribution_table": attr_table,
        "risk_budget": risk_budget,
        "incremental_var": ivar,
        "concentration": hhi,
    }
