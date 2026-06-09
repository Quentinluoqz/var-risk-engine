"""GARCH-family volatility modelling for portfolio risk analysis.

Replaces the constant-volatility assumption in Monte Carlo VaR with
time-varying conditional volatility from GARCH(1,1), EGARCH(1,1),
and GJR-GARCH(1,1) models.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arch.univariate.base  # noqa: F401  (type-checking only)


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def _import_arch():
    """Import and return the ``arch`` package, raising ImportError with a
    helpful message when it is not installed."""
    try:
        import arch  # noqa: F811
        return arch
    except ImportError as exc:
        raise ImportError(
            "The 'arch' package is required for volatility modelling. "
            "Install it with:  pip install arch"
        ) from exc


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_garch11(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> "arch.univariate.base.ARCHModelResult":
    """Fit a GARCH(p, q) model to a single-asset return series.

    The *arch* package expects returns in **percentage** form (e.g. 0.5 for
    0.5 %).  This function accepts decimal returns (e.g. 0.005) and scales
    them internally by multiplying by 100.

    Args:
        returns: Pandas Series of daily log returns in decimal form.
        p: Lag order for the GARCH variance term (default 1).
        q: Lag order for the GARCH residual term (default 1).
        dist: Distribution for the standardized residuals.  One of
            ``"normal"``, ``"studentst"``, or ``"ged"``.

    Returns:
        Fitted ``ARCHModelResult`` instance.

    Raises:
        RuntimeError: If the model fails to converge or *arch* raises any
            other fitting error.
        ImportError: If the *arch* package is not installed.
    """
    arch = _import_arch()

    if dist not in ("normal", "studentst", "ged"):
        raise ValueError(
            f"Unsupported distribution '{dist}'. "
            "Choose from 'normal', 'studentst', or 'ged'."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch.arch_model(
                returns * 100,
                vol="Garch",
                p=p,
                q=q,
                dist=dist,
                rescale=False,
            )
            result = model.fit(disp="off")
    except Exception as exc:
        raise RuntimeError(
            f"GARCH({p},{q}) fitting failed for series "
            f"'{returns.name or '<unnamed>'}': {exc}"
        ) from exc

    return result


def fit_egarch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> "arch.univariate.base.ARCHModelResult":
    """Fit an EGARCH(p, q) model to capture leverage effects.

    The Exponential GARCH model allows negative returns to increase
    volatility more than positive returns of the same magnitude,
    capturing the well-known *leverage effect* in equity markets.

    Args:
        returns: Pandas Series of daily log returns in decimal form.
        p: Lag order for the EGARCH variance term (default 1).
        q: Lag order for the EGARCH innovation term (default 1).

    Returns:
        Fitted ``ARCHModelResult`` instance.

    Raises:
        RuntimeError: If the model fails to converge.
        ImportError: If the *arch* package is not installed.
    """
    arch = _import_arch()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch.arch_model(
                returns * 100,
                vol="EGARCH",
                p=p,
                q=q,
            )
            result = model.fit(disp="off")
    except Exception as exc:
        raise RuntimeError(
            f"EGARCH({p},{q}) fitting failed for series "
            f"'{returns.name or '<unnamed>'}': {exc}"
        ) from exc

    return result


def fit_gjr_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    o: int = 1,
) -> "arch.univariate.base.ARCHModelResult":
    """Fit a GJR-GARCH(p, o, q) model (asymmetric GARCH).

    The Glosten-Jagannathan-Runkle GARCH model adds an asymmetric term
    controlled by parameter *o* so that negative shocks have a larger
    impact on conditional variance than positive shocks of the same size.

    Args:
        returns: Pandas Series of daily log returns in decimal form.
        p: Lag order for the GARCH variance term (default 1).
        q: Lag order for the GARCH residual term (default 1).
        o: Order of the asymmetric leverage term (default 1).

    Returns:
        Fitted ``ARCHModelResult`` instance.

    Raises:
        RuntimeError: If the model fails to converge.
        ImportError: If the *arch* package is not installed.
    """
    arch = _import_arch()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch.arch_model(
                returns * 100,
                vol="Garch",
                p=p,
                o=o,
                q=q,
            )
            result = model.fit(disp="off")
    except Exception as exc:
        raise RuntimeError(
            f"GJR-GARCH({p},{o},{q}) fitting failed for series "
            f"'{returns.name or '<unnamed>'}': {exc}"
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Post-estimation helpers
# ---------------------------------------------------------------------------

def conditional_volatility(
    result: "arch.univariate.base.ARCHModelResult",
) -> pd.Series:
    """Extract the in-sample conditional volatility from a fitted model.

    The *arch* package stores conditional volatility in percentage terms.
    This function divides by 100 so that the returned values are on the
    same decimal scale as the original input returns.

    Args:
        result: A fitted ``ARCHModelResult`` produced by one of the
            ``fit_*`` functions in this module.

    Returns:
        Pandas Series with a ``DatetimeIndex`` (matching the input data)
        containing the conditional volatility in decimal form.
    """
    cond_vol = result.conditional_volatility
    if cond_vol is None:
        raise ValueError(
            "The fitted model does not contain a conditional_volatility "
            "attribute.  Ensure the model was fitted successfully."
        )
    return cond_vol / 100.0


def forecast_volatility(
    result: "arch.univariate.base.ARCHModelResult",
    horizon: int = 1,
) -> np.ndarray:
    """Forecast conditional volatility for the next *horizon* days.

    Uses the fitted model's ``forecast`` method to produce out-of-sample
    variance forecasts, then converts them to volatility (standard
    deviation) in **decimal** form (i.e. divided by 100 to undo the
    percentage scaling used internally by *arch*).

    Args:
        result: A fitted ``ARCHModelResult``.
        horizon: Number of forward trading days to forecast (default 1).

    Returns:
        1-D ``numpy.ndarray`` of length *horizon* containing daily
        volatility forecasts in decimal form.

    Raises:
        RuntimeError: If the forecast call fails.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fcast = result.forecast(horizon=horizon)
    except Exception as exc:
        raise RuntimeError(
            f"Volatility forecast failed (horizon={horizon}): {exc}"
        ) from exc

    # fcast.variance is a DataFrame with shape (n_obs, horizon).
    # Only the last row contains valid out-of-sample forecasts.
    variance_row = fcast.variance.iloc[-1]        # Series of length horizon
    volatility_pct = np.sqrt(variance_row.values)  # still in percentage
    return volatility_pct / 100.0


def garch_filtered_returns(
    returns: pd.Series,
    result: "arch.univariate.base.ARCHModelResult",
) -> np.ndarray:
    """Compute GARCH-standardized residuals (filtered returns).

    Divides the original return series by the model's conditional
    volatility estimate.  If the GARCH model captures the volatility
    dynamics well, the resulting series should be approximately i.i.d.
    N(0, 1).

    Both *returns* and the internal conditional volatility are aligned
    by index; periods where conditional volatility is ``NaN`` (typically
    the first few observations used for initialization) are dropped
    before the division.

    Args:
        returns: The same return Series that was passed to the fitting
            function (decimal form).
        result: The corresponding fitted ``ARCHModelResult``.

    Returns:
        1-D ``numpy.ndarray`` of standardized residuals with NaN
        observations removed.
    """
    cond_vol = conditional_volatility(result)  # decimal scale

    # Align the two Series by index and drop NaN entries that arise
    # from the GARCH filter warm-up period.
    aligned_returns, aligned_vol = returns.align(cond_vol, join="inner")
    mask = aligned_vol.notna() & aligned_returns.notna()

    return (aligned_returns[mask] / aligned_vol[mask]).to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def _extract_gamma(params: pd.Series) -> float:
    """Return the first gamma parameter from an *arch* params Series, or
    ``NaN`` when the model has no asymmetric term."""
    for label, value in params.items():
        if "gamma" in label.lower():
            return float(value)
    return float("nan")


def compare_garch_models(
    returns_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    """Fit multiple GARCH-family models per asset and compare fit statistics.

    For every ticker in *returns_dict* the function fits three models --
    GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1,1) -- all using the
    Student-t distribution for robustness, then assembles a summary
    DataFrame for easy model selection.

    Args:
        returns_dict: Mapping of ticker symbol to a pandas Series of
            daily log returns (decimal form).

    Returns:
        ``pandas.DataFrame`` with one row per (ticker, model) combination
        and the following columns:

        * **ticker** -- the asset identifier.
        * **model** -- one of ``"GARCH"``, ``"EGARCH"``, ``"GJR-GARCH"``.
        * **aic** -- Akaike Information Criterion.
        * **bic** -- Bayesian Information Criterion.
        * **loglikelihood** -- maximized log-likelihood.
        * **omega** -- estimated constant term in the variance equation.
        * **alpha** -- estimated coefficient on lagged squared residual.
        * **beta** -- estimated coefficient on lagged conditional variance.
        * **gamma** -- asymmetric leverage coefficient (``NaN`` for plain
          GARCH which has no asymmetric term).

    Raises:
        RuntimeError: If any individual model fit fails.
        ImportError: If the *arch* package is not installed.
    """
    _import_arch()  # fail early if arch is not available

    records: list[dict] = []

    for ticker, returns in returns_dict.items():
        # -- GARCH(1,1) with Student-t ---------------------------------
        res_garch = fit_garch11(returns, p=1, q=1, dist="studentst")
        records.append({
            "ticker": ticker,
            "model": "GARCH",
            "aic": res_garch.aic,
            "bic": res_garch.bic,
            "loglikelihood": res_garch.loglikelihood,
            "omega": float(res_garch.params.get("omega", np.nan)),
            "alpha": float(res_garch.params.get("alpha[1]", np.nan)),
            "beta": float(res_garch.params.get("beta[1]", np.nan)),
            "gamma": np.nan,
        })

        # -- EGARCH(1,1) with Student-t --------------------------------
        res_egarch = fit_egarch(returns, p=1, q=1)
        records.append({
            "ticker": ticker,
            "model": "EGARCH",
            "aic": res_egarch.aic,
            "bic": res_egarch.bic,
            "loglikelihood": res_egarch.loglikelihood,
            "omega": float(res_egarch.params.get("omega", np.nan)),
            "alpha": float(res_egarch.params.get("alpha[1]", np.nan)),
            "beta": float(res_egarch.params.get("beta[1]", np.nan)),
            "gamma": _extract_gamma(res_egarch.params),
        })

        # -- GJR-GARCH(1,1,1) with Student-t ---------------------------
        res_gjr = fit_gjr_garch(returns, p=1, q=1, o=1)
        records.append({
            "ticker": ticker,
            "model": "GJR-GARCH",
            "aic": res_gjr.aic,
            "bic": res_gjr.bic,
            "loglikelihood": res_gjr.loglikelihood,
            "omega": float(res_gjr.params.get("omega", np.nan)),
            "alpha": float(res_gjr.params.get("alpha[1]", np.nan)),
            "beta": float(res_gjr.params.get("beta[1]", np.nan)),
            "gamma": _extract_gamma(res_gjr.params),
        })

    return pd.DataFrame(records)
