"""VaR Risk Engine — main entry point.

Fetches data for a demo portfolio, computes VaR/ES with three methods,
runs covariance analysis, performs rolling backtesting, and generates
publication-quality visualizations saved to reports/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from var_risk_engine.portfolio import Portfolio
from var_risk_engine.data import fetch_and_prepare
from var_risk_engine.data_validation import has_errors, validate_prices, validate_returns
from var_risk_engine.var_historical import historical_var
from var_risk_engine.var_parametric import parametric_var, parametric_var_from_cov
from var_risk_engine.var_montecarlo import montecarlo_var_with_details
from var_risk_engine.expected_shortfall import (
    expected_shortfall,
    es_parametric,
    compare_var_es,
)
from var_risk_engine.covariance import (
    sample_covariance,
    shrinkage_covariance,
    ewma_covariance,
    correlation_from_covariance,
    cholesky_factor,
)
from var_risk_engine.backtesting import (
    rolling_backtest,
    kupiec_pof_test,
    christoffersen_test,
    basel_traffic_light,
)
from var_risk_engine.volatility import (
    fit_garch11,
    conditional_volatility,
    forecast_volatility,
    compare_garch_models,
)
from var_risk_engine.stress_testing import (
    historical_stress_scenarios,
    hypothetical_stress,
    reverse_stress_test,
    stress_comparison_table,
)
from var_risk_engine.evt import (
    fit_gpd,
    evt_var,
    evt_es,
    compare_evt_methods,
    mean_excess_table,
    threshold_stability_table,
)
from var_risk_engine.copula import (
    fit_gaussian_copula,
    fit_t_copula,
    copula_var,
    compare_copulas,
)
from var_risk_engine.risk_attribution import (
    risk_attribution_summary,
)
from var_risk_engine.reporting.export import (
    prepare_output_dir,
    write_html_report,
    write_json,
    write_table,
)
from var_risk_engine.reporting import plots as report_plots

# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("var_risk_engine")


# ---------------------------------------------------------------------------
# Helper: print section header
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# 1. Portfolio setup & data
# ---------------------------------------------------------------------------

def build_portfolio(
    tickers: list[str] | None = None,
    weights: np.ndarray | None = None,
    start: str = "2019-01-01",
    end: str | None = None,
    input_path: str | Path | None = None,
) -> tuple[Portfolio, pd.DataFrame, pd.DataFrame]:
    """Define demo portfolio and fetch data."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "SPY", "TLT", "GLD"]
    if weights is None:
        weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])

    portfolio = Portfolio(
        name="Multi-Asset Risk Demo",
        tickers=tickers,
        weights=weights,
        benchmark="SPY",
    )
    portfolio.describe()

    prices, returns = fetch_and_prepare(tickers, start=start, end=end, input_path=input_path)
    logger.info("Fetched %d trading days for %d assets.", len(returns), len(tickers))

    return portfolio, prices, returns


# ---------------------------------------------------------------------------
# 2. Compute portfolio returns
# ---------------------------------------------------------------------------

def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted portfolio return series."""
    port_ret = returns.values @ weights
    return port_ret


# ---------------------------------------------------------------------------
# 3. VaR/ES comparison
# ---------------------------------------------------------------------------

def run_var_comparison(
    port_returns: np.ndarray,
    weights: np.ndarray,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    confidence: float = 0.95,
    n_sims: int = 20_000,
) -> dict:
    """Compute VaR/ES with all three methods and print results."""
    _section("VaR & Expected Shortfall Comparison")

    # --- Historical ---
    h_var = historical_var(port_returns, confidence=confidence)
    h_es = expected_shortfall(port_returns, confidence=confidence)

    # --- Parametric ---
    p_var = parametric_var(port_returns, confidence=confidence)
    p_es = es_parametric(port_returns, confidence=confidence)

    # Also demo parametric from covariance matrix
    cov = sample_covariance(returns)
    p_var_cov = parametric_var_from_cov(weights, cov, confidence=confidence)

    # --- Monte Carlo ---
    S0 = prices.iloc[-1].values
    mu = returns.mean().values * 252  # annualised
    sigma = returns.std().values * np.sqrt(252)  # annualised
    corr = correlation_from_covariance(cov)

    mc_details = montecarlo_var_with_details(
        weights=weights,
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr,
        confidence=confidence,
        n_sims=n_sims,
        T=1,
        seed=42,
    )
    mc_var = mc_details["var"]
    mc_es = mc_details["es"]

    # --- GARCH-adjusted Monte Carlo ---
    # Replace each asset's unconditional volatility with a one-day-ahead
    # GARCH forecast, while keeping the historical correlation matrix.
    garch_mc_details: dict | None = None
    garch_sigma_forecasts: list[float] = []
    try:
        for ticker in returns.columns:
            garch_result = fit_garch11(returns[ticker], dist="studentst")
            garch_sigma_forecasts.append(float(forecast_volatility(garch_result, horizon=1)[0]))

        garch_sigma = np.array(garch_sigma_forecasts) * np.sqrt(252)
        garch_mc_details = montecarlo_var_with_details(
            weights=weights,
            S0=S0,
            mu=mu,
            sigma=garch_sigma,
            corr_matrix=corr,
            confidence=confidence,
            n_sims=n_sims,
            T=1,
            seed=43,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("GARCH-adjusted Monte Carlo skipped: %s", exc)

    results = {
        "historical": {"var": h_var, "es": h_es},
        "parametric": {"var": p_var, "es": p_es, "var_from_cov": p_var_cov},
        "montecarlo": {"var": mc_var, "es": mc_es, "simulated_returns": mc_details["simulated_returns"]},
    }
    if garch_mc_details is not None:
        results["garch_montecarlo"] = {
            "var": garch_mc_details["var"],
            "es": garch_mc_details["es"],
            "simulated_returns": garch_mc_details["simulated_returns"],
            "daily_sigma_forecast": np.array(garch_sigma_forecasts),
        }

    # Print summary table
    header = f"{'Method':<18s} {f'VaR ({confidence:.0%})':>12s} {f'ES ({confidence:.0%})':>12s}"
    print(header)
    print("-" * len(header))
    print(f"{'Historical':<18s} {h_var:>12.4f} {h_es:>12.4f}")
    print(f"{'Parametric':<18s} {p_var:>12.4f} {p_es:>12.4f}")
    print(f"{'  (from cov.)':<18s} {p_var_cov:>12.4f} {'—':>12s}")
    print(f"{'Monte Carlo':<18s} {mc_var:>12.4f} {mc_es:>12.4f}")
    if garch_mc_details is not None:
        print(
            f"{'GARCH-MC':<18s} {garch_mc_details['var']:>12.4f} "
            f"{garch_mc_details['es']:>12.4f}"
        )

    # Multi-confidence table
    print("\n--- Multi-Confidence Comparison ---")
    comparison_df = compare_var_es(port_returns)
    print(comparison_df.to_string(index=False, float_format="%.4f"))

    return results


# ---------------------------------------------------------------------------
# 4. Covariance & correlation analysis
# ---------------------------------------------------------------------------

def run_covariance_analysis(returns: pd.DataFrame) -> None:
    """Visualise covariance methods and Cholesky decomposition."""
    _section("Covariance & Correlation Analysis")

    cov_methods = {
        "Sample": sample_covariance(returns),
        "Ledoit-Wolf Shrinkage": shrinkage_covariance(returns),
        "EWMA (λ=0.94)": ewma_covariance(returns),
    }

    corr = correlation_from_covariance(cov_methods["Sample"])
    tickers = list(returns.columns)
    L = cholesky_factor(cov_methods["Sample"])

    report_plots.plot_correlation_comparison(cov_methods, tickers, REPORTS_DIR)
    logger.info("Saved correlation_comparison.png")
    report_plots.plot_cholesky_factor(L, tickers, REPORTS_DIR)
    logger.info("Saved cholesky_factor.png")

    # Verify: L @ L^T ≈ Σ
    reconstructed = L @ L.T
    error = np.max(np.abs(reconstructed - cov_methods["Sample"]))
    print(f"Cholesky reconstruction max error: {error:.2e}")
    print(f"Correlation matrix (sample):\n{pd.DataFrame(corr, index=tickers, columns=tickers).round(3).to_string()}")


# ---------------------------------------------------------------------------
# 7. Backtesting
# ---------------------------------------------------------------------------

def run_backtesting(
    port_returns: np.ndarray,
    confidence: float = 0.95,
    window: int = 250,
) -> None:
    """Run rolling backtest and generate backtesting visualisation."""
    _section("VaR Backtesting")

    bt = rolling_backtest(
        returns=port_returns,
        var_func=historical_var,
        confidence=confidence,
        window=window,
    )

    n_viol = bt["n_violations"]
    n_total = len(bt["var_forecasts"])
    print(f"Rolling backtest: {n_total} forecasts, {n_viol} violations "
          f"(expected ~{n_total * (1 - confidence):.1f})")

    # Kupiec POF test
    kupiec = kupiec_pof_test(bt["actual_returns"], bt["var_forecasts"], confidence)
    print("\nKupiec POF Test:")
    print(f"  Violations : {kupiec['n_violations']} / {kupiec['n_obs']} "
          f"(rate = {kupiec['actual_rate']:.3f}, expected = {1 - confidence:.3f})")
    print(f"  Statistic  : {kupiec['test_statistic']:.4f}")
    print(f"  p-value    : {kupiec['p_value']:.4f}")
    print(f"  Reject H0  : {kupiec['reject_h0']}")

    # Christoffersen test
    christ = christoffersen_test(bt["actual_returns"], bt["var_forecasts"], confidence)
    print("\nChristoffersen Conditional Coverage Test:")
    print(f"  Statistic  : {christ['test_statistic']:.4f}")
    print(f"  p-value    : {christ['p_value']:.4f}")
    print(f"  Reject H0  : {christ['reject_h0']}")
    print(f"  Transitions: 00={christ['n00']}, 01={christ['n01']}, "
          f"10={christ['n10']}, 11={christ['n11']}")

    # Basel traffic-light rules are calibrated for 99% VaR over a 250-day window.
    basel_label = "N/A"
    if np.isclose(confidence, 0.99) and n_total == 250:
        tl = basel_traffic_light(n_viol, n_total)
        basel_label = tl["zone"].upper()
        print(f"\nBasel Traffic Light: {basel_label}")
        print(f"  Scaling factor : {tl['scaling_factor']:.2f}")
        print(f"  {tl['interpretation']}")
    else:
        print("\nBasel Traffic Light: not applied")
        print(
            "  Basel traffic-light thresholds are defined for 99% VaR over "
            "250 backtesting observations; this run uses "
            f"{confidence:.0%} VaR over {n_total} observations."
        )

    report_plots.plot_backtesting(bt, confidence, basel_label, REPORTS_DIR)
    logger.info("Saved backtesting.png")


# ---------------------------------------------------------------------------
# 8. GARCH volatility modelling
# ---------------------------------------------------------------------------

def run_garch_analysis(returns: pd.DataFrame, weights: np.ndarray) -> None:
    """Fit GARCH models and visualise conditional volatility."""
    _section("GARCH Volatility Modelling")

    tickers = list(returns.columns)
    returns_dict = {t: returns[t] for t in tickers}

    # Compare models across assets
    print("Fitting GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1,1) with Student-t ...")
    comparison = compare_garch_models(returns_dict)
    print(comparison[["ticker", "model", "aic", "bic", "alpha", "beta", "gamma"]]
          .to_string(index=False, float_format="%.4f"))

    # Fit GARCH(1,1) on the actual weighted portfolio return series.
    port_ret = pd.Series(returns.values @ weights, index=returns.index, name="portfolio")
    garch_result = fit_garch11(port_ret, dist="studentst")
    cond_vol = conditional_volatility(garch_result)
    forecast_10d = forecast_volatility(garch_result, horizon=10)

    print(f"\nPortfolio GARCH(1,1) — latest conditional vol (daily): {cond_vol.iloc[-1]:.5f}")
    print(f"10-day vol forecast: {forecast_10d.round(5)}")

    report_plots.plot_garch_volatility(returns, port_ret, cond_vol, REPORTS_DIR)
    logger.info("Saved garch_volatility.png")

    vol_by_ticker = {}
    for ticker in tickers:
        res = fit_garch11(returns[ticker], dist="studentst")
        vol_by_ticker[ticker] = conditional_volatility(res)
    report_plots.plot_garch_per_asset(vol_by_ticker, REPORTS_DIR)
    logger.info("Saved garch_per_asset.png")


# ---------------------------------------------------------------------------
# 9. Stress testing
# ---------------------------------------------------------------------------

def run_stress_testing(returns: pd.DataFrame, weights: np.ndarray) -> None:
    """Run stress tests and generate comparison visualisation."""
    _section("Stress Testing")

    # --- Historical stress scenarios ---
    print("--- Historical Stress Periods ---")
    hist_stress = historical_stress_scenarios(returns, weights)
    print(hist_stress.to_string(float_format="%.4f"))

    # --- Hypothetical stress scenarios ---
    print("\n--- Hypothetical Stress Scenarios ---")
    hypo_stress = hypothetical_stress(returns, weights)
    print(hypo_stress.to_string(float_format="%.4f"))

    # --- Reverse stress test ---
    print("\n--- Reverse Stress Test (max loss = 10%) ---")
    rst = reverse_stress_test(returns, weights, max_loss=0.10)
    print(f"  Percentile        : {rst['percentile']:.1f}th")
    print(f"  Uniform drop req. : {rst['uniform_drop_required']:.2%}")
    print(f"  Vol multiplier    : {rst['vol_multiplier']:.2f}x")
    print(f"  {rst['interpretation']}")

    # --- Master comparison table ---
    master = stress_comparison_table(returns, weights)
    print("\n--- Stress Comparison (worst first) ---")
    print(master.to_string(index=False, float_format="%.4f"))

    report_plots.plot_stress_testing(hist_stress, hypo_stress, REPORTS_DIR)
    logger.info("Saved stress_testing.png")


# ---------------------------------------------------------------------------
# 10. EVT (Extreme Value Theory) analysis
# ---------------------------------------------------------------------------

def run_evt_analysis(port_returns: np.ndarray) -> None:
    """Fit GPD to tail losses, compare EVT-VaR with other methods, and plot."""
    _section("Extreme Value Theory (EVT)")

    losses = -port_returns

    # Fit GPD
    gpd_params = fit_gpd(losses)
    print("GPD fit results:")
    print(f"  Shape (xi)       : {gpd_params['xi']:.4f}")
    print(f"  Scale (beta)     : {gpd_params['beta']:.4f}")
    print(f"  Threshold        : {gpd_params['threshold']:.4f}")
    print(f"  Exceedances      : {gpd_params['n_exceedances']} / {gpd_params['n_total']}")
    print(f"  KS statistic     : {gpd_params['ks_statistic']:.4f}  (p={gpd_params['ks_pvalue']:.4f})")

    mean_excess = mean_excess_table(losses)
    stability = threshold_stability_table(losses)
    print("\n--- EVT Threshold Diagnostics ---")
    print(stability.to_string(index=False, float_format="%.4f"))

    # EVT VaR & ES at 99%
    e_var99 = evt_var(gpd_params, confidence=0.99)
    e_es99 = evt_es(gpd_params, confidence=0.99)
    print(f"\nEVT-VaR (99%)  : {e_var99:.4f}")
    print(f"EVT-ES  (99%)  : {e_es99:.4f}")

    # Multi-confidence comparison
    print("\n--- EVT vs Historical vs Parametric ---")
    evt_cmp = compare_evt_methods(port_returns)
    print(evt_cmp.to_string(index=False, float_format="%.4f"))

    report_plots.plot_evt_analysis(losses, gpd_params, mean_excess, stability, evt_cmp, REPORTS_DIR)
    logger.info("Saved evt_gpd_qq.png")
    logger.info("Saved evt_threshold_diagnostics.png")
    logger.info("Saved evt_var_comparison.png")


# ---------------------------------------------------------------------------
# 11. Copula modelling
# ---------------------------------------------------------------------------

def run_copula_analysis(returns: pd.DataFrame, weights: np.ndarray) -> None:
    """Fit Gaussian and t-Copula, compare tail dependence, compute copula VaR."""
    _section("Copula Modelling — Gaussian vs t-Copula")

    # Fit copulas
    gauss_params = fit_gaussian_copula(returns)
    t_params = fit_t_copula(returns)

    print(f"Gaussian Copula: n_obs={gauss_params['n_obs']}")
    print(f"t-Copula: nu={t_params['nu']:.0f}, loglik={t_params['loglikelihood']:.2f}")

    # Tail dependence comparison
    print("\n--- Tail Dependence Comparison ---")
    cop_cmp = compare_copulas(returns)
    print(cop_cmp.to_string(index=False, float_format="%.4f"))

    # Copula VaR
    var_gauss = copula_var(returns, weights, gauss_params, confidence=0.95)
    var_t = copula_var(returns, weights, t_params, confidence=0.95)
    print("\nCopula VaR (95%):")
    print(f"  Gaussian Copula : {var_gauss:.4f}")
    print(f"  t-Copula        : {var_t:.4f}")
    print(f"  Difference      : {var_t - var_gauss:.4f}  "
          f"({'t > Gauss' if var_t > var_gauss else 'Gauss >= t'})")

    tickers = list(returns.columns)
    report_plots.plot_copula_analysis(tickers, gauss_params, t_params, REPORTS_DIR)
    logger.info("Saved copula_tail_dependence.png")
    logger.info("Saved copula_correlation.png")


# ---------------------------------------------------------------------------
# 12. Risk attribution
# ---------------------------------------------------------------------------

def run_risk_attribution(
    tickers: list[str],
    weights: np.ndarray,
    returns: pd.DataFrame,
) -> None:
    """Decompose portfolio VaR into per-asset risk contributions."""
    _section("Risk Attribution — Component & Marginal VaR")

    summary = risk_attribution_summary(tickers, weights, returns)

    print(f"Portfolio VaR (parametric, 95%): {summary['portfolio_var']:.4f}")
    print(f"Risk concentration (HHI): {summary['concentration']:.4f}  "
          f"(1/n={1/len(tickers):.4f} = perfectly diversified)")

    print("\n--- Risk Attribution Table ---")
    print(summary["attribution_table"].to_string(index=False, float_format="%.4f"))

    print("\n--- Risk Budget Analysis (Equal Risk Contribution target) ---")
    print(summary["risk_budget"].to_string(index=False, float_format="%.2f"))

    print("\n--- Incremental VaR ---")
    ivar = summary["incremental_var"]
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: I-VaR = {ivar[i]:.4f}")

    report_plots.plot_risk_attribution(summary, REPORTS_DIR)
    logger.info("Saved risk_attribution.png")
    logger.info("Saved risk_marginal_component.png")


def _effective_backtest_window(n_obs: int, requested_window: int) -> int:
    """Choose a valid rolling backtest window for the available sample."""
    if n_obs < 30:
        raise ValueError("Need at least 30 return observations for the full analysis pipeline.")
    return min(requested_window, max(20, n_obs // 2))


def export_outputs(
    output_path: Path,
    portfolio: Portfolio,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    port_returns: np.ndarray,
    results: dict,
    confidence: float,
    backtest_window: int,
) -> None:
    """Export machine-readable metrics, tables, and a compact HTML report."""
    effective_window = _effective_backtest_window(len(port_returns), backtest_window)
    bt = rolling_backtest(
        returns=port_returns,
        var_func=historical_var,
        confidence=confidence,
        window=effective_window,
    )
    backtesting_df = pd.DataFrame(
        {
            "date_index": bt["dates_idx"],
            "actual_return": bt["actual_returns"],
            "var_forecast": bt["var_forecasts"],
            "violation": bt["violations"],
        }
    )

    var_es_table = compare_var_es(port_returns)
    stress_results = stress_comparison_table(returns, portfolio.weights)
    risk_summary = risk_attribution_summary(
        portfolio.tickers,
        portfolio.weights,
        returns,
        confidence=confidence,
    )
    risk_attribution = risk_summary["attribution_table"]
    price_issues = validate_prices(prices)
    return_issues = validate_returns(returns)
    data_quality = pd.concat(
        [
            price_issues.assign(source="prices"),
            return_issues.assign(source="returns"),
        ],
        ignore_index=True,
    )

    metrics = {
        "portfolio": {
            "tickers": portfolio.tickers,
            "weights": [float(x) for x in portfolio.weights],
            "n_assets": portfolio.n_assets,
        },
        "data": {
            "n_price_observations": int(len(prices)),
            "n_return_observations": int(len(returns)),
            "start": str(prices.index.min().date()),
            "end": str(prices.index.max().date()),
            "data_quality_has_errors": has_errors(data_quality),
        },
        "risk": {
            "confidence": confidence,
            "historical_var": float(results["historical"]["var"]),
            "historical_es": float(results["historical"]["es"]),
            "parametric_var": float(results["parametric"]["var"]),
            "parametric_es": float(results["parametric"]["es"]),
            "monte_carlo_var": float(results["montecarlo"]["var"]),
            "monte_carlo_es": float(results["montecarlo"]["es"]),
            "garch_monte_carlo_var": float(
                results.get("garch_montecarlo", {}).get("var", np.nan)
            ),
            "garch_monte_carlo_es": float(
                results.get("garch_montecarlo", {}).get("es", np.nan)
            ),
        },
        "backtesting": {
            "window": int(effective_window),
            "n_forecasts": int(len(bt["var_forecasts"])),
            "n_violations": int(bt["n_violations"]),
            "violation_rate": float(np.mean(bt["violations"])),
        },
    }

    tables = {
        "var_es_table": var_es_table,
        "backtesting_results": backtesting_df,
        "stress_results": stress_results,
        "risk_attribution": risk_attribution,
        "data_quality": data_quality,
    }

    write_json(output_path / "metrics.json", metrics)
    write_table(output_path / "var_es_table.csv", var_es_table)
    write_table(output_path / "backtesting_results.csv", backtesting_df)
    write_table(output_path / "stress_results.csv", stress_results)
    write_table(output_path / "risk_attribution.csv", risk_attribution)
    write_table(output_path / "data_quality.csv", data_quality)

    figure_names = sorted(path.name for path in (output_path / "figures").glob("*.png"))
    write_html_report(output_path / "report.html", metrics, tables, figure_names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    tickers: list[str] | None = None,
    weights: np.ndarray | None = None,
    start: str = "2019-01-01",
    end: str | None = None,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = "outputs/latest",
    confidence: float = 0.95,
    n_sims: int = 20_000,
    backtest_window: int = 500,
) -> None:
    """Run the full VaR risk analysis pipeline."""
    global REPORTS_DIR

    output_path = prepare_output_dir(output_dir)
    REPORTS_DIR = output_path / "figures"

    print("\n" + "=" * 60)
    print("  VaR Risk Engine v0.3.0")
    print("  Historical | Parametric | Monte Carlo VaR & ES")
    print("  + GARCH | Stress Testing | EVT | Copula | Risk Attribution")
    print("=" * 60)

    # 1. Portfolio & data
    _section("1. Portfolio Setup & Data Acquisition")
    portfolio, prices, returns = build_portfolio(
        tickers=tickers,
        weights=weights,
        start=start,
        end=end,
        input_path=input_path,
    )
    port_returns = compute_portfolio_returns(returns, portfolio.weights)
    price_issues = validate_prices(prices)
    return_issues = validate_returns(returns)
    if has_errors(price_issues) or has_errors(return_issues):
        raise ValueError(
            "Input data failed validation. See data_quality.csv in the output "
            "directory when running through the export pipeline."
        )
    logger.info("Portfolio returns: %d obs, mean=%.5f, std=%.4f",
                len(port_returns), np.mean(port_returns), np.std(port_returns))

    # 2. VaR/ES comparison
    _section("2. VaR & Expected Shortfall Estimation")
    results = run_var_comparison(
        port_returns,
        portfolio.weights,
        prices,
        returns,
        confidence=confidence,
        n_sims=n_sims,
    )

    # 3. Covariance analysis
    _section("3. Covariance & Correlation Analysis")
    run_covariance_analysis(returns)

    # 4. Distribution plots
    _section("4. Distribution & VaR Visualisation")
    report_plots.plot_var_distributions(
        port_returns,
        results["montecarlo"]["simulated_returns"],
        REPORTS_DIR,
        confidence=confidence,
    )
    report_plots.plot_montecarlo(
        results["montecarlo"]["simulated_returns"],
        REPORTS_DIR,
        confidence=confidence,
    )

    # 5. Backtesting
    _section("5. Backtesting")
    effective_backtest_window = _effective_backtest_window(len(port_returns), backtest_window)
    run_backtesting(port_returns, confidence=confidence, window=effective_backtest_window)

    # 6. GARCH volatility modelling
    _section("6. GARCH Volatility Modelling")
    run_garch_analysis(returns, portfolio.weights)

    # 7. Stress testing
    _section("7. Stress Testing")
    run_stress_testing(returns, portfolio.weights)

    # 8. EVT analysis
    _section("8. Extreme Value Theory")
    run_evt_analysis(port_returns)

    # 9. Copula modelling
    _section("9. Copula Modelling")
    run_copula_analysis(returns, portfolio.weights)

    # 10. Risk attribution
    _section("10. Risk Attribution")
    run_risk_attribution(portfolio.tickers, portfolio.weights, returns)

    export_outputs(
        output_path=output_path,
        portfolio=portfolio,
        prices=prices,
        returns=returns,
        port_returns=port_returns,
        results=results,
        confidence=confidence,
        backtest_window=backtest_window,
    )

    # Summary
    _section("Output Summary")
    print(f"All figures saved to: {REPORTS_DIR}")
    print("  - correlation_comparison.png")
    print("  - cholesky_factor.png")
    print("  - var_distribution.png")
    print("  - montecarlo_var.png")
    print("  - backtesting.png")
    print("  - garch_volatility.png")
    print("  - garch_per_asset.png")
    print("  - stress_testing.png")
    print("  - evt_gpd_qq.png")
    print("  - evt_threshold_diagnostics.png")
    print("  - evt_var_comparison.png")
    print("  - copula_tail_dependence.png")
    print("  - copula_correlation.png")
    print("  - risk_attribution.png")
    print("  - risk_marginal_component.png")
    print(f"\nStructured outputs saved to: {output_path}")
    print("  - metrics.json")
    print("  - var_es_table.csv")
    print("  - backtesting_results.csv")
    print("  - stress_results.csv")
    print("  - risk_attribution.csv")
    print("  - data_quality.csv")
    print("  - report.html")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
