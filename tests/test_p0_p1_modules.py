"""Tests for P0/P1 modules: volatility, stress_testing, evt, copula, risk_attribution."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from var_risk_engine.volatility import (
    fit_garch11,
    fit_egarch,
    fit_gjr_garch,
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
    evt_var_series,
    compare_evt_methods,
    gpd_qq_plot,
    mean_excess_table,
    threshold_stability_table,
)
from var_risk_engine.copula import (
    rank_to_uniform,
    uniform_to_normal,
    fit_gaussian_copula,
    fit_t_copula,
    tail_dependence_gaussian,
    tail_dependence_t,
    simulate_copula,
    copula_var,
    compare_copulas,
)
from var_risk_engine.risk_attribution import (
    marginal_var,
    component_var,
    risk_contribution_pct,
    risk_attribution_table,
    incremental_var,
    risk_budget_analysis,
    risk_attribution_summary,
)
from var_risk_engine.var_parametric import parametric_var_from_cov
from var_risk_engine.cli import build_parser, cli_main
from var_risk_engine.data import compute_returns, fetch_prices
import var_risk_engine.data as data_module
from var_risk_engine.config import config_from_mapping


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_returns_array(rng: np.random.Generator) -> np.ndarray:
    """1000 daily returns ~ N(0.0003, 0.015) as a 1-D numpy array."""
    return rng.normal(0.0003, 0.015, size=1000)


@pytest.fixture
def sample_returns_series(sample_returns_array: np.ndarray) -> pd.Series:
    """Pandas Series of returns with a DatetimeIndex (named 'ASSET')."""
    dates = pd.bdate_range(start="2018-01-01", periods=len(sample_returns_array))
    return pd.Series(sample_returns_array, index=dates, name="ASSET")


@pytest.fixture
def sample_returns_df(rng: np.random.Generator) -> pd.DataFrame:
    """DataFrame with 3 assets, 600 trading days spanning 2018-2022
    (covers COVID and 2022 rate hike stress periods)."""
    n_days = 600
    data = rng.normal(0.0003, 0.01, size=(n_days, 3))
    dates = pd.bdate_range(start="2018-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=["EQ_SP500", "BOND_AGG", "GOLD_GLD"])


@pytest.fixture
def sample_weights() -> np.ndarray:
    """Portfolio weights for 3 assets summing to 1."""
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture
def sample_cov_matrix() -> np.ndarray:
    """A 3x3 symmetric positive-definite covariance matrix."""
    return np.array([
        [0.0004, 0.00006, 0.00002],
        [0.00006, 0.0001, 0.00001],
        [0.00002, 0.00001, 0.0002],
    ])


@pytest.fixture
def fitted_garch_result(sample_returns_series: pd.Series):
    """Pre-fitted GARCH(1,1) result for reuse across tests."""
    return fit_garch11(sample_returns_series)


@pytest.fixture
def losses_array(sample_returns_array: np.ndarray) -> np.ndarray:
    """Positive losses (negated returns) for EVT tests."""
    return -sample_returns_array


@pytest.fixture
def gpd_params(losses_array: np.ndarray) -> dict:
    """Pre-fitted GPD parameters for EVT VaR/ES tests."""
    return fit_gpd(losses_array)


# =========================================================================
# TestVolatility
# =========================================================================

class TestVolatility:
    """Tests for volatility.py -- GARCH(1,1), EGARCH, GJR-GARCH."""

    def test_fit_garch11_returns_result(self, sample_returns_series: pd.Series) -> None:
        result = fit_garch11(sample_returns_series)
        assert hasattr(result, "params")
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert hasattr(result, "loglikelihood")

    def test_fit_garch11_invalid_dist(self, sample_returns_series: pd.Series) -> None:
        with pytest.raises(ValueError, match="Unsupported distribution"):
            fit_garch11(sample_returns_series, dist="invalid_dist")

    def test_fit_egarch_returns_result(self, sample_returns_series: pd.Series) -> None:
        result = fit_egarch(sample_returns_series)
        assert hasattr(result, "params")
        assert hasattr(result, "conditional_volatility")

    def test_fit_gjr_garch_returns_result(self, sample_returns_series: pd.Series) -> None:
        result = fit_gjr_garch(sample_returns_series)
        assert hasattr(result, "params")
        assert hasattr(result, "conditional_volatility")

    def test_conditional_vol_positive(self, fitted_garch_result) -> None:
        cond_vol = conditional_volatility(fitted_garch_result)
        # All non-NaN conditional volatilities should be positive.
        valid = cond_vol.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_conditional_vol_shape(self, fitted_garch_result, sample_returns_series) -> None:
        cond_vol = conditional_volatility(fitted_garch_result)
        assert isinstance(cond_vol, pd.Series)
        assert len(cond_vol) == len(sample_returns_series)

    def test_forecast_volatility_shape(self, fitted_garch_result) -> None:
        horizon = 5
        fcast = forecast_volatility(fitted_garch_result, horizon=horizon)
        assert isinstance(fcast, np.ndarray)
        assert fcast.shape == (horizon,)

    def test_forecast_volatility_positive(self, fitted_garch_result) -> None:
        fcast = forecast_volatility(fitted_garch_result, horizon=10)
        assert (fcast > 0).all()

    def test_compare_garch_models_columns(self, sample_returns_series: pd.Series) -> None:
        returns_dict = {"ASSET": sample_returns_series}
        df = compare_garch_models(returns_dict)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"ticker", "model", "aic", "bic", "loglikelihood",
                         "omega", "alpha", "beta", "gamma"}
        assert expected_cols.issubset(set(df.columns))

    def test_compare_garch_models_rows(self, sample_returns_series: pd.Series) -> None:
        returns_dict = {"ASSET": sample_returns_series}
        df = compare_garch_models(returns_dict)
        # 3 models per ticker, 1 ticker => 3 rows
        assert len(df) == 3
        assert set(df["model"].tolist()) == {"GARCH", "EGARCH", "GJR-GARCH"}


# =========================================================================
# TestStressTesting
# =========================================================================

class TestStressTesting:
    """Tests for stress_testing.py -- historical, hypothetical, and reverse stress."""

    def test_historical_stress_scenarios_columns(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        df = historical_stress_scenarios(sample_returns_df, sample_weights)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"cumulative_return", "worst_daily_return", "var_95",
                         "es_95", "annualised_vol", "note"}
        assert expected_cols.issubset(set(df.columns))

    def test_historical_stress_scenarios_rows(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        df = historical_stress_scenarios(sample_returns_df, sample_weights)
        # 2 predefined + 1 worst window = 3 scenarios
        assert len(df) == 3

    def test_hypothetical_stress_columns(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        df = hypothetical_stress(sample_returns_df, sample_weights)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"portfolio_return", "loss_bps", "var_95", "es_95", "note"}
        assert expected_cols.issubset(set(df.columns))

    def test_hypothetical_stress_custom_shocks(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        custom_shocks = {
            "Mild Downturn": np.array([-0.10, -0.02, -0.05]),
            "Severe Crash": np.array([-0.30, -0.10, -0.15]),
        }
        df = hypothetical_stress(sample_returns_df, sample_weights, shocks=custom_shocks)
        assert len(df) == 2
        assert "Mild Downturn" in df.index
        assert "Severe Crash" in df.index

    def test_reverse_stress_test_keys(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        result = reverse_stress_test(sample_returns_df, sample_weights, max_loss=0.10)
        assert isinstance(result, dict)
        expected_keys = {"max_loss", "percentile", "uniform_drop_required",
                         "vol_multiplier", "interpretation"}
        assert expected_keys == set(result.keys())

    def test_reverse_stress_test_invalid_max_loss(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="max_loss must be positive"):
            reverse_stress_test(sample_returns_df, sample_weights, max_loss=-0.05)

    def test_stress_comparison_table_columns(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        df = stress_comparison_table(sample_returns_df, sample_weights)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"scenario", "type", "portfolio_return", "var_95",
                         "es_95", "max_daily_loss"}
        assert expected_cols.issubset(set(df.columns))
        # Should have both historical and hypothetical types
        assert "historical" in df["type"].values
        assert "hypothetical" in df["type"].values

    def test_weight_mismatch_raises(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        bad_weights = np.array([0.5, 0.5])  # 2 weights for 3 assets
        with pytest.raises(ValueError, match="weights length"):
            historical_stress_scenarios(sample_returns_df, bad_weights)
        with pytest.raises(ValueError, match="weights length"):
            hypothetical_stress(sample_returns_df, bad_weights)
        with pytest.raises(ValueError, match="weights length"):
            reverse_stress_test(sample_returns_df, bad_weights)
        with pytest.raises(ValueError, match="weights length"):
            stress_comparison_table(sample_returns_df, bad_weights)


# =========================================================================
# TestEVT
# =========================================================================

class TestEVT:
    """Tests for evt.py -- Extreme Value Theory with GPD."""

    def test_fit_gpd_keys(self, losses_array: np.ndarray) -> None:
        result = fit_gpd(losses_array)
        assert isinstance(result, dict)
        expected_keys = {"xi", "beta", "threshold", "n_exceedances",
                         "n_total", "exceedance_rate", "ks_statistic", "ks_pvalue"}
        assert expected_keys == set(result.keys())

    def test_fit_gpd_xi_reasonable(self, losses_array: np.ndarray) -> None:
        result = fit_gpd(losses_array)
        # For financial returns the GPD shape parameter is typically
        # in (-0.5, 0.5).  We allow a generous range but check it is finite.
        assert np.isfinite(result["xi"])
        assert abs(result["xi"]) < 2.0
        assert result["beta"] > 0
        assert result["n_exceedances"] >= 2
        assert 0 < result["exceedance_rate"] < 1

    def test_fit_gpd_too_few_exceedances(self) -> None:
        # An array where almost nothing exceeds a very high threshold.
        losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        with pytest.raises(ValueError, match="exceedance"):
            fit_gpd(losses, threshold=0.049)

    def test_evt_var_positive(self, gpd_params: dict) -> None:
        var = evt_var(gpd_params, confidence=0.99)
        assert isinstance(var, float)
        assert var > 0

    def test_evt_es_geq_var(self, gpd_params: dict) -> None:
        var = evt_var(gpd_params, confidence=0.99)
        es = evt_es(gpd_params, confidence=0.99)
        assert es >= var

    def test_evt_var_invalid_confidence(self, gpd_params: dict) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            evt_var(gpd_params, confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            evt_var(gpd_params, confidence=1.0)

    def test_evt_es_invalid_confidence(self, gpd_params: dict) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            evt_es(gpd_params, confidence=-0.5)

    def test_compare_evt_methods_columns(self, sample_returns_array: np.ndarray) -> None:
        df = compare_evt_methods(sample_returns_array)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"confidence", "var_historical", "es_historical",
                         "var_parametric", "es_parametric", "var_evt", "es_evt"}
        assert expected_cols == set(df.columns)
        assert len(df) == 5  # default confidence levels

    def test_compare_evt_methods_es_geq_var(
        self, sample_returns_array: np.ndarray
    ) -> None:
        df = compare_evt_methods(sample_returns_array)
        for _, row in df.iterrows():
            assert row["es_historical"] >= row["var_historical"]
            assert row["es_parametric"] >= row["var_parametric"]
            # ES_EVT may be inf if xi >= 1, but should still be >= VaR
            assert row["es_evt"] >= row["var_evt"]

    def test_gpd_qq_plot_returns_figure(
        self, losses_array: np.ndarray, gpd_params: dict
    ) -> None:
        fig, ax = gpd_qq_plot(losses_array, gpd_params)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_evt_var_series_shape(self, sample_returns_array: np.ndarray) -> None:
        window = 500
        series = evt_var_series(
            sample_returns_array,
            confidence=0.99,
            window=window,
        )
        expected_len = len(sample_returns_array) - window + 1
        assert isinstance(series, np.ndarray)
        assert series.shape == (expected_len,)

    def test_evt_var_series_invalid_window(
        self, sample_returns_array: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="window"):
            evt_var_series(sample_returns_array, window=1)

    def test_mean_excess_table_columns(self, losses_array: np.ndarray) -> None:
        df = mean_excess_table(losses_array, quantiles=[0.80, 0.90, 0.95])
        expected_cols = {"quantile", "threshold", "n_exceedances", "mean_excess"}
        assert expected_cols == set(df.columns)
        assert len(df) == 3
        assert (df["n_exceedances"] > 0).all()

    def test_threshold_stability_table_columns(self, losses_array: np.ndarray) -> None:
        df = threshold_stability_table(losses_array, quantiles=[0.80, 0.90, 0.95])
        expected_cols = {"quantile", "threshold", "n_exceedances", "xi", "beta", "ks_pvalue"}
        assert expected_cols == set(df.columns)
        assert len(df) == 3
        assert np.isfinite(df["xi"].dropna()).all()


# =========================================================================
# TestCopula
# =========================================================================

class TestCopula:
    """Tests for copula.py -- Gaussian vs t-Copula."""

    def test_rank_to_uniform_range(self, sample_returns_df: pd.DataFrame) -> None:
        u = rank_to_uniform(sample_returns_df)
        assert isinstance(u, pd.DataFrame)
        assert u.shape == sample_returns_df.shape
        assert (u > 0).all().all()
        assert (u < 1).all().all()

    def test_uniform_to_normal_finite(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        u = rank_to_uniform(sample_returns_df)
        z = uniform_to_normal(u)
        assert isinstance(z, pd.DataFrame)
        assert z.shape == u.shape
        assert np.all(np.isfinite(z.values))

    def test_fit_gaussian_copula_keys(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        params = fit_gaussian_copula(sample_returns_df)
        assert isinstance(params, dict)
        assert params["type"] == "gaussian"
        assert "corr_matrix" in params
        assert params["n_obs"] == len(sample_returns_df)
        # Correlation matrix should be 3x3
        assert params["corr_matrix"].shape == (3, 3)
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(params["corr_matrix"]), 1.0, atol=1e-10)

    def test_fit_t_copula_keys(self, sample_returns_df: pd.DataFrame) -> None:
        params = fit_t_copula(sample_returns_df)
        assert isinstance(params, dict)
        assert params["type"] == "t"
        assert "corr_matrix" in params
        assert "nu" in params
        assert "loglikelihood" in params
        assert params["n_obs"] == len(sample_returns_df)
        assert 2 <= params["nu"] <= 30

    def test_tail_dependence_gaussian_zero(self) -> None:
        # Gaussian copula always has zero tail dependence.
        for rho in [-0.5, 0.0, 0.3, 0.7, 0.99]:
            assert tail_dependence_gaussian(rho) == 0.0

    def test_tail_dependence_t_bounds(self) -> None:
        # t-copula tail dependence should be in [0, 1].
        for rho in [0.0, 0.3, 0.7, 0.9]:
            for nu in [2, 5, 10, 30]:
                td = tail_dependence_t(rho, nu)
                assert 0.0 <= td <= 1.0

    def test_tail_dependence_t_higher_corr_higher_td(self) -> None:
        # For fixed nu, higher correlation => higher tail dependence.
        nu = 5
        td_low = tail_dependence_t(0.2, nu)
        td_high = tail_dependence_t(0.8, nu)
        assert td_high > td_low

    def test_simulate_copula_gaussian_shape(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        params = fit_gaussian_copula(sample_returns_df)
        U = simulate_copula(params, n_sims=500, seed=42)
        assert U.shape == (500, 3)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_simulate_copula_t_shape(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        params = fit_t_copula(sample_returns_df)
        U = simulate_copula(params, n_sims=500, seed=42)
        assert U.shape == (500, 3)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_copula_var_positive(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        params = fit_gaussian_copula(sample_returns_df)
        var_val = copula_var(
            sample_returns_df, sample_weights, params,
            confidence=0.95, n_sims=5000, seed=42,
        )
        assert isinstance(var_val, float)
        assert var_val > 0

    def test_compare_copulas_columns(
        self, sample_returns_df: pd.DataFrame
    ) -> None:
        df = compare_copulas(sample_returns_df)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"asset_i", "asset_j", "correlation",
                         "gaussian_tail_dep", "t_tail_dep", "t_nu"}
        assert expected_cols == set(df.columns)
        # 3 assets => 3 pairs: (0,1), (0,2), (1,2)
        assert len(df) == 3
        # Gaussian tail dependence is always 0
        assert (df["gaussian_tail_dep"] == 0.0).all()
        # t-copula tail dependence >= 0
        assert (df["t_tail_dep"] >= 0.0).all()


# =========================================================================
# TestRiskAttribution
# =========================================================================

class TestRiskAttribution:
    """Tests for risk_attribution.py -- Component/Marginal VaR attribution."""

    def test_marginal_var_shape(
        self, sample_weights: np.ndarray, sample_cov_matrix: np.ndarray
    ) -> None:
        mvar = marginal_var(sample_weights, sample_cov_matrix, confidence=0.95)
        assert isinstance(mvar, np.ndarray)
        assert mvar.shape == sample_weights.shape

    def test_component_var_euler_decomposition(
        self, sample_weights: np.ndarray, sample_cov_matrix: np.ndarray
    ) -> None:
        """Euler's theorem: sum of Component VaR equals portfolio VaR."""
        cvar = component_var(sample_weights, sample_cov_matrix, confidence=0.95)
        port_var = parametric_var_from_cov(
            sample_weights, sample_cov_matrix, confidence=0.95
        )
        assert pytest.approx(np.sum(cvar), abs=1e-6) == port_var

    def test_risk_contribution_pct_sums_to_100(
        self, sample_weights: np.ndarray, sample_cov_matrix: np.ndarray
    ) -> None:
        pct = risk_contribution_pct(
            sample_weights, sample_cov_matrix, confidence=0.95
        )
        assert isinstance(pct, np.ndarray)
        assert pct.shape == sample_weights.shape
        assert pytest.approx(np.sum(pct), abs=1e-4) == 100.0

    def test_risk_attribution_table_columns(
        self, sample_weights: np.ndarray, sample_cov_matrix: np.ndarray
    ) -> None:
        tickers = ["EQ_SP500", "BOND_AGG", "GOLD_GLD"]
        df = risk_attribution_table(
            tickers, sample_weights, sample_cov_matrix, confidence=0.95
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"ticker", "weight", "weight_pct", "component_var",
                         "marginal_var", "risk_contribution_pct",
                         "portfolio_var_total"}
        assert expected_cols == set(df.columns)

    def test_risk_attribution_table_total_row(
        self, sample_weights: np.ndarray, sample_cov_matrix: np.ndarray
    ) -> None:
        tickers = ["EQ_SP500", "BOND_AGG", "GOLD_GLD"]
        df = risk_attribution_table(
            tickers, sample_weights, sample_cov_matrix, confidence=0.95
        )
        # Last row should be TOTAL
        assert df.iloc[-1]["ticker"] == "TOTAL"
        # n_assets + 1 rows (including TOTAL)
        assert len(df) == 4
        # TOTAL risk contribution should sum to ~100
        assert pytest.approx(
            df.iloc[-1]["risk_contribution_pct"], abs=1e-3
        ) == 100.0

    def test_incremental_var_shape(
        self, sample_weights: np.ndarray, sample_cov_matrix: np.ndarray
    ) -> None:
        ivar = incremental_var(
            sample_weights, sample_cov_matrix, confidence=0.95
        )
        assert isinstance(ivar, np.ndarray)
        assert ivar.shape == sample_weights.shape

    def test_risk_budget_analysis_columns(
        self, sample_weights: np.ndarray, sample_cov_matrix: np.ndarray
    ) -> None:
        df = risk_budget_analysis(
            sample_weights, sample_cov_matrix, confidence=0.95
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"ticker", "current_contribution_pct",
                         "target_contribution_pct", "deviation_pct"}
        assert expected_cols == set(df.columns)
        # Default target is equal risk contribution
        n_assets = len(sample_weights)
        expected_target = 100.0 / n_assets
        np.testing.assert_allclose(
            df["target_contribution_pct"].values,
            np.full(n_assets, expected_target),
            atol=1e-10,
        )

    def test_risk_budget_analysis_custom_target(
        self, sample_weights: np.ndarray, sample_cov_matrix: np.ndarray
    ) -> None:
        custom_target = np.array([50.0, 30.0, 20.0])
        df = risk_budget_analysis(
            sample_weights, sample_cov_matrix,
            target_contributions=custom_target, confidence=0.95,
        )
        np.testing.assert_allclose(
            df["target_contribution_pct"].values, custom_target, atol=1e-10
        )

    def test_risk_attribution_summary_keys(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        tickers = list(sample_returns_df.columns)
        result = risk_attribution_summary(
            tickers, sample_weights, sample_returns_df, confidence=0.95
        )
        assert isinstance(result, dict)
        expected_keys = {"portfolio_var", "attribution_table", "risk_budget",
                         "incremental_var", "concentration"}
        assert expected_keys == set(result.keys())

    def test_risk_attribution_summary_properties(
        self, sample_returns_df: pd.DataFrame, sample_weights: np.ndarray
    ) -> None:
        tickers = list(sample_returns_df.columns)
        result = risk_attribution_summary(
            tickers, sample_weights, sample_returns_df, confidence=0.95
        )
        # Portfolio VaR should be positive
        assert result["portfolio_var"] > 0
        # Concentration (HHI) should be in [1/n, 1]
        n_assets = len(sample_weights)
        assert 1.0 / n_assets - 1e-6 <= result["concentration"] <= 1.0 + 1e-6
        # attribution_table is a DataFrame
        assert isinstance(result["attribution_table"], pd.DataFrame)
        # risk_budget is a DataFrame
        assert isinstance(result["risk_budget"], pd.DataFrame)
        # incremental_var is an ndarray of correct shape
        assert isinstance(result["incremental_var"], np.ndarray)
        assert result["incremental_var"].shape == sample_weights.shape


# =========================================================================
# TestCLI
# =========================================================================

class TestCLI:
    """Tests for command-line argument parsing."""

    def test_cli_parser_custom_portfolio(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--tickers",
            "AAPL,MSFT",
            "--weights",
            "0.6,0.4",
            "--confidence",
            "0.99",
            "--n-sims",
            "5000",
        ])
        assert args.tickers == ["AAPL", "MSFT"]
        np.testing.assert_allclose(args.weights, np.array([0.6, 0.4]))
        assert args.confidence == 0.99
        assert args.n_sims == 5000

    def test_offline_cli_pipeline_outputs(self, tmp_path) -> None:
        output_dir = tmp_path / "offline_run"
        cli_main(
            [
                "--input",
                "data/sample_prices.csv",
                "--tickers",
                "AAPL,MSFT,SPY,TLT,GLD",
                "--weights",
                "0.25,0.25,0.20,0.15,0.15",
                "--n-sims",
                "1000",
                "--backtest-window",
                "60",
                "--output",
                str(output_dir),
            ]
        )

        expected_files = [
            "metrics.json",
            "var_es_table.csv",
            "backtesting_results.csv",
            "stress_results.csv",
            "risk_attribution.csv",
            "data_quality.csv",
            "report.html",
        ]
        for filename in expected_files:
            assert (output_dir / filename).exists()

        assert (output_dir / "figures" / "var_distribution.png").exists()
        assert (output_dir / "figures" / "evt_threshold_diagnostics.png").exists()

    def test_config_from_mapping(self) -> None:
        config = config_from_mapping(
            {
                "portfolio": {"tickers": ["A", "B"], "weights": [0.6, 0.4]},
                "data": {"source": "csv", "input": "data/sample_prices.csv"},
                "risk": {"confidence": 0.99, "n_sims": 1234},
                "backtest": {"window": 120},
                "output": {"dir": "outputs/test"},
            }
        )
        assert config.tickers == ["A", "B"]
        np.testing.assert_allclose(config.weights, np.array([0.6, 0.4]))
        assert config.source == "csv"
        assert config.input_path == "data/sample_prices.csv"
        assert config.confidence == 0.99
        assert config.n_sims == 1234
        assert config.backtest_window == 120


# =========================================================================
# TestData
# =========================================================================

class TestData:
    """Tests for data preparation and cache loading."""

    def test_compute_returns_log_and_simple(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0],
                "B": [50.0, 49.0, 51.0],
            },
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        log_returns = compute_returns(prices, method="log")
        simple_returns = compute_returns(prices, method="simple")

        assert log_returns.shape == (2, 2)
        assert simple_returns.shape == (2, 2)
        np.testing.assert_allclose(simple_returns.iloc[0].values, [0.01, -0.02])

    def test_compute_returns_invalid_method(self) -> None:
        prices = pd.DataFrame({"A": [100.0, 101.0]})
        with pytest.raises(ValueError, match="Invalid method"):
            compute_returns(prices, method="bad")

    def test_fetch_prices_loads_project_cache(self, tmp_path, monkeypatch) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "AAPL_2024-01-01_2024-01-04.csv"
        cached_file.write_text(
            "Date,AAPL\n2024-01-01,100.0\n2024-01-02,101.0\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(data_module, "_CACHE_DIR", cache_dir)
        monkeypatch.setattr(data_module, "_LEGACY_CACHE_DIR", tmp_path / "legacy")

        prices = fetch_prices(["AAPL"], start="2024-01-01", end="2024-01-04")
        assert list(prices.columns) == ["AAPL"]
        assert prices.iloc[-1, 0] == 101.0
