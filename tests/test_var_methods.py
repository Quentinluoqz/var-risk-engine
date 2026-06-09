"""Tests for core VaR risk engine modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_risk_engine.portfolio import Portfolio
from var_risk_engine.var_historical import historical_var, historical_var_series
from var_risk_engine.var_parametric import parametric_var, parametric_var_from_cov
from var_risk_engine.var_montecarlo import simulate_gbm_paths, montecarlo_var, montecarlo_var_with_details
from var_risk_engine.expected_shortfall import expected_shortfall, es_parametric, compare_var_es
from var_risk_engine.covariance import (
    sample_covariance,
    shrinkage_covariance,
    ewma_covariance,
    cholesky_factor,
    correlation_from_covariance,
)
from var_risk_engine.backtesting import (
    count_violations,
    kupiec_pof_test,
    basel_traffic_light,
    rolling_backtest,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_returns(rng: np.random.Generator) -> np.ndarray:
    """1000 daily returns ~ N(0.0003, 0.015)."""
    return rng.normal(0.0003, 0.015, size=1000)


@pytest.fixture
def sample_returns_df(rng: np.random.Generator) -> pd.DataFrame:
    """DataFrame with 3 assets, 500 days."""
    data = rng.normal(0.0003, 0.01, size=(500, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])


# =========================================================================
# Portfolio
# =========================================================================

class TestPortfolio:
    def test_valid_creation(self) -> None:
        p = Portfolio("test", ["AAPL", "MSFT"], np.array([0.5, 0.5]))
        assert p.n_assets == 2
        assert p.benchmark is None

    def test_weight_validation(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            Portfolio("bad", ["A", "B"], np.array([0.3, 0.3]))

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            Portfolio("bad", ["A"], np.array([0.5, 0.5]))

    def test_weights_not_ndarray(self) -> None:
        with pytest.raises(TypeError, match="numpy"):
            Portfolio("bad", ["A"], [1.0])  # type: ignore[arg-type]


# =========================================================================
# Historical VaR
# =========================================================================

class TestHistoricalVaR:
    def test_returns_positive(self, sample_returns: np.ndarray) -> None:
        var = historical_var(sample_returns, confidence=0.95)
        assert var > 0

    def test_higher_confidence_higher_var(self, sample_returns: np.ndarray) -> None:
        var_95 = historical_var(sample_returns, confidence=0.95)
        var_99 = historical_var(sample_returns, confidence=0.99)
        assert var_99 > var_95

    def test_holding_period_scaling(self, sample_returns: np.ndarray) -> None:
        var_1d = historical_var(sample_returns, confidence=0.95, holding_period=1)
        var_10d = historical_var(sample_returns, confidence=0.95, holding_period=10)
        assert pytest.approx(var_10d, rel=0.01) == var_1d * np.sqrt(10)

    def test_invalid_confidence(self, sample_returns: np.ndarray) -> None:
        with pytest.raises(ValueError):
            historical_var(sample_returns, confidence=1.5)

    def test_rolling_series_shape(self, sample_returns: np.ndarray) -> None:
        series = historical_var_series(sample_returns, confidence=0.95, window=250)
        assert len(series) == len(sample_returns) - 250 + 1


# =========================================================================
# Parametric VaR
# =========================================================================

class TestParametricVaR:
    def test_returns_positive(self, sample_returns: np.ndarray) -> None:
        var = parametric_var(sample_returns, confidence=0.95)
        assert var > 0

    def test_from_cov_matches(self, sample_returns: np.ndarray) -> None:
        # Single-asset case: parametric_var and from_cov should agree
        cov = np.array([[np.var(sample_returns, ddof=1)]])
        weights = np.array([1.0])
        var_direct = parametric_var(sample_returns, confidence=0.95)
        var_cov = parametric_var_from_cov(weights, cov, confidence=0.95)
        assert pytest.approx(var_cov, rel=0.05) == var_direct


# =========================================================================
# Monte Carlo VaR
# =========================================================================

class TestMonteCarloVaR:
    def test_gbm_paths_shape(self) -> None:
        S0 = np.array([100.0, 50.0])
        mu = np.array([0.05, 0.03])
        sigma = np.array([0.2, 0.15])
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])

        paths = simulate_gbm_paths(S0, mu, sigma, corr, T=1, n_sims=5000, seed=42)
        assert paths.shape == (5000, 2)

    def test_mc_var_positive(self) -> None:
        weights = np.array([0.6, 0.4])
        S0 = np.array([100.0, 50.0])
        mu = np.array([0.05, 0.03])
        sigma = np.array([0.2, 0.15])
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])

        var = montecarlo_var(weights, S0, mu, sigma, corr, confidence=0.95, n_sims=5000)
        assert var > 0

    def test_mc_details_keys(self) -> None:
        weights = np.array([1.0])
        S0 = np.array([100.0])
        mu = np.array([0.05])
        sigma = np.array([0.2])
        corr = np.array([[1.0]])

        details = montecarlo_var_with_details(weights, S0, mu, sigma, corr, n_sims=5000)
        assert set(details.keys()) == {"var", "es", "simulated_returns", "portfolio_values"}
        assert details["es"] >= details["var"]  # ES >= VaR always


# =========================================================================
# Expected Shortfall
# =========================================================================

class TestExpectedShortfall:
    def test_es_greater_than_var(self, sample_returns: np.ndarray) -> None:
        var = historical_var(sample_returns, confidence=0.95)
        es = expected_shortfall(sample_returns, confidence=0.95)
        assert es >= var  # ES is always >= VaR for same confidence

    def test_parametric_es_positive(self, sample_returns: np.ndarray) -> None:
        es = es_parametric(sample_returns, confidence=0.95)
        assert es > 0

    def test_compare_table(self, sample_returns: np.ndarray) -> None:
        df = compare_var_es(sample_returns)
        assert set(df.columns) == {"confidence", "var_historical", "es_historical", "var_parametric", "es_parametric"}
        assert len(df) == 4  # default confidence levels
        # ES >= VaR for each row
        for _, row in df.iterrows():
            assert row["es_historical"] >= row["var_historical"]


# =========================================================================
# Covariance
# =========================================================================

class TestCovariance:
    def test_sample_cov_symmetric(self, sample_returns_df: pd.DataFrame) -> None:
        cov = sample_covariance(sample_returns_df)
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)

    def test_shrinkage_positive_definite(self, sample_returns_df: pd.DataFrame) -> None:
        cov = shrinkage_covariance(sample_returns_df)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_ewma_shape(self, sample_returns_df: pd.DataFrame) -> None:
        cov = ewma_covariance(sample_returns_df)
        assert cov.shape == (3, 3)

    def test_cholesky_reconstruction(self, sample_returns_df: pd.DataFrame) -> None:
        cov = sample_covariance(sample_returns_df)
        L = cholesky_factor(cov)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, cov, atol=1e-10)

    def test_correlation_diagonal_ones(self, sample_returns_df: pd.DataFrame) -> None:
        cov = sample_covariance(sample_returns_df)
        corr = correlation_from_covariance(cov)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)


# =========================================================================
# Backtesting
# =========================================================================

class TestBacktesting:
    def test_count_violations(self) -> None:
        returns = np.array([-0.05, -0.01, 0.02, -0.031, 0.01])
        var_forecasts = np.array([0.03, 0.03, 0.03, 0.03, 0.03])
        viols = count_violations(returns, var_forecasts)
        assert viols.sum() == 2  # -0.05 and -0.031 are violations (< -0.03)

    def test_kupiec_no_violations(self) -> None:
        # All returns positive -> no violations
        returns = np.abs(np.random.default_rng(42).normal(0.01, 0.005, 250))
        var_f = np.full(250, 0.03)
        result = kupiec_pof_test(returns, var_f, confidence=0.95)
        assert result["n_violations"] == 0

    def test_basel_green(self) -> None:
        result = basel_traffic_light(3, 250)
        assert result["zone"] == "green"
        assert result["scaling_factor"] == 1.0

    def test_basel_yellow(self) -> None:
        result = basel_traffic_light(7, 250)
        assert result["zone"] == "yellow"
        assert result["scaling_factor"] > 1.0

    def test_basel_red(self) -> None:
        result = basel_traffic_light(12, 250)
        assert result["zone"] == "red"
        assert result["scaling_factor"] >= 1.3

    def test_rolling_backtest(self, sample_returns: np.ndarray) -> None:
        result = rolling_backtest(
            sample_returns,
            var_func=historical_var,
            confidence=0.95,
            window=250,
        )
        assert "var_forecasts" in result
        assert len(result["var_forecasts"]) == len(sample_returns) - 250
