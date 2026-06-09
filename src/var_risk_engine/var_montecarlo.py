"""Monte Carlo Value-at-Risk via Geometric Brownian Motion simulation."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_matrix: np.ndarray,
    T: int = 1,
    n_sims: int = 10_000,
    dt: float = 1 / 252,
    seed: int = 42,
) -> np.ndarray:
    """Simulate correlated multi-asset price paths using Geometric Brownian Motion.

    For each asset *i* the terminal price after *T* trading days is computed
    with the exact GBM solution:

        S_{t+dt} = S_t * exp((mu_i - 0.5 * sigma_i^2) * dt
                               + sigma_i * sqrt(dt) * Z_i)

    where the vector **Z** is drawn from a multivariate standard normal with
    the correlation structure induced by *corr_matrix* via Cholesky
    decomposition.

    Args:
        S0: Initial asset prices.  Shape ``(n_assets,)``.
        mu: Annualised expected (drift) returns.  Shape ``(n_assets,)``.
        sigma: Annualised volatilities.  Shape ``(n_assets,)``.
        corr_matrix: Correlation matrix (symmetric, positive-definite).
            Shape ``(n_assets, n_assets)``.
        T: Number of trading days to simulate.  Default 1.
        n_sims: Number of Monte Carlo paths.  Default 10 000.
        dt: Time step expressed as a fraction of a year.  Default 1/252
            (one trading day).
        seed: Random seed for reproducibility.

    Returns:
        ``np.ndarray`` of shape ``(n_sims, n_assets)`` containing the
        simulated terminal prices for each asset.

    Raises:
        ValueError: If the input array dimensions are inconsistent or
            *corr_matrix* is not positive-definite.
    """
    S0 = np.asarray(S0, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    corr_matrix = np.asarray(corr_matrix, dtype=float)

    n_assets = S0.shape[0]
    if mu.shape[0] != n_assets or sigma.shape[0] != n_assets:
        raise ValueError("S0, mu and sigma must all have the same length.")
    if corr_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"corr_matrix shape {corr_matrix.shape} is inconsistent with "
            f"n_assets={n_assets}"
        )

    # Cholesky factor of the correlation matrix (lower-triangular).
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "corr_matrix is not positive-definite; Cholesky decomposition "
            "failed."
        ) from exc

    rng = np.random.default_rng(seed)

    # Independent standard normals: (n_sims, n_assets).
    Z_indep = rng.standard_normal(size=(n_sims, n_assets))

    # Correlated standard normals: Z_corr = Z_indep @ L^T.
    Z_corr = Z_indep @ L.T  # shape (n_sims, n_assets)

    # Broadcast parameters to (n_sims, n_assets).
    # When T > 1 we compound step-by-step for path accuracy.
    S = np.broadcast_to(S0, (n_sims, n_assets)).copy()

    for _ in range(T):
        # Fresh correlated draws for each step (reuse the same L).
        if _ == 0:
            Z_step = Z_corr
        else:
            Z_step = rng.standard_normal(size=(n_sims, n_assets)) @ L.T

        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z_step
        S = S * np.exp(drift + diffusion)

    return S


def montecarlo_var(
    weights: np.ndarray,
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_matrix: np.ndarray,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    T: int = 1,
    seed: int = 42,
) -> float:
    """Estimate portfolio VaR using Monte Carlo GBM simulation.

    Simulates *n_sims* correlated asset price paths over *T* trading days,
    computes the portfolio return distribution, and returns the (1 -
    confidence) quantile of losses.

    Args:
        weights: Portfolio weights summing to 1.  Shape ``(n_assets,)``.
        S0: Initial asset prices.  Shape ``(n_assets,)``.
        mu: Annualised expected returns.  Shape ``(n_assets,)``.
        sigma: Annualised volatilities.  Shape ``(n_assets,)``.
        corr_matrix: Correlation matrix.  Shape ``(n_assets, n_assets)``.
        confidence: One-sided confidence level (e.g. 0.95).  Must be in
            (0, 1).
        n_sims: Number of Monte Carlo paths.
        T: Holding period in trading days.
        seed: Random seed for reproducibility.

    Returns:
        A positive float representing the estimated VaR loss.

    Raises:
        ValueError: If *confidence* is not in (0, 1) or array shapes are
            inconsistent.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    weights = np.asarray(weights, dtype=float).ravel()
    S0 = np.asarray(S0, dtype=float).ravel()

    terminal_prices = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        T=T,
        n_sims=n_sims,
        seed=seed,
    )  # (n_sims, n_assets)

    # Initial and terminal portfolio values.
    initial_value = np.dot(weights, S0)
    terminal_values = terminal_prices @ weights  # (n_sims,)

    # Portfolio returns relative to initial value.
    portfolio_returns = (terminal_values - initial_value) / initial_value

    var = -np.quantile(portfolio_returns, 1.0 - confidence)
    return float(var)


def montecarlo_var_with_details(
    weights: np.ndarray,
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_matrix: np.ndarray,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    T: int = 1,
    seed: int = 42,
) -> dict:
    """Monte Carlo VaR with additional risk metrics and simulation outputs.

    Performs the same simulation as :func:`montecarlo_var` but also computes
    Expected Shortfall (CVaR) and returns the raw simulation results for
    further analysis or plotting.

    Args:
        weights: Portfolio weights summing to 1.  Shape ``(n_assets,)``.
        S0: Initial asset prices.  Shape ``(n_assets,)``.
        mu: Annualised expected returns.  Shape ``(n_assets,)``.
        sigma: Annualised volatilities.  Shape ``(n_assets,)``.
        corr_matrix: Correlation matrix.  Shape ``(n_assets, n_assets)``.
        confidence: One-sided confidence level.  Must be in (0, 1).
        n_sims: Number of Monte Carlo paths.
        T: Holding period in trading days.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary with the following keys:

        * ``"var"`` -- VaR estimate (positive float).
        * ``"es"`` -- Expected Shortfall, i.e. the conditional mean of
          losses that exceed VaR (positive float).
        * ``"simulated_returns"`` -- 1-D ``np.ndarray`` of simulated
          portfolio returns, shape ``(n_sims,)``.
        * ``"portfolio_values"`` -- 1-D ``np.ndarray`` of simulated
          terminal portfolio values, shape ``(n_sims,)``.

    Raises:
        ValueError: If *confidence* is not in (0, 1) or array shapes are
            inconsistent.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    weights = np.asarray(weights, dtype=float).ravel()
    S0 = np.asarray(S0, dtype=float).ravel()

    terminal_prices = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        T=T,
        n_sims=n_sims,
        seed=seed,
    )  # (n_sims, n_assets)

    initial_value = np.dot(weights, S0)
    terminal_values = terminal_prices @ weights  # (n_sims,)
    portfolio_returns = (terminal_values - initial_value) / initial_value

    var_threshold = -np.quantile(portfolio_returns, 1.0 - confidence)

    # Expected Shortfall: mean of losses that exceed VaR.
    losses = -portfolio_returns
    tail_losses = losses[losses >= var_threshold]

    if tail_losses.size > 0:
        es = float(np.mean(tail_losses))
    else:
        # Degenerate case: no losses beyond VaR (extremely unlikely with
        # reasonable n_sims).
        es = float(var_threshold)

    return {
        "var": float(var_threshold),
        "es": es,
        "simulated_returns": portfolio_returns,
        "portfolio_values": terminal_values,
    }
