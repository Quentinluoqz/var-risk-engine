"""Covariance matrix estimation and Cholesky decomposition for portfolio risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Compute the sample covariance matrix from a return DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns, where each column is an asset and each
        row is a time-series observation.

    Returns
    -------
    np.ndarray
        The sample covariance matrix of shape (n_assets, n_assets).
    """
    return returns.cov().values


def shrinkage_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Compute the Ledoit-Wolf shrinkage covariance estimator.

    Uses ``sklearn.covariance.LedoitWolf`` to produce a well-conditioned
    covariance estimate that shrinks the sample covariance toward a scaled
    identity matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.

    Returns
    -------
    np.ndarray
        The shrinkage covariance matrix of shape (n_assets, n_assets).
    """
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_


def ewma_covariance(returns: pd.DataFrame, lambda_: float = 0.94) -> np.ndarray:
    """Compute the EWMA (RiskMetrics) covariance matrix.

    Iteratively updates the covariance estimate as:

        Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_t @ r_t^T

    and returns the final estimate after processing all observations.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.
    lambda_ : float, optional
        Decay factor (default 0.94, the RiskMetrics standard).

    Returns
    -------
    np.ndarray
        The EWMA covariance matrix of shape (n_assets, n_assets).
    """
    r = returns.values
    n_obs, n_assets = r.shape
    sigma = np.zeros((n_assets, n_assets))

    for t in range(n_obs):
        rt = r[t].reshape(-1, 1)
        sigma = lambda_ * sigma + (1.0 - lambda_) * (rt @ rt.T)

    return sigma


def cholesky_factor(cov_matrix: np.ndarray) -> np.ndarray:
    """Compute the Cholesky factor L such that L @ L^T = cov_matrix.

    If the matrix is not positive-definite, a small jitter is added to the
    diagonal before retrying the decomposition.

    Parameters
    ----------
    cov_matrix : np.ndarray
        A symmetric covariance matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        Lower-triangular Cholesky factor of shape (n, n).

    Raises
    ------
    np.linalg.LinAlgError
        If the matrix is still not positive-definite after adding jitter.
    """
    jitter_values = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]

    for jitter in jitter_values:
        try:
            adjusted = cov_matrix.copy()
            if jitter > 0:
                adjusted += jitter * np.eye(cov_matrix.shape[0])
            return np.linalg.cholesky(adjusted)
        except np.linalg.LinAlgError:
            continue

    raise np.linalg.LinAlgError(
        "Matrix is not positive-definite even after adding jitter to the diagonal."
    )


def correlation_from_covariance(cov_matrix: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Computes D^{-1/2} @ Sigma @ D^{-1/2}, where D is the diagonal matrix
    of variances.

    Parameters
    ----------
    cov_matrix : np.ndarray
        A covariance matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        The corresponding correlation matrix of shape (n, n), with ones on
        the diagonal and values in [-1, 1] off-diagonal.
    """
    diag = np.diag(cov_matrix)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(diag))
    corr = d_inv_sqrt @ cov_matrix @ d_inv_sqrt
    # Clip to handle floating-point noise
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr


def compare_covariance_methods(returns: pd.DataFrame) -> dict:
    """Compare multiple covariance estimation methods side by side.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.

    Returns
    -------
    dict
        Dictionary with keys ``"sample"``, ``"shrinkage"``, and ``"ewma"``,
        each containing the respective covariance matrix as a ``np.ndarray``.
    """
    return {
        "sample": sample_covariance(returns),
        "shrinkage": shrinkage_covariance(returns),
        "ewma": ewma_covariance(returns),
    }
