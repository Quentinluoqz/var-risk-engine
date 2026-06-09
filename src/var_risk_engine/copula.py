"""Copula modelling for portfolio risk -- Gaussian vs t-Copula.

Copulas separate the marginal distributions from the joint dependence
structure, allowing more flexible modelling of tail correlation.
The 2008 financial crisis highlighted the danger of relying solely on
Gaussian copulas, which underestimate tail dependence.

This module provides:

* Empirical rank-based transforms to map returns onto the copula scale.
* Maximum-likelihood fitting of Gaussian and Student-t copulas.
* Closed-form tail dependence coefficients for both copula families.
* Copula-based Monte Carlo simulation and Value-at-Risk estimation.
* A side-by-side comparison utility that highlights the tail dependence
  gap between Gaussian and t-Copulas.

Typical usage::

    import pandas as pd
    from var_risk_engine.copula import (
        fit_gaussian_copula,
        fit_t_copula,
        copula_var,
        compare_copulas,
    )

    returns = pd.read_csv("portfolio_returns.csv")
    weights = np.array([0.4, 0.3, 0.3])

    gaussian = fit_gaussian_copula(returns)
    t_params = fit_t_copula(returns)

    var_gauss  = copula_var(returns, weights, gaussian)
    var_t      = copula_var(returns, weights, t_params)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm, t as t_dist


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _multivariate_t_logpdf(
    z: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    nu: float,
) -> np.ndarray:
    """Compute the log-density of the multivariate Student-t distribution.

    Uses the density parameterisation:

    .. math::

        f(z) = \\frac{\\Gamma\\!\\bigl(\\frac{\\nu + d}{2}\\bigr)}
                     {\\Gamma\\!\\bigl(\\frac{\\nu}{2}\\bigr)\\,
                      (\\nu\\pi)^{d/2}\\,|\\Sigma|^{1/2}}
               \\left(1 + \\frac{(z - \\mu)^\\top \\Sigma^{-1}
                                  (z - \\mu)}{\\nu}\\right)^{-(\\nu+d)/2}

    where *d* is the dimensionality and *Sigma* is the positive-definite
    shape matrix.  The covariance of the distribution is
    ``nu / (nu - 2) * Sigma`` for ``nu > 2``.

    Parameters
    ----------
    z : np.ndarray
        Evaluation points, shape ``(n, d)`` or ``(d,)`` for a single point.
    mu : np.ndarray
        Location vector, shape ``(d,)``.
    Sigma : np.ndarray
        Positive-definite shape matrix, shape ``(d, d)``.
    nu : float
        Degrees of freedom (must be > 0).

    Returns
    -------
    np.ndarray
        Log-density at each point, shape ``(n,)`` or scalar.
    """
    z = np.atleast_2d(z)
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    d = z.shape[1]

    diff = z - mu  # (n, d)

    # Log-determinant and inverse via slogdet / solve for numerical stability.
    sign, log_det = np.linalg.slogdet(Sigma)
    if sign <= 0:
        # Fall back: add small ridge to diagonal.
        Sigma = Sigma + np.eye(d) * 1e-8
        sign, log_det = np.linalg.slogdet(Sigma)

    Sigma_inv = np.linalg.inv(Sigma)

    # Squared Mahalanobis distance: (z - mu)^T Sigma^{-1} (z - mu)
    mahal = np.sum(diff @ Sigma_inv * diff, axis=1)  # (n,)

    log_norm = (
        gammaln((nu + d) / 2.0)
        - gammaln(nu / 2.0)
        - (d / 2.0) * np.log(nu * np.pi)
        - 0.5 * log_det
    )

    log_kernel = -((nu + d) / 2.0) * np.log1p(mahal / nu)

    return log_norm + log_kernel


def _t_copula_loglikelihood(
    u: np.ndarray,
    R: np.ndarray,
    nu: float,
) -> float:
    """Compute the log-likelihood of the t-copula density.

    The t-copula density for observation *i* is:

    .. math::

        c(u_i) = \\frac{f_d(q_i;\\, \\nu,\\, R)}
                       {\\prod_{j=1}^{d} f_1(q_{ij};\\, \\nu)}

    where q_ij = t_nu^{-1}(u_ij).

    where :math:`f_d` is the *d*-variate t density with shape matrix *R*,
    and :math:`f_1` is the univariate standard t density.  The total
    log-likelihood is the sum over all observations.

    Parameters
    ----------
    u : np.ndarray
        Pseudo-observations on the unit interval, shape ``(n, d)``.
    R : np.ndarray
        Correlation (shape) matrix, shape ``(d, d)``.
    nu : float
        Degrees of freedom.

    Returns
    -------
    float
        Total log-likelihood.
    """
    u = np.clip(np.asarray(u, dtype=float), 1e-10, 1.0 - 1e-10)
    q = t_dist.ppf(u, df=nu)
    _n, d = q.shape
    mu = np.zeros(d)

    # Joint log-density under multivariate t.
    joint_ll = np.sum(_multivariate_t_logpdf(q, mu, R, nu))

    # Sum of marginal univariate standard-t log-densities.
    marginal_ll = 0.0
    for j in range(d):
        marginal_ll += np.sum(t_dist.logpdf(q[:, j], df=nu))

    return float(joint_ll - marginal_ll)


def _inverse_empirical_cdf(
    u_samples: np.ndarray,
    sorted_returns: np.ndarray,
) -> np.ndarray:
    """Map uniform [0, 1] samples to returns via the inverse empirical CDF.

    For each uniform value *u*, the corresponding return is obtained by
    linear interpolation into the sorted historical return series, which
    is equivalent to ``np.quantile`` with ``method='linear'``.

    Parameters
    ----------
    u_samples : np.ndarray
        Uniform samples in [0, 1], shape ``(n_sims,)``.
    sorted_returns : np.ndarray
        Historically observed returns sorted in ascending order,
        shape ``(n_obs,)``.

    Returns
    -------
    np.ndarray
        Mapped return samples, shape ``(n_sims,)``.
    """
    return np.quantile(sorted_returns, np.clip(u_samples, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Marginal transforms
# ---------------------------------------------------------------------------


def rank_to_uniform(returns: pd.DataFrame) -> pd.DataFrame:
    """Convert each asset's return series to uniform [0, 1] via empirical CDF.

    Applies the rank (empirical CDF) transform column-by-column:

    .. math::

        u_i = \\frac{\\mathrm{rank}(x_i)}{n + 1}

    where ``rank`` is 1-based and ties are averaged.  Dividing by
    ``n + 1`` (rather than ``n``) ensures that all mapped values lie
    strictly inside the open interval (0, 1), which is required for the
    subsequent probit transform.

    This is the standard first step in semiparametric copula estimation:
    it strips away whatever marginal distribution the data may have and
    maps every observation onto the unit interval.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.  Each column is one asset and each
        row is a time-series observation.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same shape, index, and column names as
        *returns*, with all values in the open interval (0, 1).
    """
    n = len(returns)
    # pandas .rank() is 1-based and uses method='average' by default.
    ranked = returns.rank(method="average")
    return ranked / (n + 1)


def uniform_to_normal(u: pd.DataFrame) -> pd.DataFrame:
    """Convert uniform [0, 1] values to standard normal via the probit transform.

    Applies the inverse standard-normal CDF (percent-point function)
    element-wise:

    .. math::

        z_i = \\Phi^{-1}(u_i)

    To avoid infinite values when *u* is extremely close to 0 or 1,
    inputs are clipped to ``[1e-10, 1 - 1e-10]`` before the transform.

    Parameters
    ----------
    u : pd.DataFrame
        DataFrame of uniform [0, 1] values (typically produced by
        :func:`rank_to_uniform`).

    Returns
    -------
    pd.DataFrame
        DataFrame of standard-normal values with the same shape, index,
        and column names as *u*.
    """
    clipped = u.clip(lower=1e-10, upper=1.0 - 1e-10)
    return pd.DataFrame(
        norm.ppf(clipped.values),
        index=u.index,
        columns=u.columns,
    )


# ---------------------------------------------------------------------------
# Copula fitting
# ---------------------------------------------------------------------------


def fit_gaussian_copula(returns: pd.DataFrame) -> dict:
    """Fit a Gaussian copula to the return data.

    The Gaussian copula is fully characterised by its correlation matrix
    *R*.  Estimation proceeds in three steps:

    1. Transform each return series to uniform via :func:`rank_to_uniform`.
    2. Map the uniform values to standard normal via
       :func:`uniform_to_normal`.
    3. Compute the sample correlation matrix of the normal-transformed
       data -- this **is** the Gaussian copula parameter.

    The Gaussian copula has **zero** tail dependence
    (``lambda_upper = lambda_lower = 0``), which is the structural
    weakness that contributed to the 2008 financial crisis.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.  Each column is one asset.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``"corr_matrix"`` -- the copula correlation matrix
          (``np.ndarray``, shape ``(d, d)``).
        * ``"type"`` -- the string ``"gaussian"``.
        * ``"n_obs"`` -- number of observations used in fitting.
    """
    u = rank_to_uniform(returns)
    z = uniform_to_normal(u)

    corr_matrix = np.corrcoef(z.values, rowvar=False)

    return {
        "corr_matrix": corr_matrix,
        "type": "gaussian",
        "n_obs": len(returns),
    }


def fit_t_copula(returns: pd.DataFrame) -> dict:
    """Fit a Student-t copula to the return data.

    Estimation steps:

    1. Transform returns to pseudo-observations on the unit interval.
    2. Estimate the correlation matrix from the normal-transformed data
       as an initial (consistent) estimate of the t-copula shape
       parameter.
    3. Estimate degrees of freedom *nu* by profile maximum likelihood:
       evaluate the t-copula log-likelihood for ``nu`` in
       ``[2, 3, ..., 30]`` and select the value that maximises it.
       Low *nu* implies heavier tails and stronger tail dependence.

    The t-copula log-likelihood for observation *i* is:

    .. math::

        \\ell_i = \\log f_d(q_i;\\, \\nu,\\, R)
                  - \\sum_{j=1}^{d} \\log f_1(q_{ij};\\, \\nu)

    where q_ij = t_nu^{-1}(u_ij), :math:`f_d` is the *d*-variate t
    density with shape matrix *R*, and :math:`f_1` is the standard
    univariate t density.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.  Each column is one asset.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``"corr_matrix"`` -- the copula correlation (shape) matrix
          (``np.ndarray``, shape ``(d, d)``).
        * ``"nu"`` -- estimated degrees of freedom (``float``).
        * ``"type"`` -- the string ``"t"``.
        * ``"n_obs"`` -- number of observations used in fitting.
        * ``"loglikelihood"`` -- maximised log-likelihood (``float``).
    """
    u = rank_to_uniform(returns)
    z = uniform_to_normal(u)

    u_values = u.values  # (n, d)
    corr_matrix = np.corrcoef(z.values, rowvar=False)

    # Profile likelihood over integer degrees of freedom.
    nu_candidates = np.arange(2, 31, dtype=float)
    best_nu = nu_candidates[0]
    best_ll = -np.inf

    for nu in nu_candidates:
        ll = _t_copula_loglikelihood(u_values, corr_matrix, nu)
        if ll > best_ll:
            best_ll = ll
            best_nu = nu

    return {
        "corr_matrix": corr_matrix,
        "nu": float(best_nu),
        "type": "t",
        "n_obs": len(returns),
        "loglikelihood": best_ll,
    }


# ---------------------------------------------------------------------------
# Tail dependence coefficients
# ---------------------------------------------------------------------------


def tail_dependence_gaussian(corr: float) -> float:
    """Upper tail dependence coefficient for a bivariate Gaussian copula.

    The Gaussian copula has **zero** tail dependence for any finite
    correlation ``rho`` in ``(-1, 1)``.  Formally:

    .. math::

        \\lambda_U = \\lim_{u \\to 1^-}
            P\\!\\bigl(X_1 > F_1^{-1}(u) \\,\\big|\\,
                       X_2 > F_2^{-1}(u)\\bigr) = 0

    This is the fundamental limitation of the Gaussian copula: it cannot
    capture the empirical observation that extreme co-movements are more
    likely than a Gaussian model predicts.

    Parameters
    ----------
    corr : float
        Pairwise correlation ``rho`` in ``(-1, 1)``.  Ignored for the
        Gaussian copula (always returns zero) but accepted for API
        symmetry with :func:`tail_dependence_t`.

    Returns
    -------
    float
        Always ``0.0``.
    """
    return 0.0


def tail_dependence_t(corr: float, nu: float) -> float:
    """Upper (and lower) tail dependence coefficient for a bivariate t-copula.

    For a bivariate Student-t copula with correlation ``rho`` and ``nu``
    degrees of freedom, the upper and lower tail dependence coefficients
    are equal by symmetry and given by:

    .. math::

        \\lambda_U = \\lambda_L
            = 2\\, t_{\\nu+1}\\!\\left(
                -\\sqrt{\\frac{(\\nu + 1)(1 - \\rho)}{1 + \\rho}}
              \\right)

    where :math:`t_{\\nu+1}` is the CDF of the standard Student-t
    distribution with :math:`\\nu + 1` degrees of freedom.

    Lower ``nu`` (heavier tails) and higher ``rho`` both increase tail
    dependence.  As ``nu -> infinity`` the t-copula converges to the
    Gaussian copula and ``lambda -> 0``.

    Parameters
    ----------
    corr : float
        Pairwise correlation ``rho`` in ``(-1, 1)``.
    nu : float
        Degrees of freedom of the t-copula (must be > 0).

    Returns
    -------
    float
        Tail dependence coefficient in ``[0, 1]``.
    """
    rho = corr
    # Guard against rho = -1 (division by zero).
    if rho <= -1.0 + 1e-12:
        return 0.0
    if rho >= 1.0 - 1e-12:
        return 1.0

    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * t_dist.cdf(arg, df=nu + 1.0))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_copula(
    copula_params: dict,
    n_sims: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate uniform [0, 1] samples from a fitted copula.

    **Gaussian copula**: draw from a multivariate normal with the copula
    correlation matrix, then apply the standard-normal CDF
    :math:`\\Phi` to obtain uniform marginals.

    **t-copula**: draw from a multivariate Student-t distribution with
    the copula correlation matrix and estimated degrees of freedom,
    then apply the univariate t CDF to each margin.  Multivariate t
    samples are generated via the stochastic representation:

    .. math::

        X = \\sqrt{\\frac{\\nu}{S}}\\; L\\, Z

    where :math:`Z \\sim N(0, I_d)`, :math:`S \\sim \\chi^2(\\nu)`, and
    :math:`L` is the Cholesky factor of the correlation matrix *R*.

    Parameters
    ----------
    copula_params : dict
        Dictionary returned by :func:`fit_gaussian_copula` or
        :func:`fit_t_copula`.  Must contain at least ``"type"`` and
        ``"corr_matrix"``.  For the t-copula, ``"nu"`` is also required.
    n_sims : int, optional
        Number of samples to generate.  Default 10 000.
    seed : int, optional
        Random seed for reproducibility.  Default 42.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_sims, n_assets)`` with values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the copula type is not ``"gaussian"`` or ``"t"``, or if the
        correlation matrix is not positive-definite.
    """
    copula_type: str = copula_params["type"]
    R: np.ndarray = np.asarray(copula_params["corr_matrix"], dtype=float)
    d = R.shape[0]

    rng = np.random.default_rng(seed)

    # Cholesky factor of the correlation matrix.
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Correlation matrix is not positive-definite; "
            "Cholesky decomposition failed."
        ) from exc

    # Independent standard normals: (n_sims, d).
    Z = rng.standard_normal(size=(n_sims, d))
    # Correlated normals: X = Z @ L^T.
    X = Z @ L.T  # (n_sims, d)

    if copula_type == "gaussian":
        # Apply standard-normal CDF to get uniform margins.
        U = norm.cdf(X)

    elif copula_type == "t":
        nu: float = copula_params["nu"]
        # Chi-squared mixing variable: S ~ chi^2(nu).
        S = rng.chisquare(df=nu, size=(n_sims, 1))  # (n_sims, 1)
        # Scale to obtain multivariate-t samples.
        X_t = X * np.sqrt(nu / S)  # (n_sims, d)
        # Apply univariate standard-t CDF to each margin.
        U = t_dist.cdf(X_t, df=nu)

    else:
        raise ValueError(
            f"Unknown copula type '{copula_type}'. "
            "Expected 'gaussian' or 't'."
        )

    return U


# ---------------------------------------------------------------------------
# Copula-based VaR
# ---------------------------------------------------------------------------


def copula_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    copula_params: dict,
    confidence: float = 0.95,
    n_sims: int = 20_000,
    seed: int = 42,
) -> float:
    """Compute portfolio VaR using the copula approach.

    The procedure preserves the empirical marginal distributions while
    imposing the copula's dependence structure on simulated scenarios:

    1. Estimate each asset's marginal distribution empirically (store
       sorted historical returns).
    2. Simulate uniform ``[0, 1]`` samples from the fitted copula via
       :func:`simulate_copula`.
    3. Transform the uniform samples back to the return scale using the
       inverse empirical CDF (quantile mapping into the sorted
       historical returns).
    4. Compute portfolio returns as the weighted sum of simulated asset
       returns.
    5. Return the ``(1 - confidence)`` quantile of the loss
       distribution as the VaR estimate.

    This semiparametric approach combines the flexibility of copulas
    (for dependence) with the robustness of empirical marginals (no
    parametric distributional assumption on individual assets).

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.  Each column is one asset.
    weights : np.ndarray
        Portfolio weights, shape ``(n_assets,)``.  Should sum to 1.
    copula_params : dict
        Fitted copula parameters from :func:`fit_gaussian_copula` or
        :func:`fit_t_copula`.
    confidence : float, optional
        One-sided confidence level (e.g. 0.95 for 95 % VaR).  Must be
        in ``(0, 1)``.  Default 0.95.
    n_sims : int, optional
        Number of Monte Carlo simulations.  Default 20 000.
    seed : int, optional
        Random seed for reproducibility.  Default 42.

    Returns
    -------
    float
        A positive float representing the estimated VaR loss at the
        given confidence level.

    Raises
    ------
    ValueError
        If *confidence* is not in ``(0, 1)`` or the number of weights
        does not match the number of asset columns.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    weights = np.asarray(weights, dtype=float).ravel()
    n_assets = returns.shape[1]
    if weights.shape[0] != n_assets:
        raise ValueError(
            f"Number of weights ({weights.shape[0]}) does not match "
            f"number of asset columns ({n_assets})."
        )

    # Step 1: Store sorted historical returns for inverse CDF mapping.
    sorted_returns = {
        col: np.sort(returns[col].values) for col in returns.columns
    }

    # Step 2: Simulate uniform samples from the copula.
    U = simulate_copula(copula_params, n_sims=n_sims, seed=seed)
    # U shape: (n_sims, n_assets)

    # Step 3: Transform uniform samples back to return scale.
    simulated_returns = np.empty_like(U)
    for j, col in enumerate(returns.columns):
        simulated_returns[:, j] = _inverse_empirical_cdf(
            U[:, j], sorted_returns[col]
        )

    # Step 4: Compute portfolio returns.
    portfolio_returns = simulated_returns @ weights  # (n_sims,)

    # Step 5: VaR is the (1 - confidence) quantile of losses.
    var_value = float(-np.quantile(portfolio_returns, 1.0 - confidence))
    return var_value


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_copulas(returns: pd.DataFrame) -> pd.DataFrame:
    """Compare Gaussian and t-Copula tail dependence for every asset pair.

    Fits both a Gaussian copula and a Student-t copula to the full set
    of returns, then computes the theoretical tail dependence coefficient
    for each bivariate pair under both models.  The resulting table
    clearly shows the structural difference: the Gaussian copula always
    reports zero tail dependence, while the t-Copula captures positive
    tail dependence (especially pronounced when the estimated ``nu`` is
    low).

    This is the core pedagogical insight of copula modelling: **the
    Gaussian copula's zero tail dependence was a key factor in the
    systematic underestimation of systemic risk before the 2008
    financial crisis**.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.  Must have at least two columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per unique asset pair and the following
        columns:

        * ``"asset_i"`` -- name of the first asset.
        * ``"asset_j"`` -- name of the second asset.
        * ``"correlation"`` -- copula correlation (from the t-copula
          fit, which is identical to the Gaussian estimate).
        * ``"gaussian_tail_dep"`` -- tail dependence under the Gaussian
          copula (always 0.0).
        * ``"t_tail_dep"`` -- tail dependence under the t-Copula.
        * ``"t_nu"`` -- estimated degrees of freedom for the t-Copula.
    """
    gauss_params = fit_gaussian_copula(returns)
    t_params = fit_t_copula(returns)

    gauss_corr: np.ndarray = gauss_params["corr_matrix"]
    t_corr: np.ndarray = t_params["corr_matrix"]
    nu: float = t_params["nu"]

    columns = list(returns.columns)
    n_assets = len(columns)

    rows: list[dict] = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            rho_gauss = float(gauss_corr[i, j])
            rho_t = float(t_corr[i, j])

            rows.append(
                {
                    "asset_i": columns[i],
                    "asset_j": columns[j],
                    "correlation": rho_t,
                    "gaussian_tail_dep": tail_dependence_gaussian(rho_gauss),
                    "t_tail_dep": tail_dependence_t(rho_t, nu),
                    "t_nu": nu,
                }
            )

    return pd.DataFrame(rows)
