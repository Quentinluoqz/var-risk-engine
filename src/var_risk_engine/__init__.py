"""VaR Risk Engine — Portfolio risk estimation via Historical, Parametric & Monte Carlo methods.

Modules
-------
portfolio          Portfolio dataclass with weight validation.
data               Yahoo Finance data fetching with local CSV cache.
var_historical     Historical VaR (quantile-based, rolling window).
var_parametric     Parametric VaR (mean-variance & covariance matrix).
var_montecarlo     Monte Carlo VaR (GBM + Cholesky decomposition).
expected_shortfall Expected Shortfall / CVaR (historical & parametric).
covariance         Covariance estimation (Sample, Ledoit-Wolf, EWMA) & Cholesky.
backtesting        Kupiec POF, Christoffersen, Basel traffic-light tests.
volatility         GARCH(1,1), EGARCH, GJR-GARCH conditional volatility.
stress_testing     Historical & hypothetical stress scenarios, reverse stress test.
evt                Extreme Value Theory — Peaks-over-Threshold with GPD.
copula             Gaussian vs t-Copula — tail dependence & semiparametric VaR.
risk_attribution   Component VaR, Marginal VaR, Euler decomposition.
"""

__version__ = "0.3.0"
