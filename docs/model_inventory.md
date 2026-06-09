# Model Inventory

This inventory frames the project like a lightweight model-risk register. It is written for interview discussion, not for formal model approval.

## Historical VaR

| Field | Description |
|---|---|
| Purpose | Empirical portfolio loss quantile benchmark |
| Inputs | Daily portfolio returns, confidence level, horizon |
| Assumptions | Recent history is informative for near-future loss distribution |
| Strengths | Transparent, non-parametric, easy to backtest |
| Weaknesses | Sample dependent, poor extrapolation beyond observed tails |
| Validation tests | Rolling exception count, Kupiec POF, Christoffersen conditional coverage |
| Failure modes | Regime shift, stale history, too few stress observations |
| When not to use | Estimating very high confidence tail risk from short samples |

## Parametric VaR / ES

| Field | Description |
|---|---|
| Purpose | Analytical normal benchmark for portfolio risk |
| Inputs | Portfolio returns or weights plus covariance matrix |
| Assumptions | Linear portfolio, normal returns, stable covariance |
| Strengths | Fast, explainable, useful baseline |
| Weaknesses | Underestimates fat tails and skew; sensitive to covariance quality |
| Validation tests | Compare exceptions vs expected rate, normal Q-Q diagnostic |
| Failure modes | Crisis tails, nonlinear exposures, unstable correlations |
| When not to use | Options-heavy or strongly nonlinear portfolios without revaluation |

## Monte Carlo GBM

| Field | Description |
|---|---|
| Purpose | Scenario-based VaR/ES from simulated correlated terminal prices |
| Inputs | Spot prices, expected returns, volatilities, correlation matrix, weights |
| Assumptions | GBM dynamics, lognormal prices, fixed correlation and volatility |
| Strengths | Flexible scenario generation, easy to extend to multi-step horizons |
| Weaknesses | Simplified dynamics; no jumps, stochastic volatility, or liquidity effects |
| Validation tests | Seeded regression tests, simulation convergence, comparison to parametric VaR |
| Failure modes | Correlation breakdown, volatility clustering, tail underestimation |
| When not to use | As a standalone crisis-risk model without stress and tail overlays |

## GARCH-Adjusted Monte Carlo

| Field | Description |
|---|---|
| Purpose | Replace static volatility with one-day-ahead conditional volatility forecasts |
| Inputs | Asset returns, GARCH model fits, correlation matrix, Monte Carlo settings |
| Assumptions | Conditional volatility forecasts are informative; correlation remains fixed |
| Strengths | Captures volatility clustering better than static sigma |
| Weaknesses | Fixed correlation; GARCH convergence can be unstable on poor data |
| Validation tests | AIC/BIC comparison, residual diagnostics, forecast sensitivity |
| Failure modes | Non-convergence, structural breaks, heavy-tail misspecification |
| When not to use | Very short histories or assets with sparse/stale prices |

## EVT POT / GPD

| Field | Description |
|---|---|
| Purpose | Estimate extreme tail VaR/ES beyond empirical quantiles |
| Inputs | Loss series, threshold, confidence levels |
| Assumptions | Excesses above threshold follow a Generalised Pareto Distribution |
| Strengths | Better high-confidence tail extrapolation than raw empirical quantiles |
| Weaknesses | Sensitive threshold choice; parameter uncertainty can be large |
| Validation tests | GPD Q-Q plot, mean-excess diagnostic, threshold stability |
| Failure modes | Too few exceedances, unstable shape parameter, regime changes |
| When not to use | Short samples with almost no tail observations |

## Gaussian Copula

| Field | Description |
|---|---|
| Purpose | Dependence benchmark with empirical marginals |
| Inputs | Asset returns, empirical CDF transforms, correlation matrix |
| Assumptions | Dependence fully captured by Gaussian correlation |
| Strengths | Simple and interpretable |
| Weaknesses | Zero tail dependence for finite correlations |
| Validation tests | Pairwise tail dependence comparison against Student-t copula |
| Failure modes | Joint crashes underestimated |
| When not to use | Portfolios where systemic tail co-movement is central |

## Student-t Copula

| Field | Description |
|---|---|
| Purpose | Model symmetric upper/lower tail dependence |
| Inputs | Pseudo-observations, t-copula degrees of freedom, correlation matrix |
| Assumptions | Symmetric t-copula dependence structure |
| Strengths | Captures joint tail dependence missing from Gaussian copula |
| Weaknesses | Symmetric tails; degrees-of-freedom estimate can be unstable |
| Validation tests | Tail dependence table, degrees-of-freedom sensitivity, VaR comparison |
| Failure modes | Asymmetric crash dependence, sparse tail data |
| When not to use | When dependence is strongly asymmetric or regime-switching |

## Component / Marginal VaR

| Field | Description |
|---|---|
| Purpose | Attribute total portfolio VaR to assets |
| Inputs | Weights, covariance matrix, confidence level |
| Assumptions | Delta-normal VaR is homogeneous in portfolio weights |
| Strengths | Additive Euler decomposition; useful for risk budgeting |
| Weaknesses | Parametric and linear; not a full revaluation attribution |
| Validation tests | Sum of Component VaR equals total VaR |
| Failure modes | Nonlinear exposures, unstable covariance, negative diversification effects |
| When not to use | Explaining option Greeks or path-dependent portfolios without revaluation |
