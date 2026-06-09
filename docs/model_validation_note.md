# Model Validation Note

## Model Purpose

This project estimates daily portfolio market risk for a liquid multi-asset portfolio. It compares Historical, Parametric, Monte Carlo, GARCH-adjusted Monte Carlo, EVT, and copula-based risk estimates, then validates VaR forecasts with statistical backtests.

The intended use is educational and analytical: model comparison, assumption testing, and model-risk discussion for market-risk or risk-analytics interviews. It is not a production capital engine.

## Portfolio And Data

Default portfolio:

| Asset | Weight | Risk role |
|---|---:|---|
| AAPL | 25% | Single-name equity risk |
| MSFT | 25% | Single-name equity risk |
| SPY | 20% | Broad equity beta |
| TLT | 15% | Duration exposure |
| GLD | 15% | Defensive/commodity exposure |

Primary data source: Yahoo Finance adjusted close prices, cached locally under `data/cache/`.

Offline sample data is available at `data/sample_prices.csv` for deterministic tests and demonstrations.

## Method Inventory

| Method | Purpose | Main assumption | Validation focus |
|---|---|---|---|
| Historical VaR/ES | Empirical benchmark | Past returns are informative | Rolling exceptions |
| Parametric VaR/ES | Analytical benchmark | Normal return distribution | Tail underestimation |
| Monte Carlo GBM | Scenario generation | GBM dynamics, fixed covariance | Simulation stability |
| GARCH-MC | Time-varying volatility | Conditional volatility forecast is informative | Forecast sensitivity |
| EVT POT/GPD | Extreme tail modelling | GPD tail above threshold | Threshold stability |
| Gaussian Copula | Dependence benchmark | No tail dependence | Crisis co-movement miss |
| Student-t Copula | Tail dependence model | Symmetric tail dependence | Degrees-of-freedom stability |
| Component VaR | Risk attribution | Delta-normal homogeneity | Euler additivity |

## Backtesting Evidence

The pipeline reports:

- Kupiec Proportion-of-Failures test for unconditional coverage
- Christoffersen conditional coverage test for independence and clustering
- Basel traffic-light classification only when the test is run under the standard 99% VaR / 250-observation setting

For non-standard settings such as 95% VaR over longer windows, the Basel traffic-light label is intentionally not applied.

## EVT Validation

The EVT module reports:

- GPD Q-Q plot
- Mean-excess table
- Threshold-stability table for shape and scale parameters
- Historical, Parametric, and EVT VaR/ES comparison across high confidence levels

The threshold remains a modelling judgment. The diagnostics are intended to make that judgment transparent.

## Known Model Limitations

- Yahoo Finance data is not institutional market data.
- GBM ignores jumps, volatility smiles, stochastic rates, and liquidity effects.
- GARCH-MC changes volatility forecasts but keeps the historical correlation matrix fixed.
- EVT results are sensitive to threshold selection and sample period.
- Student-t copula captures symmetric tail dependence but not asymmetric crash dependence.
- Component VaR uses delta-normal assumptions and is not a full revaluation attribution.

## Challenger Models

Useful extensions:

- Filtered Historical Simulation using GARCH-standardized residuals
- EWMA VaR challenger against Historical VaR
- Cornish-Fisher VaR for skew/kurtosis adjustment
- EVT with threshold selected by stability criteria
- Dynamic Conditional Correlation GARCH for time-varying correlation

## Validation Conclusion

The project is suitable as a market-risk and model-validation case study. The key interview message is not that one model is "correct", but that each model exposes a different assumption and must be challenged through backtesting, diagnostics, and sensitivity analysis.
