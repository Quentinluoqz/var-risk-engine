# FRTB And Basel Discussion

## Why This Matters

The Fundamental Review of the Trading Book (FRTB) changed the market-risk conversation from simple VaR reporting toward Expected Shortfall, liquidity horizons, modellability, and stronger desk-level validation.

This project does not implement a full FRTB capital engine. It provides a compact analytical bridge from classic VaR methods to the kinds of questions asked in market-risk and model-risk interviews.

## VaR Versus Expected Shortfall

VaR answers: what loss threshold is exceeded with probability `1 - alpha`?

Expected Shortfall answers: conditional on exceeding that threshold, what is the average loss?

ES is preferred for tail-risk capital because it is coherent and more sensitive to the severity of tail losses. This is why the project reports ES next to VaR throughout the pipeline.

## Basel Traffic Light

Basel traffic-light backtesting is tied to a specific regulatory context:

- 99% one-day VaR
- 250 backtesting observations
- Exception counts mapped into green, yellow, and red zones

The project therefore applies the traffic-light label only under that standard setting. For 95% VaR or longer windows, it reports Kupiec and Christoffersen diagnostics without assigning a Basel zone.

## FRTB Concepts Not Fully Implemented

| FRTB concept | Project treatment | Gap |
|---|---|---|
| Expected Shortfall | Implemented as historical and parametric ES | Not a full regulatory ES capital calculation |
| Liquidity horizons | Discussed only | No liquidity-horizon scaling by risk factor bucket |
| Risk-factor modellability | Not implemented | No RFET / NMRF classification |
| P&L attribution | Not implemented | No hypothetical vs risk-theoretical P&L test |
| Backtesting | Kupiec, Christoffersen, Basel helper | Not desk-level regulatory backtesting |
| Default risk charge | Not implemented | No credit jump-to-default capital |

## Interview Framing

For BNP Paribas, Societe Generale, Natixis, Amundi, or AXA-style risk discussions, the strongest framing is:

> I built a modular market-risk engine to compare VaR, ES, GARCH, EVT, copula, stress testing, and attribution methods. I then tightened the project with model-validation diagnostics, correct Basel backtesting scope, and documentation of the gap between an analytical prototype and a regulatory capital engine.

This is more credible than claiming the project is a production or regulatory-grade system.

## Recommended Next Extension

The most valuable next technical extension would be Filtered Historical Simulation:

1. Fit GARCH to returns
2. Standardize residuals
3. Bootstrap residuals
4. Reapply current conditional volatility
5. Compare FHS VaR/ES against Historical, Parametric, MC, and EVT

That extension would connect volatility modelling, non-parametric simulation, and validation in a way that is highly relevant to market-risk interviews.
