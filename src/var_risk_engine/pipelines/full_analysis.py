"""Full end-to-end VaR analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run_full_analysis(
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
    """Run the complete analysis pipeline.

    This module is the orchestration layer used by the CLI. Plotting lives in
    ``reporting.plots``; section-level helpers remain in ``main.py`` for
    backward compatibility.
    """
    from var_risk_engine.main import main

    main(
        tickers=tickers,
        weights=weights,
        start=start,
        end=end,
        input_path=input_path,
        output_dir=output_dir,
        confidence=confidence,
        n_sims=n_sims,
        backtest_window=backtest_window,
    )
