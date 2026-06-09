"""Configuration loading for the VaR risk engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class EngineConfig:
    """Runtime configuration for a full analysis run."""

    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "SPY", "TLT", "GLD"])
    weights: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.25, 0.20, 0.15, 0.15], dtype=float)
    )
    start: str = "2019-01-01"
    end: str | None = None
    source: str = "yahoo"
    input_path: str | None = None
    confidence: float = 0.95
    confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99])
    horizon_days: int = 1
    n_sims: int = 20_000
    backtest_window: int = 500
    backtest_method: str = "historical"
    output_dir: str | None = None


def load_config(path: str | Path) -> EngineConfig:
    """Load an ``EngineConfig`` from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return config_from_mapping(raw)


def config_from_mapping(raw: dict[str, Any]) -> EngineConfig:
    """Build an ``EngineConfig`` from a nested mapping."""
    portfolio = raw.get("portfolio", {})
    data = raw.get("data", {})
    risk = raw.get("risk", {})
    backtest = raw.get("backtest", {})
    output = raw.get("output", {})

    tickers = list(portfolio.get("tickers", ["AAPL", "MSFT", "SPY", "TLT", "GLD"]))
    weights = np.asarray(
        portfolio.get("weights", [0.25, 0.25, 0.20, 0.15, 0.15]),
        dtype=float,
    )

    return EngineConfig(
        tickers=tickers,
        weights=weights,
        start=str(data.get("start", "2019-01-01")),
        end=data.get("end"),
        source=str(data.get("source", "yahoo")),
        input_path=data.get("input"),
        confidence=float(risk.get("confidence", risk.get("confidence_levels", [0.95])[0])),
        confidence_levels=[float(x) for x in risk.get("confidence_levels", [0.95, 0.99])],
        horizon_days=int(risk.get("horizon_days", 1)),
        n_sims=int(risk.get("n_sims", 20_000)),
        backtest_window=int(backtest.get("window", 500)),
        backtest_method=str(backtest.get("method", "historical")),
        output_dir=output.get("dir"),
    )
