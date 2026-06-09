"""Command-line interface for the VaR risk engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from var_risk_engine.config import EngineConfig, load_config


def _parse_csv(value: str) -> list[str]:
    """Parse a comma-separated string into a clean list of tokens."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return items


def _parse_weights(value: str) -> np.ndarray:
    """Parse comma-separated weights into a numpy array."""
    try:
        weights = np.array([float(item.strip()) for item in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must be comma-separated numbers") from exc
    if weights.size == 0:
        raise argparse.ArgumentTypeError("weights must contain at least one number")
    return weights


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="var-engine",
        description="Run a portfolio VaR/ES analysis with backtesting and model diagnostics.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file, for example configs/demo.yaml.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Local price CSV to run fully offline, for example data/sample_prices.csv.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run using the bundled offline sample price file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for metrics, tables, figures, and report.html.",
    )
    parser.add_argument(
        "--tickers",
        type=_parse_csv,
        default=["AAPL", "MSFT", "SPY", "TLT", "GLD"],
        help="Comma-separated tickers, for example AAPL,MSFT,SPY,TLT,GLD.",
    )
    parser.add_argument(
        "--weights",
        type=_parse_weights,
        default=np.array([0.25, 0.25, 0.20, 0.15, 0.15]),
        help="Comma-separated portfolio weights. Must match tickers and sum to 1.",
    )
    parser.add_argument("--start", default="2019-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="VaR confidence level, such as 0.95 or 0.99.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=20_000,
        help="Number of Monte Carlo simulations.",
    )
    parser.add_argument(
        "--backtest-window",
        type=int,
        default=500,
        help="Rolling backtest estimation window.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> EngineConfig:
    """Merge config-file settings with explicit CLI arguments."""
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = EngineConfig()

    if args.offline:
        config.source = "csv"
        config.input_path = "data/sample_prices.csv"
        if args.output is None and config.output_dir is None:
            config.output_dir = "outputs/offline_demo"

    if args.input is not None:
        config.source = "csv"
        config.input_path = args.input
    if args.output is not None:
        config.output_dir = args.output

    # CLI scalar defaults intentionally override config only when no config is used.
    # This keeps `--config` predictable while preserving old CLI behavior.
    if args.config is None:
        config.tickers = args.tickers
        config.weights = args.weights
        config.start = args.start
        config.end = args.end
        config.confidence = args.confidence
        config.n_sims = args.n_sims
        config.backtest_window = args.backtest_window

    if config.source == "csv" and config.input_path is None:
        raise argparse.ArgumentTypeError("CSV/offline runs require --input or data.input in config")

    if config.input_path is not None and not Path(config.input_path).exists():
        raise argparse.ArgumentTypeError(f"Input CSV does not exist: {config.input_path}")

    return config


def cli_main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and run the analysis."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except (argparse.ArgumentTypeError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if len(config.tickers) != len(config.weights):
        parser.error("--tickers and --weights must have the same length")
    if not np.isclose(float(np.sum(config.weights)), 1.0, atol=1e-6):
        parser.error("--weights must sum to 1.0")
    if not 0 < config.confidence < 1:
        parser.error("--confidence must be in (0, 1)")
    if config.n_sims < 1_000:
        parser.error("--n-sims must be at least 1000")
    if config.backtest_window < 20:
        parser.error("--backtest-window must be at least 20")

    from var_risk_engine.pipelines.full_analysis import run_full_analysis

    run_full_analysis(
        tickers=config.tickers,
        weights=config.weights,
        start=config.start,
        end=config.end,
        input_path=config.input_path,
        output_dir=config.output_dir,
        confidence=config.confidence,
        n_sims=config.n_sims,
        backtest_window=config.backtest_window,
    )


def main_entry() -> None:
    """Setuptools console-script wrapper."""
    cli_main()


def main() -> None:
    """Backward-compatible console-script wrapper for older editable installs."""
    main_entry()


if __name__ == "__main__":
    main_entry()
