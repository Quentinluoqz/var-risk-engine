from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Portfolio:
    """A collection of assets with associated weights.

    Attributes:
        name: Human-readable portfolio identifier.
        tickers: List of asset ticker symbols (e.g. ["AAPL", "MSFT"]).
        weights: numpy array of portfolio weights, must sum to 1.0.
        benchmark: Optional benchmark ticker for relative performance tracking.
    """

    name: str
    tickers: list[str]
    weights: np.ndarray
    benchmark: str | None = None

    def __post_init__(self) -> None:
        """Validate inputs after initialisation.

        Raises:
            TypeError: If *tickers* is empty or *weights* is not a numpy array.
            ValueError: If the number of tickers and weights disagree,
                        or if weights do not sum to 1.0 (within tolerance).
        """
        if not self.tickers:
            raise ValueError("tickers list must not be empty.")

        if not isinstance(self.weights, np.ndarray):
            raise TypeError(
                f"weights must be a numpy ndarray, got {type(self.weights).__name__}."
            )

        if len(self.tickers) != self.weights.shape[0]:
            raise ValueError(
                f"Length mismatch: {len(self.tickers)} tickers but "
                f"{self.weights.shape[0]} weights."
            )

        weight_sum: float = float(np.sum(self.weights))
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum:.8f}."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_assets(self) -> int:
        """Return the number of assets in the portfolio."""
        return len(self.tickers)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def describe(self) -> None:
        """Print a formatted summary table of the portfolio holdings."""
        header = f"Portfolio: {self.name}"
        separator = "-" * max(len(header), 40)

        print(separator)
        print(header)
        print(separator)

        if self.benchmark is not None:
            print(f"Benchmark : {self.benchmark}")

        print(f"Assets    : {self.n_assets}")
        print(separator)

        col_ticker = "Ticker"
        col_weight = "Weight"
        col_pct = "Weight %"
        print(f"  {col_ticker:<10s} {col_weight:>10s} {col_pct:>10s}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")

        for ticker, w in zip(self.tickers, self.weights):
            print(f"  {ticker:<10s} {w:>10.6f} {w * 100:>9.4f}%")

        print(separator)
        print(f"  {'Total':<10s} {float(np.sum(self.weights)):>10.6f} {float(np.sum(self.weights)) * 100:>9.4f}%")
        print(separator)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Portfolio(name={self.name!r}, tickers={self.tickers}, "
            f"weights=array({self.weights.tolist()}), benchmark={self.benchmark!r})"
        )
