from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cache helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_CACHE_DIR = Path(__file__).resolve().parent / "data"
_CACHE_DIR = Path(
    os.environ.get(
        "VAR_RISK_ENGINE_CACHE_DIR",
        _PROJECT_ROOT / "data" / "cache",
    )
)


def _cache_path(ticker: str, start: str, end: str) -> Path:
    """Return the filesystem path for a cached CSV file."""
    return _CACHE_DIR / f"{ticker}_{start}_{end}.csv"


def _ensure_cache_dir() -> None:
    """Create the cache directory if it does not already exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_cached_file(ticker: str, start: str, end: str) -> Path | None:
    """Return the preferred cache path if present, else a legacy cache path."""
    preferred = _cache_path(ticker, start, end)
    if preferred.exists():
        return preferred

    legacy = _LEGACY_CACHE_DIR / preferred.name
    if legacy.exists():
        return legacy

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_prices_csv(
    path: str | Path,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load adjusted-close prices from a local CSV file.

    The CSV must contain a date column named ``Date`` or an index-like first
    column, plus one numeric price column per asset.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input price CSV not found: {csv_path}")

    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    prices.sort_index(inplace=True)

    if prices.empty:
        raise ValueError(f"Input price CSV is empty: {csv_path}")

    if tickers is not None:
        missing = [ticker for ticker in tickers if ticker not in prices.columns]
        if missing:
            raise ValueError(
                "Input price CSV is missing requested ticker columns: "
                f"{missing}"
            )
        prices = prices.loc[:, tickers]

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices.dropna(how="any", inplace=True)

    if (prices <= 0).any().any():
        raise ValueError("Input price CSV contains non-positive prices.")
    if prices.empty:
        raise ValueError("Input price CSV has no complete positive price rows.")

    return prices


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily adjusted-close prices for *tickers*.

    The function first checks a local CSV cache under ``data/``.  If a cached
    file exists for a given ticker it is loaded directly; otherwise the data is
    fetched from Yahoo Finance via *yfinance* and saved to the cache.

    Parameters
    ----------
    tickers:
        List of ticker symbols, e.g. ``["AAPL", "MSFT", "SPY"]``.
    start:
        Start date in ``YYYY-MM-DD`` format.
    end:
        End date in ``YYYY-MM-DD`` format.  Defaults to ``None`` (today).

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``DatetimeIndex`` named ``Date`` and one column per
        ticker containing adjusted close prices.

    Raises
    ------
    ValueError
        If *tickers* is empty or contains non-string elements.
    RuntimeError
        If *yfinance* raises an unexpected error during download.
    """
    if not tickers:
        raise ValueError("tickers list must not be empty.")

    for t in tickers:
        if not isinstance(t, str):
            raise ValueError(
                f"All tickers must be strings, got {type(t).__name__}: {t!r}"
            )

    # Resolve the end date string used for cache keys (yfinance handles None).
    end_str: str = end if end is not None else "latest"

    _ensure_cache_dir()

    frames: dict[str, pd.Series[float]] = {}

    for ticker in tickers:
        cached = _cache_path(ticker, start, end_str)
        source_cache = _resolve_cached_file(ticker, start, end_str)

        if source_cache is not None:
            logger.info("Loading %s from cache (%s).", ticker, source_cache.name)
            try:
                df_cached = pd.read_csv(
                    source_cache,
                    index_col=0,
                    parse_dates=True,
                )
                if isinstance(df_cached, pd.DataFrame):
                    series = df_cached.iloc[:, 0]
                else:
                    series = df_cached
                series.name = ticker
                frames[ticker] = series
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Cache read failed for %s (%s); re-downloading.",
                    ticker,
                    exc,
                )

        # Download from Yahoo Finance
        logger.info("Downloading %s from Yahoo Finance.", ticker)
        try:
            import yfinance as yf  # local import keeps startup fast

            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"yfinance download failed for {ticker!r}: {exc}"
            ) from exc

        if raw.empty:
            raise ValueError(
                f"No data returned for ticker {ticker!r} "
                f"(start={start}, end={end}). Verify the symbol is valid."
            )

        # yfinance >= 0.2 may return MultiIndex columns even for single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            close_data = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
            if isinstance(close_data, pd.DataFrame):
                close_col = close_data.iloc[:, 0]
            else:
                close_col = close_data
        else:
            close_col = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]

        close_col = close_col.astype(float)
        close_col.name = ticker

        # Persist to the preferred project-level cache.
        try:
            close_col.to_csv(cached)
            logger.info("Cached %s to %s.", ticker, cached.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache write failed for %s (%s).", ticker, exc)

        frames[ticker] = close_col

    prices = pd.concat(frames, axis=1)
    prices.index.name = "Date"
    prices.sort_index(inplace=True)
    # Drop rows where any ticker is missing data
    prices.dropna(how="any", inplace=True)
    return prices


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    """Compute daily returns from a price DataFrame.

    Parameters
    ----------
    prices:
        DataFrame of prices (one column per asset, ``DatetimeIndex``).
    method:
        ``"log"`` for continuously-compounded (log) returns or ``"simple"``
        for arithmetic returns.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily returns with the same shape/columns as *prices*,
        with the first row dropped (``NaN``-free).

    Raises
    ------
    ValueError
        If *method* is not ``"log"`` or ``"simple"``.
    """
    method = method.lower().strip()
    valid_methods = {"log", "simple"}

    if method not in valid_methods:
        raise ValueError(
            f"Invalid method {method!r}. Must be one of {sorted(valid_methods)}."
        )

    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    # Drop the first row which is NaN by construction
    returns = returns.iloc[1:].copy()
    returns.index.name = "Date"
    return returns


def fetch_and_prepare(
    tickers: list[str],
    start: str,
    end: str | None = None,
    input_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: fetch prices and compute log returns.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    start:
        Start date (``YYYY-MM-DD``).
    end:
        End date (``YYYY-MM-DD``).  Defaults to ``None`` (today).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(prices, returns)`` — both with ``DatetimeIndex`` and one column per
        ticker.  Returns use the log method by default.
    """
    if input_path is not None:
        prices = load_prices_csv(input_path, tickers=tickers)
    else:
        prices = fetch_prices(tickers, start=start, end=end)
    returns = compute_returns(prices, method="log")
    return prices, returns
