"""Data-quality checks for market-risk input prices and returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_prices(
    prices: pd.DataFrame,
    stale_threshold: int = 5,
) -> pd.DataFrame:
    """Return data-quality issues found in a price matrix."""
    issues: list[dict[str, str | int | float]] = []

    if prices.empty:
        issues.append({"check": "empty_prices", "severity": "error", "detail": "Price table is empty."})
        return pd.DataFrame(issues)

    duplicate_count = int(prices.index.duplicated().sum())
    if duplicate_count:
        issues.append(
            {
                "check": "duplicate_dates",
                "severity": "error",
                "detail": "Duplicate dates found in price index.",
                "count": duplicate_count,
            }
        )

    missing_count = int(prices.isna().sum().sum())
    if missing_count:
        issues.append(
            {
                "check": "missing_prices",
                "severity": "error",
                "detail": "Missing price observations found.",
                "count": missing_count,
            }
        )

    non_positive_count = int((prices <= 0).sum().sum())
    if non_positive_count:
        issues.append(
            {
                "check": "non_positive_prices",
                "severity": "error",
                "detail": "Non-positive price observations found.",
                "count": non_positive_count,
            }
        )

    for column in prices.columns:
        unchanged = prices[column].diff().fillna(0).eq(0)
        run_lengths = unchanged.astype(int).groupby((~unchanged).cumsum()).cumsum()
        max_stale = int(run_lengths.max())
        if max_stale >= stale_threshold:
            issues.append(
                {
                    "check": "stale_prices",
                    "severity": "warning",
                    "detail": f"{column} has a stale price run.",
                    "count": max_stale,
                }
            )

    return pd.DataFrame(issues, columns=["check", "severity", "detail", "count"])


def validate_returns(
    returns: pd.DataFrame,
    outlier_z: float = 8.0,
) -> pd.DataFrame:
    """Return data-quality issues found in a return matrix."""
    issues: list[dict[str, str | int | float]] = []

    if returns.empty:
        issues.append({"check": "empty_returns", "severity": "error", "detail": "Return table is empty."})
        return pd.DataFrame(issues)

    missing_count = int(returns.isna().sum().sum())
    if missing_count:
        issues.append(
            {
                "check": "missing_returns",
                "severity": "error",
                "detail": "Missing return observations found.",
                "count": missing_count,
            }
        )

    for column in returns.columns:
        series = returns[column].dropna()
        sigma = float(series.std(ddof=1))
        if sigma <= 0 or not np.isfinite(sigma):
            issues.append(
                {
                    "check": "zero_return_volatility",
                    "severity": "error",
                    "detail": f"{column} has zero or invalid return volatility.",
                    "count": len(series),
                }
            )
            continue
        z_scores = (series - float(series.mean())) / sigma
        outliers = int((z_scores.abs() > outlier_z).sum())
        if outliers:
            issues.append(
                {
                    "check": "return_outliers",
                    "severity": "warning",
                    "detail": f"{column} has extreme return observations.",
                    "count": outliers,
                }
            )

    return pd.DataFrame(issues, columns=["check", "severity", "detail", "count"])


def has_errors(issues: pd.DataFrame) -> bool:
    """Return True when a validation issue table contains errors."""
    return not issues.empty and bool((issues["severity"] == "error").any())
