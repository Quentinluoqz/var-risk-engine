"""Structured output export for full analysis runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def prepare_output_dir(output_dir: str | Path | None) -> Path:
    """Create and return the run output directory."""
    if output_dir is None:
        output_path = Path("outputs") / "latest"
    else:
        output_path = Path(output_dir)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)
    return output_path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable indentation."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_table(path: str | Path, table: pd.DataFrame) -> None:
    """Write a DataFrame to CSV."""
    table.to_csv(path, index=False)


def write_html_report(
    path: str | Path,
    metrics: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    figure_names: list[str],
) -> None:
    """Write a compact static HTML report for offline review."""
    sections = [
        "<html><head><meta charset='utf-8'><title>VaR Risk Engine Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45}"
        "table{border-collapse:collapse;margin:12px 0 24px 0}"
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "img{max-width:920px;width:100%;margin:10px 0 28px 0}"
        "code{background:#f4f4f4;padding:2px 4px}</style></head><body>",
        "<h1>VaR Risk Engine Report</h1>",
        "<h2>Run Metrics</h2>",
        "<pre>",
        json.dumps(metrics, indent=2, sort_keys=True),
        "</pre>",
    ]

    for name, table in tables.items():
        sections.append(f"<h2>{name.replace('_', ' ').title()}</h2>")
        sections.append(table.to_html(index=False, float_format=lambda x: f"{x:.6f}"))

    sections.append("<h2>Figures</h2>")
    for figure_name in figure_names:
        sections.append(f"<h3>{figure_name}</h3>")
        sections.append(f"<img src='figures/{figure_name}' alt='{figure_name}'>")

    sections.append("</body></html>")

    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))
