"""Plotting helpers for VaR risk engine reports."""

from __future__ import annotations

# ruff: noqa: E402

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "var-risk-engine-matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import norm  # noqa: E402

from var_risk_engine.copula import tail_dependence_gaussian, tail_dependence_t
from var_risk_engine.evt import gpd_qq_plot
from var_risk_engine.var_historical import historical_var
from var_risk_engine.var_parametric import parametric_var

sns.set_theme(style="whitegrid", palette="muted")

COLORS = {
    "primary": "#1B3A5C",
    "secondary": "#E8734A",
    "tertiary": "#4CAF50",
    "quaternary": "#9C27B0",
    "bg": "#FAFAFA",
}


def ensure_figures_dir(output_dir: str | Path) -> Path:
    """Create and return a figure output directory."""
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def plot_correlation_comparison(
    cov_methods: dict[str, np.ndarray],
    tickers: list[str],
    figures_dir: str | Path,
) -> None:
    """Plot correlation matrices produced by multiple covariance estimators."""
    from var_risk_engine.covariance import correlation_from_covariance

    figures_dir = ensure_figures_dir(figures_dir)
    fig, axes = plt.subplots(1, len(cov_methods), figsize=(16, 5))
    if len(cov_methods) == 1:
        axes = [axes]
    for ax, (name, cov_mat) in zip(axes, cov_methods.items()):
        corr = correlation_from_covariance(cov_mat)
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=tickers,
            yticklabels=tickers,
            ax=ax,
            square=True,
        )
        ax.set_title(name, fontsize=11, color=COLORS["primary"])
    fig.suptitle("Correlation Matrices - Three Estimation Methods", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "correlation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cholesky_factor(L: np.ndarray, tickers: list[str], figures_dir: str | Path) -> None:
    """Plot a Cholesky factor heatmap."""
    figures_dir = ensure_figures_dir(figures_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        L,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        xticklabels=tickers,
        yticklabels=tickers,
        ax=ax,
        square=True,
    )
    ax.set_title("Cholesky Factor L  (L @ L^T = Sigma)", fontsize=12, color=COLORS["primary"])
    plt.tight_layout()
    fig.savefig(figures_dir / "cholesky_factor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_var_distributions(
    port_returns: np.ndarray,
    mc_returns: np.ndarray,
    figures_dir: str | Path,
    confidence: float = 0.95,
) -> None:
    """Plot historical and Monte Carlo return distributions with VaR overlays."""
    figures_dir = ensure_figures_dir(figures_dir)
    h_var = historical_var(port_returns, confidence)
    p_var = parametric_var(port_returns, confidence)
    mc_var = -np.quantile(mc_returns, 1 - confidence)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(
        port_returns,
        bins=80,
        color=COLORS["primary"],
        alpha=0.6,
        edgecolor="white",
        density=True,
        label="Historical returns",
    )
    ax.axvline(-h_var, color=COLORS["secondary"], linestyle="--", linewidth=2, label=f"Historical VaR = {h_var:.4f}")
    ax.axvline(-p_var, color=COLORS["tertiary"], linestyle="--", linewidth=2, label=f"Parametric VaR = {p_var:.4f}")
    ax.axvline(-mc_var, color=COLORS["quaternary"], linestyle=":", linewidth=2, label=f"Monte Carlo VaR = {mc_var:.4f}")
    ax.set_xlabel("Daily Return", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Portfolio Return Distribution with VaR Thresholds", fontsize=12, color=COLORS["primary"])
    ax.legend(fontsize=9)

    ax = axes[1]
    sorted_returns = np.sort(port_returns)
    theoretical = np.linspace(0.001, 0.999, len(sorted_returns))
    z_theo = norm.ppf(theoretical)
    z_emp = (sorted_returns - np.mean(port_returns)) / np.std(port_returns)
    ax.scatter(z_theo, z_emp, s=3, alpha=0.4, color=COLORS["primary"])
    lim = max(abs(z_theo.min()), abs(z_theo.max()))
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.5, label="Normal reference")
    ax.set_xlabel("Theoretical Quantiles (Standard Normal)", fontsize=11)
    ax.set_ylabel("Empirical Quantiles (Standardised)", fontsize=11)
    ax.set_title("Q-Q Plot - Normality Check", fontsize=12, color=COLORS["primary"])
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(figures_dir / "var_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_montecarlo(mc_returns: np.ndarray, figures_dir: str | Path, confidence: float = 0.95) -> None:
    """Plot Monte Carlo simulated portfolio returns."""
    figures_dir = ensure_figures_dir(figures_dir)
    var_mc = -np.quantile(mc_returns, 1 - confidence)
    es_mc = -np.mean(mc_returns[mc_returns <= -var_mc])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        mc_returns,
        bins=120,
        color=COLORS["primary"],
        alpha=0.5,
        edgecolor="white",
        density=True,
        label=f"MC simulated returns (n={len(mc_returns):,})",
    )
    ax.axvline(-var_mc, color=COLORS["secondary"], linewidth=2, linestyle="--", label=f"VaR ({confidence:.0%}) = {var_mc:.4f}")
    ax.axvline(-es_mc, color=COLORS["tertiary"], linewidth=2, linestyle="--", label=f"ES ({confidence:.0%}) = {es_mc:.4f}")
    ax.fill_between(
        [mc_returns.min(), -var_mc],
        0,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 20,
        alpha=0.15,
        color=COLORS["secondary"],
        label="Tail region (loss > VaR)",
    )
    ax.set_xlabel("Simulated Portfolio Return", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Monte Carlo VaR - GBM Simulation ({len(mc_returns):,} paths)",
        fontsize=12,
        color=COLORS["primary"],
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(figures_dir / "montecarlo_var.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_backtesting(bt: dict, confidence: float, basel_label: str, figures_dir: str | Path) -> None:
    """Plot rolling VaR backtesting results."""
    figures_dir = ensure_figures_dir(figures_dir)
    n_viol = bt["n_violations"]
    n_total = len(bt["var_forecasts"])
    dates = bt["dates_idx"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.fill_between(
        dates,
        -bt["var_forecasts"],
        color=COLORS["primary"],
        alpha=0.15,
        label=f"VaR band ({confidence:.0%})",
    )
    ax.plot(
        dates,
        bt["actual_returns"],
        color=COLORS["primary"],
        linewidth=0.7,
        alpha=0.8,
        label="Actual returns",
    )
    violation_idx = dates[bt["violations"]]
    violation_ret = bt["actual_returns"][bt["violations"]]
    ax.scatter(violation_idx, violation_ret, color=COLORS["secondary"], s=25, zorder=5, label=f"Violations ({n_viol})")
    ax.set_ylabel("Daily Return", fontsize=11)
    ax.set_title(
        f"Historical VaR Backtest - {n_viol} violations / {n_total} days "
        f"(Basel: {basel_label})",
        fontsize=12,
        color=COLORS["primary"],
    )
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    cum_violations = np.cumsum(bt["violations"].astype(int))
    expected_cum = np.arange(1, n_total + 1) * (1 - confidence)
    ax.plot(dates, cum_violations, color=COLORS["secondary"], linewidth=1.5, label="Cumulative violations")
    ax.plot(dates, expected_cum, color=COLORS["tertiary"], linewidth=1.5, linestyle="--", label="Expected (linear)")
    ax.fill_between(dates, expected_cum * 0.5, expected_cum * 1.5, alpha=0.1, color=COLORS["tertiary"], label="Acceptable band (0.5x-1.5x)")
    ax.set_xlabel("Trading Day Index", fontsize=11)
    ax.set_ylabel("Cumulative Violations", fontsize=11)
    ax.set_title("Cumulative VaR Violations vs Expected", fontsize=11, color=COLORS["primary"])
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(figures_dir / "backtesting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_garch_volatility(
    returns: pd.DataFrame,
    port_ret: pd.Series,
    cond_vol: pd.Series,
    figures_dir: str | Path,
) -> None:
    """Plot portfolio returns against conditional volatility."""
    figures_dir = ensure_figures_dir(figures_dir)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    ax = axes[0]
    ax.plot(returns.index, port_ret.values, color=COLORS["primary"], linewidth=0.6, alpha=0.7, label="Daily returns")
    ax.set_ylabel("Daily Return", fontsize=11)
    ax.set_title("Portfolio Returns vs GARCH(1,1) Conditional Volatility", fontsize=12, color=COLORS["primary"])
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    ax.plot(cond_vol.index, cond_vol.values, color=COLORS["secondary"], linewidth=1.0, label="Conditional Volatility (GARCH)")
    ax.axhline(port_ret.std(), color=COLORS["tertiary"], linestyle="--", linewidth=1.5, label=f"Unconditional sigma = {port_ret.std():.4f}")
    ax.set_ylabel("Volatility (daily)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Time-Varying Volatility - GARCH vs Constant", fontsize=11, color=COLORS["primary"])
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(figures_dir / "garch_volatility.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_garch_per_asset(vol_by_ticker: dict[str, pd.Series], figures_dir: str | Path) -> None:
    """Plot per-asset conditional volatility."""
    figures_dir = ensure_figures_dir(figures_dir)
    tickers = list(vol_by_ticker)
    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 3 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        cv = vol_by_ticker[ticker]
        ax.plot(cv.index, cv.values, color=COLORS["primary"], linewidth=0.8)
        ax.set_ylabel("sigma (daily)", fontsize=9)
        ax.set_title(f"{ticker}", fontsize=10, color=COLORS["primary"])
    axes[-1].set_xlabel("Date", fontsize=11)
    fig.suptitle("Per-Asset GARCH(1,1) Conditional Volatility", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(figures_dir / "garch_per_asset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stress_testing(
    hist_stress: pd.DataFrame,
    hypo_stress: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    """Plot historical and hypothetical stress results."""
    figures_dir = ensure_figures_dir(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    hist_clean = hist_stress.dropna(subset=["cumulative_return"])
    if not hist_clean.empty:
        colors = [COLORS["secondary"] if v < 0 else COLORS["tertiary"] for v in hist_clean["cumulative_return"]]
        bars = ax.barh(range(len(hist_clean)), hist_clean["cumulative_return"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(hist_clean)))
        ax.set_yticklabels(hist_clean.index, fontsize=9)
        ax.set_xlabel("Cumulative Return", fontsize=11)
        ax.set_title("Historical Stress - Cumulative Portfolio Return", fontsize=11, color=COLORS["primary"])
        ax.axvline(0, color="black", linewidth=0.8)
        for bar, val in zip(bars, hist_clean["cumulative_return"]):
            ax.text(val - 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", ha="right", fontsize=9, color="white", fontweight="bold")

    ax = axes[1]
    hypo_clean = hypo_stress[["var_95", "es_95"]].dropna()
    x_pos = np.arange(len(hypo_clean))
    width = 0.35
    ax.bar(x_pos - width / 2, hypo_clean["var_95"], width, label="VaR (95%)", color=COLORS["secondary"], edgecolor="white")
    ax.bar(x_pos + width / 2, hypo_clean["es_95"], width, label="ES (95%)", color=COLORS["quaternary"], edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hypo_clean.index, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Risk Measure", fontsize=11)
    ax.set_title("Hypothetical Stress - Stressed VaR & ES", fontsize=11, color=COLORS["primary"])
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(figures_dir / "stress_testing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_evt_analysis(
    losses: np.ndarray,
    gpd_params: dict,
    mean_excess: pd.DataFrame,
    stability: pd.DataFrame,
    evt_cmp: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    """Plot EVT Q-Q, threshold diagnostics, and VaR comparison."""
    figures_dir = ensure_figures_dir(figures_dir)
    fig_qq, ax_qq = gpd_qq_plot(losses, gpd_params)
    ax_qq.set_title(ax_qq.get_title(), fontsize=12, color=COLORS["primary"])
    fig_qq.savefig(figures_dir / "evt_gpd_qq.png", dpi=150, bbox_inches="tight")
    plt.close(fig_qq)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(mean_excess["threshold"], mean_excess["mean_excess"], marker="o", color=COLORS["primary"])
    axes[0].set_xlabel("Threshold", fontsize=11)
    axes[0].set_ylabel("Mean Excess", fontsize=11)
    axes[0].set_title("Mean Excess Diagnostic", fontsize=11, color=COLORS["primary"])

    valid_stability = stability.dropna(subset=["xi"])
    axes[1].plot(valid_stability["threshold"], valid_stability["xi"], marker="o", color=COLORS["secondary"], label="GPD shape xi")
    axes[1].axhline(0, color="gray", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Threshold", fontsize=11)
    axes[1].set_ylabel("Shape Parameter xi", fontsize=11)
    axes[1].set_title("Threshold Stability", fontsize=11, color=COLORS["primary"])
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(figures_dir / "evt_threshold_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    confs = evt_cmp["confidence"].values
    x_pos = np.arange(len(confs))
    width = 0.25
    ax.bar(x_pos - width, evt_cmp["var_historical"], width, label="Historical VaR", color=COLORS["primary"], edgecolor="white")
    ax.bar(x_pos, evt_cmp["var_parametric"], width, label="Parametric VaR", color=COLORS["tertiary"], edgecolor="white")
    ax.bar(x_pos + width, evt_cmp["var_evt"], width, label="EVT VaR (GPD)", color=COLORS["secondary"], edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{c:.1%}" for c in confs], fontsize=10)
    ax.set_xlabel("Confidence Level", fontsize=11)
    ax.set_ylabel("VaR Estimate", fontsize=11)
    ax.set_title("VaR Comparison: Historical vs Parametric vs EVT", fontsize=12, color=COLORS["primary"])
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "evt_var_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_copula_analysis(
    tickers: list[str],
    gauss_params: dict,
    t_params: dict,
    figures_dir: str | Path,
) -> None:
    """Plot copula tail-dependence and correlation matrices."""
    figures_dir = ensure_figures_dir(figures_dir)
    n = len(tickers)
    t_corr = t_params["corr_matrix"]
    nu = t_params["nu"]

    gauss_td = np.zeros((n, n))
    t_td = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                gauss_td[i, j] = tail_dependence_gaussian(t_corr[i, j])
                t_td[i, j] = tail_dependence_t(t_corr[i, j], nu)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(gauss_td, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=tickers, yticklabels=tickers, ax=axes[0], vmin=0, vmax=0.5, square=True)
    axes[0].set_title("Gaussian Copula - Tail Dependence\n(always 0)", fontsize=11, color=COLORS["primary"])
    sns.heatmap(t_td, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=tickers, yticklabels=tickers, ax=axes[1], vmin=0, vmax=0.5, square=True)
    axes[1].set_title(f"t-Copula - Tail Dependence  (nu={nu:.0f})", fontsize=11, color=COLORS["primary"])
    fig.suptitle("Tail Dependence: Gaussian vs t-Copula", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "copula_tail_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    gauss_corr = gauss_params["corr_matrix"]
    sns.heatmap(gauss_corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, xticklabels=tickers, yticklabels=tickers, ax=axes[0], square=True)
    axes[0].set_title("Gaussian Copula Correlation", fontsize=11, color=COLORS["primary"])
    sns.heatmap(t_corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, xticklabels=tickers, yticklabels=tickers, ax=axes[1], square=True)
    axes[1].set_title(f"t-Copula Correlation  (nu={nu:.0f})", fontsize=11, color=COLORS["primary"])
    fig.suptitle("Copula Correlation Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "copula_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_attribution(summary: dict, figures_dir: str | Path) -> None:
    """Plot risk contribution, weights, marginal VaR, and component VaR."""
    figures_dir = ensure_figures_dir(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    attr_df = summary["attribution_table"]
    asset_rows = attr_df[attr_df["ticker"] != "TOTAL"]
    sorted_tickers = asset_rows["ticker"].values
    risk_pcts = asset_rows["risk_contribution_pct"].values

    colors_pie = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"], "#FFB74D"][:len(sorted_tickers)]
    if np.any(risk_pcts < 0):
        bar_colors = [COLORS["secondary"] if val >= 0 else COLORS["tertiary"] for val in risk_pcts]
        axes[0].barh(sorted_tickers, risk_pcts, color=bar_colors, edgecolor="white")
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Risk Contribution (%)", fontsize=10)
        axes[0].set_title("Risk Contribution (%)\nNegative values indicate hedging effect", fontsize=11, color=COLORS["primary"])
    else:
        axes[0].pie(risk_pcts, labels=sorted_tickers, autopct="%1.1f%%", colors=colors_pie, startangle=140, textprops={"fontsize": 10})
        axes[0].set_title("Risk Contribution (%)\nEuler Decomposition", fontsize=11, color=COLORS["primary"])

    weight_pcts = asset_rows["weight_pct"].values
    x_pos = np.arange(len(sorted_tickers))
    width = 0.35
    axes[1].bar(x_pos - width / 2, weight_pcts, width, label="Weight (%)", color=COLORS["primary"], edgecolor="white")
    axes[1].bar(x_pos + width / 2, risk_pcts, width, label="Risk Contribution (%)", color=COLORS["secondary"], edgecolor="white")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(sorted_tickers, fontsize=10)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].set_title("Weight vs Risk Contribution", fontsize=11, color=COLORS["primary"])
    axes[1].legend(fontsize=10)
    axes[1].axhline(100 / len(sorted_tickers), color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Equal risk budget")
    plt.tight_layout()
    fig.savefig(figures_dir / "risk_attribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    mvar_vals = asset_rows["marginal_var"].values
    cvar_vals = asset_rows["component_var"].values
    x_pos = np.arange(len(sorted_tickers))
    width = 0.35
    ax.bar(x_pos - width / 2, mvar_vals, width, label="Marginal VaR", color=COLORS["tertiary"], edgecolor="white")
    ax.bar(x_pos + width / 2, cvar_vals, width, label="Component VaR", color=COLORS["quaternary"], edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_tickers, fontsize=10)
    ax.set_ylabel("VaR Contribution", fontsize=11)
    ax.set_title("Marginal VaR & Component VaR per Asset", fontsize=12, color=COLORS["primary"])
    ax.legend(fontsize=10)
    for bar_group in [mvar_vals, cvar_vals]:
        for idx, val in enumerate(bar_group):
            xpos = x_pos[idx] + (-width / 2 if bar_group is mvar_vals else width / 2)
            ax.text(xpos, val + 0.0005, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(figures_dir / "risk_marginal_component.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
