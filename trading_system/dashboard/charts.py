"""
Chart generation for backtest results.

Generates: equity curves, drawdown curves, monthly heatmaps,
rolling metrics, trade distributions, parameter sensitivity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class ChartGenerator:
    """Generates charts for backtest analysis."""

    def __init__(self, output_dir: str = "data/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def equity_curve(
        self,
        equity: pd.Series,
        title: str = "Equity Curve",
        filename: str = "equity_curve.png",
    ) -> str:
        """Plot equity curve."""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(equity.index, equity.values, linewidth=1.5, color="#2196F3")
        ax.fill_between(equity.index, equity.values, alpha=0.1, color="#2196F3")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def drawdown_curve(
        self,
        equity: pd.Series,
        title: str = "Drawdown",
        filename: str = "drawdown.png",
    ) -> str:
        """Plot drawdown curve."""
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.6)
        ax.plot(drawdown.index, drawdown.values, color="#F44336", linewidth=1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def monthly_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap",
        filename: str = "monthly_heatmap.png",
    ) -> str:
        """Plot monthly returns heatmap."""
        if returns.empty:
            return ""

        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

        # Create year x month matrix
        data = {}
        for date, ret in monthly.items():
            year = date.year
            month = date.month
            if year not in data:
                data[year] = {}
            data[year][month] = ret

        years = sorted(data.keys())
        months = range(1, 13)
        matrix = np.zeros((len(years), 12))

        for i, year in enumerate(years):
            for m in months:
                matrix[i, m - 1] = data[year].get(m, 0)

        fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.8)))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)

        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)

        # Annotate cells
        for i in range(len(years)):
            for j in range(12):
                val = matrix[i, j]
                if val != 0:
                    color = "white" if abs(val) > 0.05 else "black"
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                           fontsize=8, color=color)

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Return")
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def rolling_sharpe(
        self,
        equity: pd.Series,
        window: int = 500,
        title: str = "Rolling Sharpe Ratio",
        filename: str = "rolling_sharpe.png",
    ) -> str:
        """Plot rolling Sharpe ratio."""
        returns = equity.pct_change().dropna()
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(365 * 24)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1, color="#9C27B0")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Sharpe Ratio")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def trade_distribution(
        self,
        trades: list[dict],
        title: str = "Trade P&L Distribution",
        filename: str = "trade_distribution.png",
    ) -> str:
        """Plot trade P&L distribution."""
        if not trades:
            return ""

        pnls = [t.get("pnl", 0) for t in trades]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(pnls, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
        axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_title("P&L Distribution")
        axes[0].set_xlabel("P&L ($)")
        axes[0].set_ylabel("Count")

        # Cumulative P&L
        cum_pnl = np.cumsum(pnls)
        axes[1].plot(cum_pnl, color="#4CAF50", linewidth=1.5)
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_title("Cumulative P&L")
        axes[1].set_xlabel("Trade #")
        axes[1].set_ylabel("Cumulative P&L ($)")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def parameter_sensitivity(
        self,
        results_by_param: dict[str, list[dict]],
        param_name: str,
        metric: str = "sharpe",
        title: str = "Parameter Sensitivity",
        filename: str = "param_sensitivity.png",
    ) -> str:
        """Plot parameter sensitivity heatmap."""
        if not results_by_param:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))

        values = []
        metrics = []
        for param_val, results in results_by_param.items():
            vals = [r.get("results", {}).get(metric, 0) for r in results]
            values.append(float(param_val) if isinstance(param_val, (int, float)) else len(values))
            metrics.append(np.mean(vals) if vals else 0)

        ax.bar(range(len(values)), metrics, color="#FF9800", alpha=0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v}" for v in values], rotation=45)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def strategy_comparison(
        self,
        equity_curves: dict[str, pd.Series],
        title: str = "Strategy Comparison",
        filename: str = "strategy_comparison.png",
    ) -> str:
        """Plot multiple strategy equity curves."""
        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, (name, equity) in enumerate(equity_curves.items()):
            color = colors[i % len(colors)]
            ax.plot(equity.index, equity.values, label=name, linewidth=1.5, color=color)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Equity ($)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)
