"""
Backtest Chart Generator

Runs backtests for all 3 strategies + portfolio, then generates
professional matplotlib charts saved as PNGs for the report.

Charts generated:
  1. Individual equity curves + drawdown overlay
  2. Portfolio equity curve + drawdown
  3. Monthly returns heatmap
  4. Rolling Sharpe ratio
  5. Strategy correlation heatmap
  6. Drawdown comparison
  7. Trade distribution histogram
  8. Combined overview (4-panel)

Usage:
  python scripts/generate_charts.py           # Generate all charts
  python scripts/generate_charts.py --quick   # Fast mode (fewer candles)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0f",
    "axes.facecolor": "#111118",
    "axes.edgecolor": "#1a1a2e",
    "axes.labelcolor": "#888888",
    "text.color": "#e0e0e0",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "grid.color": "#1a1a2e",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.dpi": 150,
})

COLORS = {
    "equity": "#00d4ff",
    "equity2": "#4ade80",
    "equity3": "#fbbf24",
    "drawdown": "#f87171",
    "benchmark": "#555555",
    "positive": "#4ade80",
    "negative": "#f87171",
    "neutral": "#888888",
    "bg": "#0a0a0f",
    "card": "#111118",
}

CHARTS_DIR = Path("data/results/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, ".")

from trading_system.backtester.engine import BacktestEngine
from trading_system.config import BacktestConfig, ExecutionConfig, FeeConfig, SlippageConfig
from trading_system.strategies import get_strategy


# ── Strategy Definitions ─────────────────────────────────────────

STRATEGIES = [
    {
        "name": "MACD ETH 4h",
        "strategy": "MACD",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True},
        "weight": 0.41,
    },
    {
        "name": "ROC_Momentum ETH 4h",
        "strategy": "ROC_Momentum",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"roc_period": 10, "signal_period": 5, "use_ema": True, "ema_period": 12},
        "weight": 0.17,
    },
    {
        "name": "MACD BTC 4h",
        "strategy": "MACD",
        "pair": "BTC/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True},
        "weight": 0.43,
    },
]


def load_data(pair: str, timeframe: str) -> pd.DataFrame:
    """Load cached OHLCV data."""
    # Convert pair name: ETH/USDT:USDT -> ETH_USDT_USDT
    pair_dir = pair.replace("/", "_").replace(":", "_")
    # Try multiple naming patterns
    for pattern in [
        f"data/raw/{pair_dir}/{timeframe}.parquet",
        f"data/raw/{pair_dir}/klines_{timeframe}.parquet",
        f"data/raw/{pair}/{timeframe}.parquet",
        f"data/raw/{pair}/klines_{timeframe}.parquet",
    ]:
        p = Path(pattern)
        if p.exists():
            df = pd.read_parquet(p)
            # Ensure timestamp column is datetime
            if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            return df
    raise FileNotFoundError(f"No data for {pair}/{timeframe}")


def run_backtest(strat_def: dict) -> object:
    """Run a single strategy backtest and return BacktestResults."""
    df = load_data(strat_def["pair"], strat_def["timeframe"])
    strat = get_strategy(strat_def["strategy"])
    signals = strat.generate_signals(df, strat_def["params"])

    config = BacktestConfig(
        fees=FeeConfig(maker_fee=0.0002, taker_fee=0.0005),
        slippage=SlippageConfig(model="fixed", base_slippage=0.0001),
        execution=ExecutionConfig(
            model="next_open",
            initial_capital=10000.0,
            risk_per_trade=0.02,
            stop_loss_atr_mult=3.0,
        ),
    )
    engine = BacktestEngine(config)
    return engine.run(
        df=df, signals=signals,
        strategy_name=strat_def["strategy"],
        params=strat_def["params"],
        pair=strat_def["pair"],
        timeframe=strat_def["timeframe"],
    )


def build_portfolio_equity(results_list: list) -> tuple[pd.Series, pd.Series]:
    """Build portfolio equity from individual strategy equity curves (weight-averaged returns)."""
    returns_list = []
    for r, s in zip(results_list, STRATEGIES):
        eq = r.equity_curve
        ret = eq.pct_change().fillna(0)
        ret.name = s["name"]
        returns_list.append(ret * s["weight"])

    combined = pd.concat(returns_list, axis=1)
    port_returns = combined.sum(axis=1)
    port_equity = (1 + port_returns).cumprod() * 10000

    # Drawdown
    running_max = port_equity.cummax()
    drawdown = (port_equity - running_max) / running_max

    return port_equity, drawdown


def format_date_axis(ax, dates):
    """Format x-axis with dates."""
    if len(dates) > 500:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    elif len(dates) > 200:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


# ═══════════════════════════════════════════════════════════════
# Chart 1: Individual Equity Curves + Drawdown
# ═══════════════════════════════════════════════════════════════
def chart_individual_equity(results: list):
    fig, axes = plt.subplots(len(results), 2, figsize=(16, 4 * len(results)),
                              gridspec_kw={"width_ratios": [3, 1]})
    if len(results) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Individual Strategy Performance", fontsize=16, color=COLORS["equity"], y=0.98)

    for i, (r, s) in enumerate(zip(results, STRATEGIES)):
        dates = r.equity_curve.index
        eq = r.equity_curve.values
        dd = r.drawdown_curve.values

        # Equity curve
        ax = axes[i, 0]
        ax.plot(dates, eq, color=COLORS["equity"], linewidth=1.5, label="Equity")
        ax.fill_between(dates, 10000, eq, where=eq >= 10000, alpha=0.1, color=COLORS["positive"])
        ax.fill_between(dates, 10000, eq, where=eq < 10000, alpha=0.1, color=COLORS["negative"])
        ax.axhline(10000, color=COLORS["benchmark"], linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_ylabel("Equity ($)")
        ax.set_title(f"{s['name']}  |  Return: {r.total_return*100:.1f}%  |  Sharpe: {r.sharpe:.2f}  |  MaxDD: {r.max_drawdown*100:.2f}%",
                      fontsize=11, pad=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)
        format_date_axis(ax, dates)

        # Drawdown
        ax2 = axes[i, 1]
        ax2.fill_between(dates, 0, dd * 100, color=COLORS["drawdown"], alpha=0.6)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax2.grid(True, alpha=0.3)
        format_date_axis(ax2, dates)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = CHARTS_DIR / "01_individual_equity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
# Chart 2: Portfolio Equity Curve
# ═══════════════════════════════════════════════════════════════
def chart_portfolio_equity(port_equity: pd.Series, port_dd: pd.Series, results: list):
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 0.8], hspace=0.15)

    dates = port_equity.index

    # Equity
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, port_equity, color=COLORS["equity"], linewidth=2, label="Portfolio")
    for r, s, c in zip(results, STRATEGIES, [COLORS["equity2"], COLORS["equity3"], "#f472b6"]):
        ax1.plot(r.equity_curve.index, r.equity_curve.values, color=c, linewidth=0.8,
                 alpha=0.5, linestyle="--", label=s["name"])
    ax1.axhline(10000, color=COLORS["benchmark"], linestyle="--", alpha=0.3)
    ax1.fill_between(dates, 10000, port_equity, where=port_equity >= 10000,
                     alpha=0.08, color=COLORS["positive"])
    ax1.fill_between(dates, 10000, port_equity, where=port_equity < 10000,
                     alpha=0.08, color=COLORS["negative"])
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("Portfolio Equity Curve (Sharpe-Weighted)", fontsize=14, color=COLORS["equity"], pad=12)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)
    format_date_axis(ax1, dates)

    # Drawdown
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dates, 0, port_dd * 100, color=COLORS["drawdown"], alpha=0.7)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.grid(True, alpha=0.3)

    # Monthly returns bar
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    monthly = port_equity.resample("ME").last().pct_change().dropna() * 100
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in monthly.values]
    ax3.bar(monthly.index, monthly.values, width=20, color=colors, alpha=0.7)
    ax3.axhline(0, color=COLORS["benchmark"], linewidth=0.5)
    ax3.set_ylabel("Monthly %")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax3.grid(True, alpha=0.3)
    format_date_axis(ax3, dates)

    plt.tight_layout()
    path = CHARTS_DIR / "02_portfolio_equity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
# Chart 3: Monthly Returns Heatmap
# ═══════════════════════════════════════════════════════════════
def chart_monthly_heatmap(port_equity: pd.Series):
    monthly = port_equity.resample("ME").last().pct_change().dropna() * 100

    # Build year x month matrix
    years = sorted(monthly.index.year.unique())
    months = range(1, 13)
    data = np.full((len(years), 12), np.nan)

    for date, val in monthly.items():
        yi = years.index(date.year)
        mi = date.month - 1
        data[yi, mi] = val

    fig, ax = plt.subplots(figsize=(14, 3 + len(years) * 0.8))

    # Custom colormap: red -> black -> green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rg", [COLORS["negative"], "#111118", COLORS["positive"]])
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 3)

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_title("Monthly Returns Heatmap (%)", fontsize=14, color=COLORS["equity"], pad=12)

    # Annotate cells
    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(data[i, j]):
                color = "white" if abs(data[i, j]) > vmax * 0.5 else "#cccccc"
                ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Return %", shrink=0.8)
    plt.tight_layout()
    path = CHARTS_DIR / "03_monthly_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
# Chart 4: Rolling Sharpe
# ═══════════════════════════════════════════════════════════════
def chart_rolling_sharpe(results: list, port_equity: pd.Series):
    window = 500  # candles

    fig, ax = plt.subplots(figsize=(16, 5))

    for r, s, c in zip(results, STRATEGIES,
                        [COLORS["equity"], COLORS["equity2"], COLORS["equity3"]]):
        ret = r.equity_curve.pct_change().dropna()
        rolling = ret.rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        ax.plot(rolling.index, rolling.values, color=c, linewidth=1, alpha=0.7, label=s["name"])

    # Portfolio rolling sharpe
    port_ret = port_equity.pct_change().dropna()
    port_rolling = port_ret.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    ax.plot(port_rolling.index, port_rolling.values, color=COLORS["equity"],
            linewidth=2, label="Portfolio")

    ax.axhline(0, color=COLORS["benchmark"], linestyle="--", linewidth=0.8)
    ax.axhline(2, color=COLORS["positive"], linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Rolling Sharpe Ratio")
    ax.set_title(f"Rolling Sharpe Ratio ({window}-candle window)", fontsize=14, color=COLORS["equity"], pad=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    format_date_axis(ax, port_equity.index)

    plt.tight_layout()
    path = CHARTS_DIR / "05_rolling_sharpe.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
# Chart 5: Correlation Heatmap
# ═══════════════════════════════════════════════════════════════
def chart_correlation(results: list):
    returns = {}
    for r, s in zip(results, STRATEGIES):
        ret = r.equity_curve.pct_change().dropna()
        returns[s["name"]] = ret

    df = pd.DataFrame(returns)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom", ["#f87171", "#111118", "#4ade80"])

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    names = [s["name"] for s in STRATEGIES]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if abs(corr.values[i, j]) > 0.5 else "#cccccc"
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    ax.set_title("Strategy Return Correlation", fontsize=14, color=COLORS["equity"], pad=12)
    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    plt.tight_layout()
    path = CHARTS_DIR / "04_correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
# Chart 6: Drawdown Comparison
# ═══════════════════════════════════════════════════════════════
def chart_drawdown_comparison(results: list, port_dd: pd.Series):
    fig, ax = plt.subplots(figsize=(16, 5))

    for r, s, c in zip(results, STRATEGIES,
                        [COLORS["equity"], COLORS["equity2"], COLORS["equity3"]]):
        dd = r.drawdown_curve * 100
        ax.fill_between(dd.index, 0, dd.values, alpha=0.2, color=c)
        ax.plot(dd.index, dd.values, color=c, linewidth=0.8, label=s["name"])

    ax.fill_between(port_dd.index, 0, port_dd.values * 100, alpha=0.3, color=COLORS["equity"])
    ax.plot(port_dd.index, port_dd.values * 100, color=COLORS["equity"],
            linewidth=2, label="Portfolio")

    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Comparison", fontsize=14, color=COLORS["equity"], pad=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    format_date_axis(ax, port_dd.index)

    plt.tight_layout()
    path = CHARTS_DIR / "06_drawdown_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
# Chart 7: Trade Distribution
# ═══════════════════════════════════════════════════════════════
def chart_trade_distribution(results: list):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, r, s in zip(axes, results, STRATEGIES):
        trades = r.trades
        if not trades:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
            continue

        pnls = [t.get("pnl_pct", 0) * 100 for t in trades if "pnl_pct" in t]
        if not pnls:
            continue

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        ax.hist(losses, bins=30, color=COLORS["negative"], alpha=0.7, label=f"Losses ({len(losses)})")
        ax.hist(wins, bins=30, color=COLORS["positive"], alpha=0.7, label=f"Wins ({len(wins)})")
        ax.axvline(0, color=COLORS["benchmark"], linestyle="--", linewidth=1)
        ax.set_xlabel("Trade P&L (%)")
        ax.set_ylabel("Count")
        ax.set_title(s["name"], fontsize=11, pad=8)
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Trade P&L Distribution", fontsize=14, color=COLORS["equity"], y=1.02)
    plt.tight_layout()
    path = CHARTS_DIR / "07_trade_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
# Chart 8: Combined Overview (4-panel)
# ═══════════════════════════════════════════════════════════════
def chart_combined_overview(results: list, port_equity: pd.Series, port_dd: pd.Series):
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

    # Panel 1: Portfolio equity (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    dates = port_equity.index
    ax1.plot(dates, port_equity, color=COLORS["equity"], linewidth=2)
    ax1.fill_between(dates, 10000, port_equity, where=port_equity >= 10000,
                     alpha=0.08, color=COLORS["positive"])
    ax1.fill_between(dates, 10000, port_equity, where=port_equity < 10000,
                     alpha=0.08, color=COLORS["negative"])
    ax1.axhline(10000, color=COLORS["benchmark"], linestyle="--", alpha=0.3)
    ax1.set_title("Portfolio Equity Curve", fontsize=14, color=COLORS["equity"], pad=10)
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)
    format_date_axis(ax1, dates)

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, 0, port_dd * 100, color=COLORS["drawdown"], alpha=0.6)
    ax2.set_title("Portfolio Drawdown", fontsize=12, pad=8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.grid(True, alpha=0.3)
    format_date_axis(ax2, dates)

    # Panel 3: Strategy returns comparison (bar chart)
    ax3 = fig.add_subplot(gs[2, 0])
    names = [s["name"] for s in STRATEGIES]
    returns = [r.total_return * 100 for r in results]
    sharpes = [r.sharpe for r in results]
    x = np.arange(len(names))
    bars = ax3.bar(x, returns, color=[COLORS["equity"], COLORS["equity2"], COLORS["equity3"]], alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax3.set_ylabel("Total Return (%)")
    ax3.set_title("Individual Returns", fontsize=12, pad=8)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, ret, sharpe in zip(bars, returns, sharpes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{ret:.0f}%\nS:{sharpe:.1f}", ha="center", va="bottom", fontsize=8, color=COLORS["equity"])

    # Panel 4: Monthly returns bar
    ax4 = fig.add_subplot(gs[2, 1])
    monthly = port_equity.resample("ME").last().pct_change().dropna() * 100
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in monthly.values]
    ax4.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7, width=0.8)
    ax4.axhline(0, color=COLORS["benchmark"], linewidth=0.5)
    ax4.set_title("Monthly Returns (%)", fontsize=12, pad=8)
    ax4.set_ylabel("Return (%)")
    ax4.grid(True, alpha=0.3, axis="y")

    # Add year labels on x-axis
    tick_positions = []
    tick_labels = []
    for i, date in enumerate(monthly.index):
        if i == 0 or date.year != monthly.index[i-1].year:
            tick_positions.append(i)
            tick_labels.append(str(date.year))
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels, fontsize=9)

    fig.suptitle("Trading Strategy Research — Combined Overview", fontsize=18,
                 color=COLORS["equity"], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = CHARTS_DIR / "08_combined_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    quick = "--quick" in sys.argv

    print("=== Backtest Chart Generator ===")
    print()

    # Run backtests
    results = []
    for s in STRATEGIES:
        print(f"Running backtest: {s['name']}...")
        t0 = time.time()
        r = run_backtest(s)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s | Return: {r.total_return*100:.1f}% | Sharpe: {r.sharpe:.2f} | Trades: {r.total_trades}")
        results.append(r)

    # Build portfolio equity
    print("\nBuilding portfolio equity curve...")
    port_equity, port_dd = build_portfolio_equity(results)
    port_return = (port_equity.iloc[-1] / port_equity.iloc[0] - 1) * 100
    print(f"  Portfolio return: {port_return:.1f}%")

    # Generate charts
    print("\nGenerating charts...")
    paths = []
    paths.append(chart_individual_equity(results))
    paths.append(chart_portfolio_equity(port_equity, port_dd, results))
    paths.append(chart_monthly_heatmap(port_equity))
    paths.append(chart_rolling_sharpe(results, port_equity))
    paths.append(chart_correlation(results))
    paths.append(chart_drawdown_comparison(results, port_dd))
    paths.append(chart_trade_distribution(results))
    paths.append(chart_combined_overview(results, port_equity, port_dd))

    print(f"\nDone! {len(paths)} charts saved to {CHARTS_DIR}/")
    for p in paths:
        size = p.stat().st_size / 1024
        print(f"  {p.name}: {size:.0f} KB")


if __name__ == "__main__":
    main()
