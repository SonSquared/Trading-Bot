"""
Weekly Chart Generator

Reads paper trading data and generates professional charts:
  1. Equity curve with drawdown overlay
  2. Monthly returns heatmap
  3. Trade P&L distribution

Sends all charts via Telegram as images.

Usage:
  python scripts/weekly_chart.py

Designed to run via GitHub Actions weekly cron (see weekly_chart.yml).
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

import subprocess

# --- Config ---
TRADE_LOG = Path("data/results/paper_trades.jsonl")
STATE_FILE = Path("data/results/paper_state.json")
CHART_DIR = Path("data/results/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)
INITIAL_CAPITAL = 10000.0


# --- Telegram ---
def load_telegram_config():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        return {"bot_token": token, "chat_id": chat_id}
    try:
        import yaml
        cfg_path = Path("configs/bot_live.yaml")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            tg = cfg.get("bot", {}).get("telegram", {})
            if tg.get("bot_token"):
                return {"bot_token": tg["bot_token"], "chat_id": str(tg.get("chat_id", ""))}
    except Exception:
        pass
    return {}


def send_photo(token: str, chat_id: str, photo_path: str, caption: str = "") -> bool:
    result = subprocess.run(
        ["curl", "-s", "-m", "30", "-X", "POST",
         f"https://api.telegram.org/bot{token}/sendPhoto",
         "-F", f"chat_id={chat_id}",
         "-F", f"photo=@{photo_path}",
         "-F", f"caption={caption}",
         "-F", "parse_mode=HTML"],
        capture_output=True, text=True, timeout=35,
    )
    if result.stdout:
        data = json.loads(result.stdout)
        return data.get("ok", False)
    return False


def send_message(token: str, chat_id: str, text: str) -> bool:
    result = subprocess.run(
        ["curl", "-s", "-m", "15", "-X", "POST",
         f"https://api.telegram.org/bot{token}/sendMessage",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"})],
        capture_output=True, text=True, timeout=20,
    )
    if result.stdout:
        data = json.loads(result.stdout)
        return data.get("ok", False)
    return False


# --- Data Loading ---
def load_trades() -> list:
    trades = []
    if TRADE_LOG.exists():
        with open(TRADE_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return trades


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": INITIAL_CAPITAL, "positions": {}, "total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}


def build_equity_curve(trades: list) -> list:
    """Build equity curve from trades."""
    equity = INITIAL_CAPITAL
    curve = [{"timestamp": trades[0]["timestamp"] if trades else datetime.now(timezone.utc).isoformat(), "equity": equity}]

    for t in trades:
        if t.get("action") == "CLOSE":
            equity += t.get("pnl_usd", 0)
            curve.append({"timestamp": t["timestamp"], "equity": equity})
        elif t.get("action", "").startswith("OPEN"):
            # Track position opening
            pass

    # Add current state
    state = load_state()
    curve.append({"timestamp": datetime.now(timezone.utc).isoformat(), "equity": state.get("cash", INITIAL_CAPITAL)})

    return curve


# --- Chart Generation ---
def generate_equity_chart(trades: list, output_path: str) -> str:
    """Generate equity curve chart with drawdown overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.style.use("dark_background")

    curve = build_equity_curve(trades)
    if len(curve) < 2:
        return ""

    dates = [datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00")) for c in curve]
    equities = [c["equity"] for c in curve]

    # Calculate drawdown
    peak = equities[0]
    drawdowns = []
    for e in equities:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100
        drawdowns.append(-dd)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[3, 1], sharex=True)
    fig.patch.set_facecolor("#0d1117")

    # Equity curve
    ax1.set_facecolor("#0d1117")
    ax1.fill_between(dates, INITIAL_CAPITAL, equities, alpha=0.3, color="#00ff88")
    ax1.plot(dates, equities, color="#00ff88", linewidth=2, label="Equity")

    # Benchmark
    ax1.axhline(y=INITIAL_CAPITAL, color="#555555", linestyle="--", alpha=0.5, label="Initial Capital")

    ax1.set_title("Paper Trading Equity Curve", color="white", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Equity ($)", color="white", fontsize=11)
    ax1.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#333333")
    ax1.grid(True, alpha=0.15)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Drawdown
    ax2.set_facecolor("#0d1117")
    ax2.fill_between(dates, drawdowns, 0, alpha=0.4, color="#ff4444")
    ax2.plot(dates, drawdowns, color="#ff4444", linewidth=1)
    ax2.set_ylabel("Drawdown (%)", color="white", fontsize=11)
    ax2.set_xlabel("Date", color="white", fontsize=11)
    ax2.grid(True, alpha=0.15)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


def generate_trade_dist_chart(trades: list, output_path: str) -> str:
    """Generate trade P&L distribution chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    close_trades = [t for t in trades if t.get("action") == "CLOSE"]
    if not close_trades:
        return ""

    pnls = [t.get("pnl_usd", 0) for t in close_trades]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Histogram
    wins = [p for p in pnls if p >= 0]
    losses = [p for p in pnls if p < 0]

    if wins:
        ax.hist(wins, bins=20, color="#00ff88", alpha=0.7, label=f"Wins ({len(wins)})")
    if losses:
        ax.hist(losses, bins=20, color="#ff4444", alpha=0.7, label=f"Losses ({len(losses)})")

    ax.axvline(x=0, color="#ffffff", linestyle="--", alpha=0.5)
    ax.axvline(x=np.mean(pnls), color="#ffaa00", linestyle="-", linewidth=2, alpha=0.8, label=f"Avg: ${np.mean(pnls):+.2f}")

    ax.set_title("Trade P&L Distribution", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("P&L ($)", color="white", fontsize=11)
    ax.set_ylabel("Count", color="white", fontsize=11)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333333")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


def generate_summary_card(trades: list, output_path: str) -> str:
    """Generate a summary card image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = load_state()
    equity = state.get("cash", INITIAL_CAPITAL)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wr = state["wins"] / state["total_trades"] * 100 if state["total_trades"] > 0 else 0

    close_trades = [t for t in trades if t.get("action") == "CLOSE"]
    pnls = [t.get("pnl_usd", 0) for t in close_trades] if close_trades else [0]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.92, "PAPER TRADING WEEKLY REPORT", fontsize=16, fontweight="bold",
            color="white", ha="center", transform=ax.transAxes)

    # Metrics
    metrics = [
        (0.15, 0.72, f"${equity:,.2f}", "#00ff88", "Equity"),
        (0.50, 0.72, f"{total_return:+.2f}%", "#00ff88" if total_return >= 0 else "#ff4444", "Return"),
        (0.85, 0.72, f"{state['total_trades']}", "#ffffff", "Trades"),
        (0.15, 0.42, f"{wr:.1f}%", "#ffaa00" if wr > 50 else "#ff4444", "Win Rate"),
        (0.50, 0.42, f"${state['total_pnl']:+.2f}", "#00ff88" if state["total_pnl"] >= 0 else "#ff4444", "Realized P&L"),
        (0.85, 0.42, f"{state.get('max_drawdown', 0)*100:.2f}%", "#ff4444", "Max DD"),
    ]

    for x, y, value, color, label in metrics:
        ax.text(x, y, value, fontsize=18, fontweight="bold", color=color, ha="center", transform=ax.transAxes)
        ax.text(x, y - 0.12, label, fontsize=10, color="#888888", ha="center", transform=ax.transAxes)

    # Timestamp
    ax.text(0.5, 0.08, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            fontsize=9, color="#555555", ha="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# --- Main ---
def main():
    print("=" * 50)
    print("WEEKLY CHART GENERATOR")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 50)

    config = load_telegram_config()
    if not config.get("bot_token"):
        print("ERROR: Telegram not configured. Set TELEGRAM_BOT_TOKEN env var.")
        sys.exit(1)

    token = config["bot_token"]
    chat_id = config["chat_id"]

    # Load data
    trades = load_trades()
    state = load_state()
    print(f"Trades: {len(trades)}")
    print(f"Equity: ${state.get('cash', INITIAL_CAPITAL):,.2f}")

    if not trades:
        send_message(token, chat_id, "📊 Weekly report: No trades this week. All strategies FLAT.")
        print("No trades — sent notification only.")
        return

    # Generate charts
    print("\nGenerating charts...")
    equity_chart = generate_equity_chart(trades, str(CHART_DIR / "weekly_equity.png"))
    trade_chart = generate_trade_dist_chart(trades, str(CHART_DIR / "weekly_trades.png"))
    summary_card = generate_summary_card(trades, str(CHART_DIR / "weekly_summary.png"))

    # Send via Telegram
    print("\nSending to Telegram...")
    equity = state.get("cash", INITIAL_CAPITAL)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if summary_card:
        send_photo(token, chat_id, summary_card, f"📊 Weekly Report — Equity: ${equity:,.2f} ({total_return:+.2f}%)")
        print("  Sent summary card")

    if equity_chart:
        send_photo(token, chat_id, equity_chart, "📈 Equity curve with drawdown")
        print("  Sent equity chart")

    if trade_chart:
        send_photo(token, chat_id, trade_chart, "📊 Trade P&L distribution")
        print("  Sent trade distribution")

    print("\nDone.")


if __name__ == "__main__":
    main()
