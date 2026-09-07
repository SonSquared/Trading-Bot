#!/usr/bin/env python3
"""
Paper Trading Runner.

Initializes the Portfolio Trading Bot in paper mode and runs
signal generation cycles. Can also run a single cycle for testing.

Usage:
  python scripts/run_paper_trading.py              # Single cycle
  python scripts/run_paper_trading.py --continuous  # Continuous mode
  python scripts/run_paper_trading.py --status      # Show bot status
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import SystemConfig
from trading_system.bot.portfolio_bot import PortfolioTradingBot


def run_single_cycle(bot: PortfolioTradingBot) -> dict:
    """Run a single signal generation cycle and display results."""
    print("=" * 70)
    print("  PORTFOLIO BOT - Single Signal Cycle")
    print("=" * 70)

    results = bot.run_once()

    for pair, agg in results.items():
        print(f"\n  {pair}:")
        print(f"    Weighted Score: {agg['weighted_score']:+.3f}")
        print(f"    Direction:      {'LONG' if agg['direction'] > 0 else 'SHORT' if agg['direction'] < 0 else 'FLAT'}")
        print(f"    Confidence:     {agg['confidence']:.1%}")
        print("    Strategy Signals:")
        for label, signal in agg["signals"].items():
            direction = "LONG" if signal > 0 else "SHORT" if signal < 0 else "FLAT"
            print(f"      {label:<25s} => {direction}")

    return results


def run_continuous(bot: PortfolioTradingBot):
    """Run the bot in continuous mode with periodic signal printing."""
    print("=" * 70)
    print("  PORTFOLIO BOT - Continuous Paper Trading")
    print("=" * 70)
    print(f"  Checking every {bot.bot_config.check_interval_seconds}s")
    print("  Press Ctrl+C to stop\n")

    cycle = 0
    try:
        while bot._running:
            cycle += 1
            print(f"\n  --- Cycle {cycle} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
            results = bot.run_once()

            for pair, agg in results.items():
                direction = "LONG" if agg["direction"] > 0 else "SHORT" if agg["direction"] < 0 else "FLAT"
                print(f"  {pair}: {direction} (score={agg['weighted_score']:+.3f}, conf={agg['confidence']:.0%})")
                for label, signal in agg["signals"].items():
                    sig = "+" if signal > 0 else "-" if signal < 0 else "0"
                    print(f"    {label}: {sig}", end="")
                print()

            time.sleep(bot.bot_config.check_interval_seconds)

    except KeyboardInterrupt:
        print("\n  Stopped by user")


def show_status(bot: PortfolioTradingBot):
    """Display current bot status."""
    status = bot.get_status()
    print("=" * 70)
    print("  PORTFOLIO BOT STATUS")
    print("=" * 70)
    print(f"  Mode:       {status['mode']}")
    print(f"  Running:    {status['running']}")
    print(f"  Strategies: {status['n_strategies']}")

    print("\n  Strategy Allocation:")
    for s in status["strategies"]:
        last = s.get("last_signal", "N/A")
        if isinstance(last, int):
            last = "LONG" if last > 0 else "SHORT" if last < 0 else "FLAT"
        print(f"    {s['label']:<30s} weight={s['weight']:.1%}  TF={s['timeframe']}  last={last}")

    print("\n  Risk Status:")
    risk = status["risk"]
    print(f"    Emergency Stop:     {risk['emergency_stop']}")
    print(f"    Daily P&L:          ${risk['daily_pnl']:.2f}")
    print(f"    Current Drawdown:   {risk['current_drawdown']:.2%}")
    print(f"    Consecutive Losses: {risk['consecutive_losses']}")
    print(f"    Total Positions:    {risk['total_positions']}")
    print(f"    Daily Trades:       {risk['daily_trades']}")

    print("\n  Last Signals:")
    for pair, sig in status["last_signals"].items():
        direction = "LONG" if sig["direction"] > 0 else "SHORT" if sig["direction"] < 0 else "FLAT"
        print(f"    {pair}: {direction} (score={sig['score']:+.3f}) at {sig['timestamp']}")
        for label, signal in sig.get("signals", {}).items():
            s = "LONG" if signal > 0 else "SHORT" if signal < 0 else "FLAT"
            print(f"      {label}: {s}")

    print(f"\n  Total Trades: {status['total_trades']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Portfolio Trading Bot")
    parser.add_argument("--continuous", action="store_true",
                        help="Run in continuous mode")
    parser.add_argument("--status", action="store_true",
                        help="Show bot status")
    parser.add_argument("--mode", choices=["paper", "dry_run", "live", "backtest"],
                        default=None, help="Override bot mode")
    args = parser.parse_args()

    # Load config
    cfg = SystemConfig.default()

    # Override mode if specified
    if args.mode:
        cfg.bot.mode = args.mode

    # Force paper mode for safety
    if cfg.bot.mode not in ("paper", "dry_run"):
        print("  WARNING: Forcing paper mode for safety. Use --mode paper to confirm.")
        cfg.bot.mode = "paper"

    # Initialize bot
    bot = PortfolioTradingBot(cfg)

    if args.status:
        show_status(bot)
    elif args.continuous:
        bot._running = True
        run_continuous(bot)
    else:
        # Single cycle
        results = run_single_cycle(bot)

        # Also save results
        output_path = Path("data/results/paper_trading_snapshot.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

        # Show bot status
        print()
        show_status(bot)


if __name__ == "__main__":
    main()
