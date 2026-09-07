#!/usr/bin/env python3
"""
Cost Sensitivity Analysis: Find the fee breakeven point for each strategy.

Tests performance at 0.5x, 1x, 2x, 3x, and 5x normal fees to determine
how much fee inflation the strategies can absorb before becoming unprofitable.

This answers: "If fees double or triple, do we still make money?"
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import SystemConfig, BacktestConfig, FeeConfig, ExecutionConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy


# Base fee config (Binance futures taker)
BASE_TAKER_FEE = 0.0005   # 0.05%
BASE_MAKER_FEE = 0.0002   # 0.02%

STRATEGIES = [
    {
        "name": "MACD",
        "label": "MACD ETH 4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
    },
    {
        "name": "ROC_Momentum",
        "label": "ROC_Momentum ETH 4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1,
                   "trend_ema": 50, "trend_filter": False},
    },
    {
        "name": "MACD",
        "label": "MACD BTC 4h",
        "pair": "BTC/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
    },
]

# Fee multipliers to test
FEE_MULTS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]


def run_backtest(df, strategy_name, params, pair, tf, taker_fee, maker_fee):
    """Run backtest with specific fee configuration."""
    bt_config = BacktestConfig(
        fees=FeeConfig(taker_fee=taker_fee, maker_fee=maker_fee),
        execution=ExecutionConfig(
            initial_capital=10000.0,
            stop_loss_atr_mult=3.0,
        ),
    )
    engine = BacktestEngine(bt_config)
    strategy = get_strategy(strategy_name)
    signals = strategy.generate_signals(df, params)
    result = engine.run(df, signals, strategy_name, params, pair, tf)
    return {
        "return": result.total_return * 100,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "max_dd": result.max_drawdown * 100,
        "trades": result.total_trades,
        "win_rate": result.win_rate * 100,
        "profit_factor": result.profit_factor,
        "total_fees": result.total_fees,
        "net_profit": result.net_profit,
    }


def find_breakeven(df, strategy_name, params, pair, tf, base_taker, base_maker,
                    max_mult=20.0, step=0.5):
    """Binary search for the fee multiplier where return crosses zero."""
    lo, hi = 0.0, max_mult

    for _ in range(20):  # max iterations
        mid = (lo + hi) / 2
        taker = base_taker * mid
        maker = base_maker * mid
        result = run_backtest(df, strategy_name, params, pair, tf, taker, maker)

        if result["return"] > 0:
            lo = mid
        else:
            hi = mid

        if hi - lo < 0.1:
            break

    return lo  # Last profitable multiplier


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)

    print("=" * 120)
    print("  COST SENSITIVITY ANALYSIS: Fee Breakeven Detection")
    print("=" * 120)

    print(f"\n  Base fees: Taker={BASE_TAKER_FEE*100:.3f}%  Maker={BASE_MAKER_FEE*100:.3f}%")
    print(f"  Fee multipliers tested: {FEE_MULTS}")
    print("  Stop loss: 3x ATR (active)")

    all_results = []

    for strat_def in STRATEGIES:
        print(f"\n{'=' * 120}")
        print(f"  {strat_def['label']} ({strat_def['name']} on {strat_def['pair']})")
        print(f"  Params: {strat_def['params']}")
        print(f"{'=' * 120}")

        strategy = get_strategy(strat_def["name"])
        df = loader.load(strat_def["pair"], strat_def["timeframe"])

        strat_results = []

        for mult in FEE_MULTS:
            taker = BASE_TAKER_FEE * mult
            maker = BASE_MAKER_FEE * mult
            result = run_backtest(
                df, strat_def["name"], strat_def["params"],
                strat_def["pair"], strat_def["timeframe"],
                taker, maker
            )

            strat_results.append({
                "mult": mult,
                "taker_fee": taker,
                "abs_fee_bps": taker * 2 * 10000,  # round-trip in bps
                **result,
            })

            rt_bps = taker * 2 * 10000  # round-trip in bps
            status = "+" if result["return"] > 0 else "x"
            print(f"  {status} {mult:>5.1f}x fees ({taker*100:.4f}% taker, "
                  f"{rt_bps:.1f} bps RT): "
                  f"Ret={result['return']:>+8.1f}%  "
                  f"Sharpe={result['sharpe']:>7.2f}  "
                  f"MaxDD={result['max_dd']:>6.2f}%  "
                  f"Fees=${result['total_fees']:>8.0f}  "
                  f"Net=${result['net_profit']:>+8.0f}  "
                  f"PF={result['profit_factor']:>5.2f}")

        # Find breakeven
        breakeven = find_breakeven(
            df, strat_def["name"], strat_def["params"],
            strat_def["pair"], strat_def["timeframe"],
            BASE_TAKER_FEE, BASE_MAKER_FEE
        )

        be_taker = BASE_TAKER_FEE * breakeven
        be_bps = be_taker * 2 * 10000

        print(f"\n  >>> BREAKEVEN: {breakeven:.1f}x normal fees "
              f"({be_taker*100:.4f}% taker, {be_bps:.0f} bps round-trip)")

        all_results.append({
            "strategy": strat_def["label"],
            "breakeven_mult": breakeven,
            "breakeven_taker_pct": be_taker * 100,
            "breakeven_bps": be_bps,
            "results": strat_results,
        })

    # ── Summary Table ──────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  SUMMARY: Fee Breakeven by Strategy")
    print(f"{'=' * 120}")

    print(f"\n  {'Strategy':<25} {'Breakeven':>12} {'Taker Fee':>12} {'Round-Trip':>12} {'Safety Margin':>15}")
    print(f"  {'-' * 80}")

    for r in all_results:
        # Safety margin: how many times current fees before breakeven
        safety = r["breakeven_mult"]
        print(f"  {r['strategy']:<25} {r['breakeven_mult']:>10.1f}x  "
              f"{r['breakeven_taker_pct']:>10.4f}%  "
              f"{r['breakeven_bps']:>10.0f} bps  "
              f"{safety:>13.1f}x current")

    # ── Fee Impact at Common Tiers ─────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  PERFORMANCE AT COMMON FEE TIERS")
    print(f"{'=' * 120}")

    # Common fee scenarios
    fee_tiers = [
        ("VIP0 Binance (0.04%)", 0.0004),
        ("VIP1 Binance (0.035%)", 0.00035),
        ("VIP3 Binance (0.025%)", 0.00025),
        ("Market Maker (0.02%)", 0.0002),
        ("High-fee exchange (0.1%)", 0.001),
        ("DEX typical (0.3%)", 0.003),
        ("DEX high gas (0.5%)", 0.005),
    ]

    for tier_name, tier_fee in fee_tiers:
        mult = tier_fee / BASE_TAKER_FEE
        print(f"\n  {tier_name} ({mult:.1f}x base):")
        print(f"    {'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'PF':>8} {'Verdict':>10}")
        print(f"    {'-' * 70}")

        for strat_def in STRATEGIES:
            result = run_backtest(
                loader.load(strat_def["pair"], strat_def["timeframe"]),
                strat_def["name"], strat_def["params"],
                strat_def["pair"], strat_def["timeframe"],
                tier_fee, tier_fee * 0.4  # maker = 40% of taker
            )
            verdict = "PROFITABLE" if result["return"] > 0 else "LOSING"
            print(f"    {strat_def['label']:<25} {result['return']:>+9.1f}% "
                  f"{result['sharpe']:>8.2f} {result['max_dd']:>7.2f}% "
                  f"{result['profit_factor']:>8.2f} {verdict:>10}")

    # ── Fee cost breakdown at 1x and breakeven ─────────────────────
    print(f"\n\n{'=' * 120}")
    print("  FEE COST ANALYSIS")
    print(f"{'=' * 120}")

    for strat_def in STRATEGIES:
        df = loader.load(strat_def["pair"], strat_def["timeframe"])
        # At base fees
        r_base = run_backtest(
            df, strat_def["name"], strat_def["params"],
            strat_def["pair"], strat_def["timeframe"],
            BASE_TAKER_FEE, BASE_MAKER_FEE
        )

        print(f"\n  {strat_def['label']}:")
        print(f"    Total fees paid:     ${r_base['total_fees']:>10.2f}")
        print(f"    Net profit:          ${r_base['net_profit']:>10.2f}")
        print(f"    Fee % of gross:      {r_base['total_fees'] / max(abs(r_base['net_profit'] + r_base['total_fees']), 1) * 100:>9.1f}%")
        print(f"    Trades:              {r_base['trades']:>10d}")
        print(f"    Avg fee per trade:   ${r_base['total_fees'] / max(r_base['trades'], 1):>10.2f}")

    # Save
    out_path = Path("data/results/cost_sensitivity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("=" * 120)


if __name__ == "__main__":
    main()
