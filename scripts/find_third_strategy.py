#!/usr/bin/env python3
"""
Find the Best Third Strategy to Replace BB_Squeeze.

Tests RSI_Reversion, ZScore_Reversion, Bollinger_Reversion with
focused param grids on ETH 4h and BTC 4h, then validates the
winner with walk-forward and Monte Carlo analysis.
"""

import sys
import json
import time
import itertools
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from trading_system.validation.walk_forward import WalkForwardAnalyzer
from trading_system.validation.monte_carlo import MonteCarloAnalyzer
from trading_system.optimization.scoring import (
    _normalize_sharpe, _normalize_sortino,
    _normalize_calmar, _normalize_profit_factor,
)

# ── Focused param grids (tight, informed by sensitivity) ──────────

FOCUSED_GRIDS = {
    "RSI_Reversion": {
        "grid": {
            "rsi_period": [7, 10, 14],
            "entry_oversold": [25, 30, 35],
            "entry_overbought": [65, 70, 75],
            "exit_neutral_low": [40, 45, 50],
            "exit_neutral_high": [50, 55, 60],
            "use_bb_filter": [False, True],
            "bb_period": [15, 20],
            "bb_std": [1.5, 2.0, 2.5],
        },
    },
    "ZScore_Reversion": {
        "grid": {
            "lookback": [15, 20, 30, 40],
            "entry_threshold": [1.5, 2.0, 2.5, 3.0],
            "exit_threshold": [0.0, 0.25, 0.5],
            "use_sma_baseline": [True, False],
        },
    },
    "Bollinger_Reversion": {
        "grid": {
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_filter": [False, True],
            "rsi_period": [10, 14],
            "rsi_oversold": [25, 30],
            "rsi_overbought": [70, 75],
            "exit_at_middle": [True, False],
        },
    },
}

PAIRS = [
    ("ETH/USDT:USDT", "4h"),
    ("BTC/USDT:USDT", "4h"),
]


def generate_combinations(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def compute_composite(r):
    sharpe = r.get("sharpe", 0) or 0
    sortino = r.get("sortino", 0) or 0
    calmar = r.get("calmar", 0) or 0
    pf = r.get("profit_factor", 0) or 0
    trades = r.get("total_trades", 0) or 0
    sharpe_n = _normalize_sharpe(sharpe)
    sortino_n = _normalize_sortino(sortino)
    calmar_n = _normalize_calmar(calmar)
    pf_n = _normalize_profit_factor(pf)
    penalty = 0.0
    if sharpe > 4:
        penalty += 0.2
    elif sharpe > 3:
        penalty += 0.1
    if trades < 20:
        penalty += 0.2
    elif trades < 50:
        penalty += 0.1
    trade_penalty = (30 - trades) / 30 * 0.5 if trades < 30 else 0.0
    score = (0.40 * sharpe_n + 0.20 * sortino_n + 0.15 * calmar_n
             + 0.15 * pf_n + 0.10 * (1 - penalty) - trade_penalty)
    return max(0.0, min(1.0, score))


def run_grid_search():
    """Phase 1: Grid search for all 3 strategies across pairs."""
    print("=" * 70)
    print("  PHASE 1: Grid Search for Third Strategy Candidates")
    print("=" * 70)

    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    engine = BacktestEngine(cfg.backtest)

    results = []  # (strat, pair, tf, params, metrics)

    for strat_name, spec in FOCUSED_GRIDS.items():
        grid = spec["grid"]
        combos = generate_combinations(grid)
        strategy = get_strategy(strat_name)

        for pair, tf in PAIRS:
            df = loader.load(pair, tf)
            funding = loader.load_funding_rates(pair)
            if df is None or df.empty:
                print(f"  SKIP {pair} {tf}: no data")
                continue

            print(f"\n  {strat_name} on {pair} {tf} ({len(df):,} candles, {len(combos)} combos)")
            t0 = time.time()
            ok = 0
            best = None

            for combo in combos:
                try:
                    signals = strategy.generate_signals(df, combo)
                    result = engine.run(df, signals, strat_name, combo, pair, tf, funding)

                    r = {
                        "sharpe": result.sharpe,
                        "sortino": result.sortino,
                        "calmar": result.calmar,
                        "total_return": result.total_return,
                        "max_drawdown": result.max_drawdown,
                        "total_trades": result.total_trades,
                        "win_rate": result.win_rate,
                        "profit_factor": result.profit_factor,
                    }
                    score = compute_composite(r)

                    if best is None or score > best[1]:
                        best = (combo, score, r)

                    ok += 1
                except Exception as e:
                    pass

            elapsed = time.time() - t0
            if best:
                combo, score, r = best
                print(f"    Best: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:.1f}% "
                      f"MaxDD={r['max_drawdown']*100:.2f}% Trades={r['total_trades']} "
                      f"WR={r['win_rate']:.0%} PF={r['profit_factor']:.2f} Score={score:.4f}")
                print(f"    Params: {json.dumps(combo)}")
                print(f"    Time: {elapsed:.1f}s ({ok}/{len(combos)} OK)")
                results.append((strat_name, pair, tf, combo, r, score))
            else:
                print(f"    No valid results in {elapsed:.1f}s")

    return results


def run_validation(strat_name, pair, tf, params):
    """Phase 2: Walk-forward + Monte Carlo on a single config."""
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    bt_config = cfg.backtest
    engine = BacktestEngine(bt_config)
    wf_analyzer = WalkForwardAnalyzer(bt_config)
    mc_analyzer = MonteCarloAnalyzer(n_simulations=5000, seed=42)

    df = loader.load(pair, tf)
    funding = loader.load_funding_rates(pair)
    strategy = get_strategy(strat_name)

    # Param grid for walk-forward (focused subset)
    grid = FOCUSED_GRIDS[strat_name]["grid"]

    print(f"\n  --- Walk-Forward: {strat_name} {pair} {tf} ---")
    t0 = time.time()
    try:
        wf = wf_analyzer.run_walk_forward(
            df, strat_name, grid,
            n_windows=5, train_pct=0.7,
            pair=pair, timeframe=tf,
            funding_rates=funding,
        )
        agg = wf["aggregate"]
        print(f"  WF: Avg OOS Sharpe={agg['avg_sharpe']:.2f}  Min={agg['min_sharpe']:.2f}  "
              f"Profitable={agg['pct_profitable_windows']:.0%}  "
              f"AvgRet={agg['avg_return']*100:.2f}%  AvgMaxDD={agg['avg_max_drawdown']*100:.2f}%")
        for w in wf["windows"]:
            print(f"    W{w['window']}: Sharpe={w['sharpe']:.2f} Ret={w['total_return']*100:.2f}% Trades={w['trades']}")
    except Exception as e:
        print(f"  WF ERROR: {e}")
        import traceback; traceback.print_exc()
        agg = {}

    print(f"  Time: {time.time()-t0:.1f}s")

    # Monte Carlo
    print(f"\n  --- Monte Carlo: {strat_name} {pair} {tf} ---")
    try:
        signals = strategy.generate_signals(df, params)
        result = engine.run(df, signals, strat_name, params, pair, tf, funding)
        mc = mc_analyzer.run_trade_shuffling(result.trades, initial_capital=10000.0)
        print(f"  MC: P(Loss)={mc['prob_loss']:.1%}  P(Ruin)={mc['prob_ruin']:.1%}  "
              f"E[Ret]={mc['expected_return']:.1%}  E[MaxDD]={mc['expected_max_drawdown']:.2%}")
        ci = mc.get("final_equity_95_ci", [0, 0, 0])
        print(f"      95% CI Equity: [{ci[0]:.0f}, {ci[1]:.0f}, {ci[2]:.0f}]")
    except Exception as e:
        print(f"  MC ERROR: {e}")
        import traceback; traceback.print_exc()
        mc = {}
        result = None

    return agg, mc, result


def main():
    t0 = time.time()

    # Phase 1: Grid search
    results = run_grid_search()

    if not results:
        print("No valid results!")
        return

    # Rank by composite score
    results.sort(key=lambda x: x[5], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"  GRID SEARCH RESULTS — Ranked by Composite Score")
    print(f"{'=' * 70}")
    print(f"  {'#':>3s} {'Strategy':<20s} {'Pair':<5s} {'TF':<4s} {'Score':>6s} {'Sharpe':>7s} {'Ret':>8s} {'MaxDD':>7s} {'Trades':>7s} {'WR':>5s} {'PF':>5s}")
    print(f"  {'-' * 95}")
    for i, (sn, pair, tf, params, r, score) in enumerate(results[:20]):
        print(f"  {i+1:>3d} {sn:<20s} {pair.split('/')[0]:<5s} {tf:<4s} "
              f"{score:>6.4f} {r['sharpe']:>7.2f} {r['total_return']*100:>7.1f}% "
              f"{r['max_drawdown']*100:>6.2f}% {r['total_trades']:>7d} {r['win_rate']:>4.0%} {r['profit_factor']:>5.2f}")
        if i < 5:
            print(f"       params: {json.dumps(params)}")

    # Pick the best per strategy
    seen = {}
    for sn, pair, tf, params, r, score in results:
        if sn not in seen:
            seen[sn] = (sn, pair, tf, params, r, score)

    print(f"\n  Best per strategy:")
    for sn in seen:
        s, p, tf, params, r, score = seen[sn]
        print(f"  {s:<20s} {p.split('/')[0]:<5s} {tf} Sharpe={r['sharpe']:.2f} "
              f"Ret={r['total_return']*100:.1f}% DD={r['max_drawdown']*100:.2f}% "
              f"Trades={r['total_trades']} Score={score:.4f}")

    # Phase 2: Validate top 3 candidates with walk-forward + MC
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: Walk-Forward + Monte Carlo Validation")
    print(f"{'=' * 70}")

    top3 = results[:3]
    validation_results = []

    for sn, pair, tf, params, r, score in top3:
        print(f"\n{'=' * 70}")
        print(f"  Validating: {sn} on {pair} {tf}")
        print(f"  IS Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:.1f}%  Trades={r['total_trades']}")
        print(f"  Params: {json.dumps(params)}")

        agg, mc, bt_result = run_validation(sn, pair, tf, params)
        validation_results.append({
            "strategy": sn, "pair": pair, "timeframe": tf,
            "params": params,
            "in_sample": r,
            "wf_aggregate": agg,
            "mc": mc,
        })

    # Phase 3: Comparison with BB_Squeeze
    print(f"\n{'=' * 70}")
    print(f"  PHASE 3: Final Comparison (Including BB_Squeeze Baseline)")
    print(f"{'=' * 70}")

    # Add BB_Squeeze baseline
    bb_params = {'bb_period': 15, 'bb_std': 2.0, 'kc_period': 15,
                 'kc_atr_mult': 1.0, 'squeeze_lookback': 6, 'momentum_period': 8}
    print(f"\n  BB_Squeeze baseline: WF Avg Sharpe=1.15, 40% profitable windows")

    print(f"\n  {'Strategy':<20s} {'IS Sharpe':>9s} {'WF OOS':>8s} {'WF Min':>8s} {'WF %Profit':>10s} {'MC P(Loss)':>10s} {'Verdict'}")
    print(f"  {'-' * 85}")

    for vr in validation_results:
        wf_agg = vr["wf_aggregate"]
        mc = vr["mc"]
        is_sharpe = vr["in_sample"]["sharpe"]
        wf_avg = wf_agg.get("avg_sharpe", 0)
        wf_min = wf_agg.get("min_sharpe", 0)
        wf_pct = wf_agg.get("pct_profitable_windows", 0)
        mc_loss = mc.get("prob_loss", 0.5)

        # Verdict
        if wf_pct >= 0.8 and wf_min > 0 and mc_loss < 0.05:
            verdict = "✅ BEST"
        elif wf_pct >= 0.6 and mc_loss < 0.10:
            verdict = "✅ GOOD"
        elif wf_pct >= 0.4:
            verdict = "⚠️  MARGINAL"
        else:
            verdict = "❌ WEAK"

        print(f"  {vr['strategy']:<20s} {is_sharpe:>9.2f} {wf_avg:>8.2f} {wf_min:>8.2f} "
              f"{wf_pct:>9.0%} {mc_loss:>9.1%} {verdict}")

    # Save results
    import json as json_mod
    output_path = Path("data/results/third_strategy_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json_mod.dump(validation_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
