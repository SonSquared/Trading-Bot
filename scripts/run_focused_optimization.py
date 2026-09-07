#!/usr/bin/env python3
"""
Focused Re-optimization with Tight Parameter Constraints.

Based on sensitivity analysis:
- ROC_Momentum: lock smooth_period=1, focus roc_period in 3-9
- MACD: constrain slow >= 10, focus fast in 4-12, signal in 2-7
"""

import sys
import json
import sqlite3
import time
import itertools
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from trading_system.optimization.scoring import (
    _normalize_sharpe, _normalize_sortino,
    _normalize_calmar, _normalize_profit_factor,
)

DB_PATH = Path("data/results/focused_experiments.db")


def compute_composite(r):
    sharpe = r.get("sharpe", 0) or 0
    sortino = r.get("sortino", 0) or 0
    calmar = r.get("calmar", 0) or 0
    pf = r.get("profit_factor", 0) or 0
    trades = r.get("total_trades", 0) or 0
    wr = r.get("win_rate", 0) or 0
    sharpe_n = _normalize_sharpe(sharpe)
    sortino_n = _normalize_sortino(sortino)
    calmar_n = _normalize_calmar(calmar)
    pf_n = _normalize_profit_factor(pf)
    penalty = 0.0
    if sharpe > 4: penalty += 0.2
    elif sharpe > 3: penalty += 0.1
    if trades < 20: penalty += 0.2
    elif trades < 50: penalty += 0.1
    trade_penalty = (30 - trades) / 30 * 0.5 if trades < 30 else 0.0
    score = (0.40 * sharpe_n + 0.20 * sortino_n + 0.15 * calmar_n
             + 0.15 * pf_n + 0.10 * (1 - penalty) - trade_penalty)
    return max(0.0, min(1.0, score))


FOCUSED_GRIDS = {
    "ROC_Momentum": {
        "description": "Lock smooth_period=1, finer roc_period",
        "grid": {
            "roc_period": [3, 4, 5, 6, 7, 8, 9],
            "roc_threshold": [-2, -1, 0, 1],
            "smooth_period": [1],
            "trend_filter": [False, True],
            "trend_ema": [50, 75, 100, 150],
        },
    },
    "MACD": {
        "description": "Constrain slow>=10, finer fast/signal",
        "grid": {
            "fast": [4, 5, 6, 7, 8, 9, 10, 12],
            "slow": [10, 12, 14, 16, 18, 20, 24, 28, 32],
            "signal": [2, 3, 4, 5, 6, 7],
            "use_histogram": [False],
        },
    },
}

PAIRS = [
    ("ETH/USDT:USDT", "1h"), ("ETH/USDT:USDT", "4h"), ("ETH/USDT:USDT", "30m"),
    ("BTC/USDT:USDT", "1h"), ("BTC/USDT:USDT", "4h"), ("BTC/USDT:USDT", "30m"),
]


def generate_combinations(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main():
    t0_total = time.time()

    # Init DB
    conn = sqlite3.connect(str(DB_PATH), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            strategy_name TEXT, pair TEXT, timeframe TEXT,
            parameters TEXT, sharpe REAL, sortino REAL, calmar REAL,
            total_return REAL, max_drawdown REAL, total_trades INTEGER,
            win_rate REAL, profit_factor REAL, net_profit REAL, total_fees REAL,
            composite_score REAL, success INTEGER DEFAULT 0,
            error TEXT, timestamp TEXT
        )
    """)
    conn.commit()

    # Check existing
    cur.execute("SELECT COUNT(*) FROM experiments WHERE success = 1")
    existing_ok = cur.fetchone()[0]
    cur.execute("SELECT experiment_id FROM experiments WHERE success = 1")
    done_ids = {r[0] for r in cur.fetchall()}
    if existing_ok > 0:
        print(f"Resuming: {existing_ok} experiments already completed")

    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    engine = BacktestEngine(cfg.backtest)

    total_ok = 0
    total_fail = 0

    for strat_name, spec in FOCUSED_GRIDS.items():
        grid = spec["grid"]
        combos = generate_combinations(grid)
        n_combos = len(combos)
        print(f"\n{'=' * 70}")
        print(f"  {strat_name}: {spec['description']}")
        print(f"  Grid: {n_combos} combos x {len(PAIRS)} pair/tf = {n_combos * len(PAIRS)} experiments")
        print(f"{'=' * 70}")

        strategy = get_strategy(strat_name)

        for pair, tf in PAIRS:
            df = loader.load(pair, tf)
            if df is None or df.empty:
                print(f"  SKIP {pair} {tf}: no data")
                continue

            funding = loader.load_funding_rates(pair)
            print(f"\n  --- {pair} {tf} ({len(df):,} candles) ---")
            t0 = time.time()
            ok = 0
            fail = 0
            n_done = 0
            best_sh = 0
            best_params = None

            for combo in combos:
                params_str = json.dumps(combo, sort_keys=True)
                eid = hashlib.md5(f"{strat_name}_{pair}_{tf}_{params_str}".encode()).hexdigest()

                # Skip already done
                if eid in done_ids:
                    ok += 1
                    total_ok += 1
                    continue

                try:
                    signals = strategy.generate_signals(df, combo)
                    result = engine.run(df, signals, strat_name, combo, pair, tf, funding)

                    r = {
                        "sharpe": result.sharpe, "sortino": result.sortino,
                        "calmar": result.calmar, "total_return": result.total_return,
                        "max_drawdown": result.max_drawdown, "total_trades": result.total_trades,
                        "win_rate": result.win_rate, "profit_factor": result.profit_factor,
                        "net_profit": result.net_profit, "total_fees": result.total_fees,
                    }
                    score = compute_composite(r)

                    cur.execute("""
                        INSERT OR REPLACE INTO experiments
                        (experiment_id, strategy_name, pair, timeframe, parameters,
                         sharpe, sortino, calmar, total_return, max_drawdown,
                         total_trades, win_rate, profit_factor, net_profit, total_fees,
                         composite_score, success, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """, (eid, strat_name, pair, tf, params_str,
                          r["sharpe"], r["sortino"], r["calmar"],
                          r["total_return"], r["max_drawdown"],
                          r["total_trades"], r["win_rate"], r["profit_factor"],
                          r["net_profit"], r["total_fees"], score,
                          time.strftime("%Y-%m-%d %H:%M:%S")))

                    ok += 1
                    total_ok += 1
                    n_done += 1

                    if r["sharpe"] > best_sh:
                        best_sh = r["sharpe"]
                        best_params = combo

                    # Commit every 50 experiments to avoid data loss on timeout
                    if n_done % 50 == 0:
                        conn.commit()

                except Exception as e:
                    try:
                        cur.execute("""
                            INSERT OR REPLACE INTO experiments
                            (experiment_id, strategy_name, pair, timeframe, parameters,
                             success, error, timestamp)
                            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                        """, (eid, strat_name, pair, tf, params_str,
                              str(e)[:500], time.strftime("%Y-%m-%d %H:%M:%S")))
                        fail += 1
                        total_fail += 1
                    except sqlite3.OperationalError:
                        # DB locked, skip this error record
                        fail += 1
                        total_fail += 1

            conn.commit()
            elapsed = time.time() - t0
            n_total = ok + fail
            speed = n_total / elapsed if elapsed > 0 else 0
            print(f"    Done: {ok}/{n_total} OK, {fail} fail, {elapsed:.1f}s ({speed:.0f}/s)")
            if best_params:
                print(f"    Best: Sharpe={best_sh:.2f} params={json.dumps(best_params)}")

    conn.close()
    elapsed_total = time.time() - t0_total

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FOCUSED RE-OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total: {total_ok} OK, {total_fail} fail")
    print(f"  Time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    # Top results
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    cur = conn.cursor()
    cur.execute("""
        SELECT strategy_name, pair, timeframe, parameters,
               sharpe, total_return, max_drawdown, total_trades,
               win_rate, profit_factor, composite_score
        FROM experiments WHERE success = 1 AND total_trades >= 20
        ORDER BY composite_score DESC LIMIT 20
    """)
    rows = cur.fetchall()

    print("\n  Top 20 by composite score:")
    print(f"  {'#':>3s} {'Strategy':<16s} {'Pair':<5s} {'TF':<4s} {'Score':>6s} {'Sharpe':>7s} {'Ret':>8s} {'MaxDD':>7s} {'Trades':>7s} {'PF':>5s}")
    print(f"  {'-' * 85}")
    for i, r in enumerate(rows):
        sn, pair, tf, params, sh, ret, dd, tr, wr, pf, sc = r
        print(f"  {i+1:>3d} {sn:<16s} {pair.split('/')[0]:<5s} {tf:<4s} "
              f"{sc:>6.4f} {sh:>7.2f} {ret*100:>7.1f}% {dd*100:>6.1f}% {tr:>7d} {pf:>5.2f}")
        if i < 5:
            print(f"       params: {params}")

    # Best per strategy
    cur.execute("""
        SELECT strategy_name, pair, timeframe, parameters,
               sharpe, total_return, max_drawdown, total_trades, composite_score
        FROM experiments WHERE success = 1 AND total_trades >= 20
        ORDER BY composite_score DESC
    """)
    all_rows = cur.fetchall()
    seen = set()
    print("\n  Best per strategy:")
    for r in all_rows:
        sn = r[0]
        if sn not in seen:
            seen.add(sn)
            print(f"  {sn:<16s} {r[1].split('/')[0]:<5s} {r[2]:<4s} "
                  f"Sharpe={r[4]:.2f} Ret={r[5]*100:.1f}% DD={r[6]*100:.1f}% "
                  f"Trades={r[7]} Score={r[8]:.4f}")
            print(f"    params: {r[3]}")

    conn.close()
    print(f"\n  DB: {DB_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
