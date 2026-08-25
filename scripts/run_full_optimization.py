#!/usr/bin/env python3
"""
Full optimization: all strategies x all pairs x all timeframes.

Sequential execution using the vectorized engine (~60ms/trial).
Saves results to SQLite incrementally (every 500 experiments) so progress
survives crashes. On restart, completed experiments are skipped (resume).

Usage:
    python scripts/run_full_optimization.py
    python scripts/run_full_optimization.py --random-cap 5000
    python scripts/run_full_optimization.py --split validation
    python scripts/run_full_optimization.py --resume   # skip completed
"""

from __future__ import annotations

import sys
import os
import time
import hashlib
import sqlite3
import json
import hashlib as _hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.optimization.param_space import count_combinations, sample_random, generate_grid
from trading_system.strategies import get_strategy, ALL_STRATEGIES

# Grid size above which we use random sampling
RANDOM_SAMPLE_THRESHOLD = 2000

DB_PATH = Path("data/results/experiments.db")
LOG_PATH = Path("data/results/optimization_log.txt")

_log_fh = None

def log(msg, level="info"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [{level}] {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if _log_fh:
        _log_fh.write(line)
        _log_fh.flush()


def make_experiment_id(strategy_name, params, pair, timeframe):
    param_str = str(sorted(params.items()))
    raw = f"{strategy_name}_{pair}_{timeframe}_{param_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def init_db(db_path):
    """Create DB and table, return connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            strategy_name TEXT,
            pair TEXT,
            timeframe TEXT,
            parameters TEXT,
            results TEXT,
            success INTEGER,
            error TEXT,
            composite_score REAL,
            total_return REAL,
            sharpe REAL,
            sortino REAL,
            max_drawdown REAL,
            total_trades INTEGER,
            win_rate REAL,
            profit_factor REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn


def load_completed_ids(conn):
    """Load set of already-completed experiment IDs."""
    cur = conn.execute("SELECT experiment_id FROM experiments WHERE success = 1")
    return {row[0] for row in cur.fetchall()}


def save_batch_to_db(conn, batch):
    """Save a batch of results to DB."""
    for r in batch:
        res = r.get("results", {})
        try:
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (experiment_id, strategy_name, pair, timeframe, parameters,
                    results, success, error, composite_score, total_return,
                    sharpe, sortino, max_drawdown, total_trades, win_rate,
                    profit_factor, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r.get("experiment_id", ""),
                    r.get("strategy_name", ""),
                    r.get("pair", ""),
                    r.get("timeframe", ""),
                    json.dumps(r.get("parameters", {})),
                    json.dumps(res),
                    1 if r.get("success") else 0,
                    r.get("error", ""),
                    res.get("composite_score", 0.0),
                    res.get("total_return", 0.0),
                    res.get("sharpe", 0.0),
                    res.get("sortino", 0.0),
                    res.get("max_drawdown", 0.0),
                    res.get("total_trades", 0),
                    res.get("win_rate", 0.0),
                    res.get("profit_factor", 0.0),
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        except Exception as e:
            log(f"DB save error for {r.get('experiment_id','?')}: {e}", "error")
    conn.commit()


def run_single_experiment(engine, strategy_name, params, df, pair, timeframe, funding_rates):
    """Run one experiment. Returns result dict."""
    try:
        strategy = get_strategy(strategy_name)
        signals = strategy.generate_signals(df, params)
        results = engine.run(
            df=df,
            signals=signals,
            strategy_name=strategy_name,
            params=params,
            pair=pair,
            timeframe=timeframe,
            funding_rates=funding_rates,
        )
        return {
            "experiment_id": make_experiment_id(strategy_name, params, pair, timeframe),
            "strategy_name": strategy_name,
            "parameters": params,
            "pair": pair,
            "timeframe": timeframe,
            "results": results.to_dict(),
            "success": True,
        }
    except Exception as e:
        return {
            "experiment_id": make_experiment_id(strategy_name, params, pair, timeframe),
            "strategy_name": strategy_name,
            "parameters": params,
            "pair": pair,
            "timeframe": timeframe,
            "results": {},
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    global _log_fh
    import argparse

    parser = argparse.ArgumentParser(description="Full optimization (with checkpointing)")
    parser.add_argument("--random-cap", type=int, default=3000, help="Max random samples for huge grids")
    parser.add_argument("--pairs", default="BTC/USDT:USDT,ETH/USDT:USDT")
    parser.add_argument("--timeframes", default="15m,30m,1h,4h")
    parser.add_argument("--split", default="in_sample", help="Data split to optimize on")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save to DB every N experiments")
    args = parser.parse_args()

    _log_fh = open(LOG_PATH, "a", encoding="utf-8")

    cfg = SystemConfig.default()
    pair_list = [p.strip() for p in args.pairs.split(",")]
    tf_list = [t.strip() for t in args.timeframes.split(",")]

    log("=" * 70)
    log("  FULL OPTIMIZATION RUN" + (" [RESUME]" if args.resume else ""))
    log("=" * 70)
    log(f"  Strategies: {len(ALL_STRATEGIES)}")
    log(f"  Pairs: {pair_list}")
    log(f"  Timeframes: {tf_list}")
    log(f"  Mode: Sequential (vectorized engine)")
    log(f"  Checkpoint every: {args.checkpoint_every}")

    # Show combo counts
    log("  Parameter combinations per strategy:")
    total_per_spt = 0
    for strat in ALL_STRATEGIES:
        meta = strat.meta()
        grid = strat.param_grid()
        combos = count_combinations(grid)
        total_per_spt += combos
        method = "random" if combos > RANDOM_SAMPLE_THRESHOLD else "grid"
        log(f"    {meta.name:25s} | {meta.family:15s} | {combos:>8,} combos | {method}")

    grand_total = total_per_spt * len(pair_list) * len(tf_list)
    log(f"  Total per strategy/pair/tf: {total_per_spt:,}")
    log(f"  Grand total (all pairs x tfs): {grand_total:,}")

    # Initialize DB
    conn = init_db(DB_PATH)
    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(conn)
        log(f"  Resuming: {len(completed_ids):,} experiments already completed")

    # Load all data upfront
    loader = DataLoader(cfg)

    log("  Loading data...")
    data_cache = {}
    for pair in pair_list:
        for tf in tf_list:
            df = loader.load(pair, tf)
            if df is None or df.empty:
                log(f"    WARNING: No data for {pair} {tf}")
                continue

            splits = loader.split_data(df)
            data_slice = splits.get(args.split, df)
            funding = loader.load_funding_rates(pair)

            data_cache[(pair, tf)] = {
                "df": data_slice,
                "funding": funding if funding is not None and not funding.empty else None,
            }
            log(f"    {pair} {tf}: {len(data_slice)} candles ({args.split})")

    if not data_cache:
        log("ERROR: No data loaded. Exiting.")
        _log_fh.close()
        return

    # Create engine once
    engine = BacktestEngine(cfg.backtest)

    # Build experiment plan
    experiments = []
    for strat in ALL_STRATEGIES:
        meta = strat.meta()
        grid = strat.param_grid()
        combos = count_combinations(grid)

        if combos > RANDOM_SAMPLE_THRESHOLD:
            param_list = sample_random(grid, args.random_cap, seed=42)
            method = "random"
        else:
            param_list = generate_grid(grid)
            method = "grid"

        for pair in pair_list:
            for tf in tf_list:
                if (pair, tf) not in data_cache:
                    continue
                cached = data_cache[(pair, tf)]
                for params in param_list:
                    eid = make_experiment_id(meta.name, params, pair, tf)
                    if args.resume and eid in completed_ids:
                        continue
                    experiments.append({
                        "strategy": meta.name,
                        "family": meta.family,
                        "params": params,
                        "pair": pair,
                        "tf": tf,
                        "df": cached["df"],
                        "funding": cached["funding"],
                        "method": method,
                    })

    total = len(experiments)
    log(f"  Experiments to run: {total:,}")
    if args.resume and completed_ids:
        log(f"  (skipped {len(completed_ids):,} already done)")
    log("")

    # Run experiments with incremental checkpointing
    all_results = []
    successful = 0
    failed = 0
    start_time = time.time()
    report_interval = max(1, total // 20)  # ~5% increments
    checkpoint_every = args.checkpoint_every

    log("  Running experiments...")
    log("")

    for i, exp in enumerate(experiments):
        result = run_single_experiment(
            engine=engine,
            strategy_name=exp["strategy"],
            params=exp["params"],
            df=exp["df"],
            pair=exp["pair"],
            timeframe=exp["tf"],
            funding_rates=exp["funding"],
        )

        all_results.append(result)
        if result["success"]:
            successful += 1
        else:
            failed += 1

        # Incremental checkpoint
        if len(all_results) >= checkpoint_every:
            save_batch_to_db(conn, all_results)
            all_results = []  # clear buffer

        # Progress reporting
        total_done = i + 1
        if total_done % report_interval == 0 or total_done == total:
            elapsed = time.time() - start_time
            rate = total_done / elapsed if elapsed > 0 else 0
            eta_min = (total - total_done) / rate / 60 if rate > 0 else 0
            pct = total_done / total * 100
            log(
                f"  [{pct:5.1f}%] {total_done:>6,}/{total:,} | "
                f"OK: {successful:>6,} | Fail: {failed:>4,} | "
                f"{rate:.1f}/s | ETA: {eta_min:.1f}min"
            )

            # Print first few failures for debugging
            if failed > 0 and failed <= 10:
                for r in all_results:
                    if not r.get("success") and "error" in r:
                        log(f"    FAIL: {r['strategy_name']} {r['pair']} -> {r.get('error', '?')[:120]}")
                        break

    # Save remaining results
    if all_results:
        save_batch_to_db(conn, all_results)

    elapsed = time.time() - start_time
    log("")
    log(f"  Complete: {successful:,} successful, {failed:,} failed in {elapsed/60:.1f} minutes")

    # Read all results from DB for summary
    cur = conn.execute("SELECT * FROM experiments WHERE success = 1 ORDER BY sharpe DESC")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]

    log("")
    log("=" * 70)
    log("  TOP 30 RESULTS BY SHARPE RATIO")
    log("=" * 70)
    log("")
    log(f"  {'#':>3} {'Strategy':25s} {'Pair':8s} {'TF':5s} {'Sharpe':>8s} {'Sortino':>8s} {'Return':>8s} {'MaxDD':>8s} {'Trades':>6s} {'WinR':>6s} {'PF':>6s}")
    log("  " + "-" * 105)

    for i, row in enumerate(rows[:30]):
        d = dict(zip(cols, row))
        pair_short = d.get("pair", "?").split("/")[0]
        log(
            f"  {i+1:>3} {d.get('strategy_name','?'):25s} {pair_short:8s} {d.get('timeframe','?'):5s} "
            f"{d.get('sharpe',0):>8.2f} {d.get('sortino',0):>8.2f} "
            f"{d.get('total_return',0):>7.1%} {d.get('max_drawdown',0):>7.1%} "
            f"{d.get('total_trades',0):>6d} {d.get('win_rate',0):>5.1%} "
            f"{d.get('profit_factor',0):>6.2f}"
        )

    # Summary by strategy
    log("")
    log("=" * 70)
    log("  RESULTS BY STRATEGY")
    log("=" * 70)

    cur2 = conn.execute("""
        SELECT strategy_name,
               COUNT(*) as cnt,
               AVG(sharpe) as avg_sh,
               MAX(sharpe) as best_sh
        FROM experiments
        WHERE success = 1
        GROUP BY strategy_name
        ORDER BY best_sh DESC
    """)
    for row in cur2.fetchall():
        log(f"    {row[0]:25s} | Experiments: {row[1]:>5,} | Avg Sharpe: {row[2]:>6.2f} | Best: {row[3]:>6.2f}")

    # Failure summary
    cur3 = conn.execute("SELECT COUNT(*) FROM experiments WHERE success = 0")
    fail_count = cur3.fetchone()[0]
    if fail_count > 0:
        log("")
        log(f"  {fail_count:,} failures (logged in DB with error details)")

    conn.close()
    _log_fh.close()

    log("")
    log("Full optimization complete! Results in data/results/experiments.db")


if __name__ == "__main__":
    main()
