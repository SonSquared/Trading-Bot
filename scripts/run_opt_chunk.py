#!/usr/bin/env python3
"""
Run optimization for a fixed duration then exit.
Designed to be called repeatedly with --resume to complete the full run.

Usage:
    python scripts/run_opt_chunk.py --minutes 9
    python scripts/run_opt_chunk.py --minutes 5 --experiments 5000
"""

import sys
import time
import hashlib
import sqlite3
import json
from pathlib import Path
import numpy as np


def to_json_safe(obj):
    """Recursively convert numpy types to Python builtins for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.optimization.param_space import count_combinations, sample_random, generate_grid
from trading_system.strategies import get_strategy, ALL_STRATEGIES

DB_PATH = Path("data/results/experiments.db")
RANDOM_SAMPLE_THRESHOLD = 2000


def make_experiment_id(strategy_name, params, pair, timeframe):
    param_str = str(sorted(params.items()))
    raw = f"{strategy_name}_{pair}_{timeframe}_{param_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=9, help="Run for N minutes then exit")
    parser.add_argument("--experiments", type=int, default=999999, help="Max experiments this chunk")
    parser.add_argument("--random-cap", type=int, default=3000)
    parser.add_argument("--split", default="in_sample")
    parser.add_argument("--pairs", default="BTC/USDT:USDT,ETH/USDT:USDT")
    parser.add_argument("--timeframes", default="15m,30m,1h,4h")
    args = parser.parse_args()

    cfg = SystemConfig.default()
    pair_list = [p.strip() for p in args.pairs.split(",")]
    tf_list = [t.strip() for t in args.timeframes.split(",")]

    # Initialize DB
    db = DB_PATH
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY, strategy_name TEXT, pair TEXT, timeframe TEXT,
        parameters TEXT, results TEXT, success INTEGER, error TEXT,
        composite_score REAL, total_return REAL, sharpe REAL, sortino REAL,
        max_drawdown REAL, total_trades INTEGER, win_rate REAL,
        profit_factor REAL, timestamp TEXT
    )""")
    conn.commit()

    completed_ids = {row[0] for row in conn.execute(
        "SELECT experiment_id FROM experiments WHERE success = 1"
    ).fetchall()}
    print(f"Already completed: {len(completed_ids):,}", flush=True)

    # Load data
    loader = DataLoader(cfg)
    data_cache = {}
    for pair in pair_list:
        for tf in tf_list:
            df = loader.load(pair, tf)
            if df is None or df.empty:
                continue
            splits = loader.split_data(df)
            data_slice = splits.get(args.split, df)
            funding = loader.load_funding_rates(pair)
            data_cache[(pair, tf)] = {
                "df": data_slice,
                "funding": funding if funding is not None and not funding.empty else None,
            }

    engine = BacktestEngine(cfg.backtest)

    # Build experiment list
    experiments = []
    for strat in ALL_STRATEGIES:
        meta = strat.meta()
        grid = strat.param_grid()
        combos = count_combinations(grid)
        if combos > RANDOM_SAMPLE_THRESHOLD:
            param_list = sample_random(grid, args.random_cap, seed=42)
        else:
            param_list = generate_grid(grid)
        for pair in pair_list:
            for tf in tf_list:
                if (pair, tf) not in data_cache:
                    continue
                cached = data_cache[(pair, tf)]
                for params in param_list:
                    eid = make_experiment_id(meta.name, params, pair, tf)
                    if eid in completed_ids:
                        continue
                    experiments.append({
                        "strategy": meta.name, "params": params,
                        "pair": pair, "tf": tf,
                        "df": cached["df"], "funding": cached["funding"],
                    })

    total = len(experiments)
    print(f"Experiments to run: {total:,}", flush=True)

    deadline = time.time() + args.minutes * 60
    ok = 0
    fail = 0
    batch = []
    t0 = time.time()
    max_exp = min(total, args.experiments)

    for i, exp in enumerate(experiments):
        if time.time() > deadline or i >= max_exp:
            break

        try:
            strategy = get_strategy(exp["strategy"])
            signals = strategy.generate_signals(exp["df"], exp["params"])
            results = engine.run(
                df=exp["df"], signals=signals,
                strategy_name=exp["strategy"], params=exp["params"],
                pair=exp["pair"], timeframe=exp["tf"], funding_rates=exp["funding"],
            )
            res = results.to_dict()
            res = to_json_safe(res)
            params = to_json_safe(exp["params"])
            batch.append((
                make_experiment_id(exp["strategy"], exp["params"], exp["pair"], exp["tf"]),
                exp["strategy"], exp["pair"], exp["tf"],
                json.dumps(params), json.dumps(res), 1, "",
                0.0, res.get("total_return", 0.0), res.get("sharpe", 0.0),
                res.get("sortino", 0.0), res.get("max_drawdown", 0.0),
                res.get("total_trades", 0), res.get("win_rate", 0.0),
                res.get("profit_factor", 0.0), time.strftime("%Y-%m-%d %H:%M:%S"),
            ))
            ok += 1
        except Exception as e:
            fail += 1
            err = f"{type(e).__name__}: {e}"
            batch.append((
                make_experiment_id(exp["strategy"], exp["params"], exp["pair"], exp["tf"]),
                exp["strategy"], exp["pair"], exp["tf"],
                json.dumps(exp["params"]), json.dumps({}), 0, err[:500],
                0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ))
            if fail <= 3:
                print(f"  FAIL: {exp['strategy']} {exp['pair']} {exp['tf']} -> {err[:100]}", flush=True)

        if len(batch) >= 200:
            conn.executemany("""INSERT OR REPLACE INTO experiments
                (experiment_id, strategy_name, pair, timeframe, parameters, results,
                 success, error, composite_score, total_return, sharpe, sortino,
                 max_drawdown, total_trades, win_rate, profit_factor, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", batch)
            conn.commit()
            batch = []

    # Final save
    if batch:
        conn.executemany("""INSERT OR REPLACE INTO experiments
            (experiment_id, strategy_name, pair, timeframe, parameters, results,
             success, error, composite_score, total_return, sharpe, sortino,
             max_drawdown, total_trades, win_rate, profit_factor, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", batch)
        conn.commit()

    elapsed = time.time() - t0
    total_done = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    print(f"Chunk done: +{ok} OK, +{fail} fail in {elapsed:.0f}s | Total in DB: {total_done:,} / 123,052", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
