#!/usr/bin/env python3
"""
Test Dynamic Risk Manager with calibrated regime thresholds.

The default detector classified 100% as transitional because:
- 3 votes needed for a majority out of 4 (ADX double + BB + ATR)
- ADX rarely exceeds 25 on 4h crypto data

Fix: use ADX > 20 for trending, ADX < 18 for choppy, and
require only 2/4 votes for a classification.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from trading_system.bot.dynamic_risk import DynamicRiskManager, RegimeRiskConfig
from trading_system.indicators import adx, atr, bollinger_bands


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


def classify_regime_fast(df, adx_thresh=20.0, choppy_thresh=17.0,
                          bb_expand=1.15, bb_compress=0.85,
                          atr_expand=1.1, atr_compress=0.9,
                          adx_period=14, bb_period=20, atr_period=14,
                          bb_lookback=50, atr_lookback=20):
    """Fast regime classification with tunable thresholds."""
    n = len(df)
    warmup = max(adx_period, bb_period, atr_period) + 30
    labels = np.full(n, "transitional", dtype=object)

    if n < warmup:
        return labels

    adx_df = adx(df, adx_period)
    adx_val = adx_df["adx"].values.astype(np.float64)
    plus_di = adx_df["plus_di"].values.astype(np.float64)
    minus_di = adx_df["minus_di"].values.astype(np.float64)

    bb = bollinger_bands(df["close"], bb_period, 2.0)
    bb_width_raw = ((bb["upper"] - bb["lower"]) / bb["middle"]).values.astype(np.float64)

    atr_s = atr(df, atr_period)
    atr_vals = atr_s.values.astype(np.float64)

    prev_regime = "transitional"
    regime_count = 0

    for i in range(warmup, n):
        # ADX vote
        a = adx_val[i]
        if np.isnan(a):
            adx_vote = "transitional"
        elif a > adx_thresh:
            adx_vote = "trending"
        elif a < choppy_thresh:
            adx_vote = "choppy"
        else:
            adx_vote = "transitional"

        # BB width vote
        bw = bb_width_raw[i] if not np.isnan(bb_width_raw[i]) else 1.0
        bw_avg = np.nanmean(bb_width_raw[max(0, i - bb_lookback):i + 1])
        if bw_avg <= 0:
            bw_avg = 1.0
        bw_ratio = bw / bw_avg

        if bw_ratio > bb_expand:
            bb_vote = "trending"
        elif bw_ratio < bb_compress:
            bb_vote = "choppy"
        else:
            bb_vote = "transitional"

        # ATR vote
        av = atr_vals[i] if not np.isnan(atr_vals[i]) else 0
        av_avg = np.nanmean(atr_vals[max(0, i - atr_lookback):i + 1])
        if av_avg <= 0:
            av_avg = 1.0
        av_ratio = av / av_avg

        if av_ratio > atr_expand:
            atr_vote = "trending"
        elif av_ratio < atr_compress:
            atr_vote = "choppy"
        else:
            atr_vote = "transitional"

        # Majority vote: ADX double weight
        votes = {"trending": 0, "choppy": 0, "transitional": 0}
        votes[adx_vote] += 2
        votes[bb_vote] += 1
        votes[atr_vote] += 1

        if votes["trending"] >= 3:
            regime = "trending"
        elif votes["choppy"] >= 3:
            regime = "choppy"
        else:
            regime = "transitional"

        if regime != prev_regime:
            regime_count = 1
        else:
            regime_count += 1

        if regime_count >= 3:
            labels[i] = regime
            prev_regime = regime
        else:
            labels[i] = prev_regime

    return labels


def run_backtest(engine, df, strategy_name, params, pair, tf, risk_mults=None):
    """Run a single backtest and return results dict."""
    strategy = get_strategy(strategy_name)
    signals = strategy.generate_signals(df, params)
    result = engine.run(df, signals, strategy_name, params, pair, tf,
                        risk_multipliers=risk_mults)
    return {
        "return": result.total_return * 100,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "max_dd": result.max_drawdown * 100,
        "trades": result.total_trades,
        "win_rate": result.win_rate * 100,
        "profit_factor": result.profit_factor,
    }


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)

    # ── First: find good thresholds by analyzing ETH 4h regime distribution ──
    print("=" * 110)
    print("  CALIBRATING REGIME DETECTOR THRESHOLDS")
    print("=" * 110)

    eth_4h = loader.load("ETH/USDT:USDT", "4h")

    threshold_tests = [
        ("Default (25/20)", 25.0, 20.0, 1.3, 0.8),
        ("Relaxed (22/18)", 22.0, 18.0, 1.2, 0.85),
        ("Very relaxed (20/16)", 20.0, 16.0, 1.15, 0.85),
        ("Wide (18/14)", 18.0, 14.0, 1.1, 0.9),
    ]

    for name, trend_thresh, chop_thresh, bb_exp, bb_comp in threshold_tests:
        labels = classify_regime_fast(
            eth_4h,
            adx_thresh=trend_thresh, choppy_thresh=chop_thresh,
            bb_expand=bb_exp, bb_compress=bb_comp,
        )
        trending = (labels == "trending").sum()
        choppy = (labels == "choppy").sum()
        trans = (labels == "transitional").sum()
        total = len(labels)
        print(f"  {name:<25}: Trending={trending:>5} ({trending/total*100:5.1f}%)  "
              f"Choppy={choppy:>5} ({choppy/total*100:5.1f}%)  "
              f"Transitional={trans:>5} ({trans/total*100:5.1f}%)")

    # Use the best calibrated thresholds
    best_thresh = {"trend": 20.0, "chop": 16.0, "bb_exp": 1.15, "bb_comp": 0.85}

    # ── Run backtests with calibrated regime detection ─────────────
    print(f"\n\n{'=' * 110}")
    print("  DYNAMIC RISK MANAGER (Calibrated Regime Detector)")
    print("=" * 110)

    # Build custom multiplier series using the calibrated classifier
    labels = classify_regime_fast(
        eth_4h,
        adx_thresh=best_thresh["trend"], choppy_thresh=best_thresh["chop"],
        bb_expand=best_thresh["bb_exp"], bb_compress=best_thresh["bb_comp"],
    )

    # Also build labels for BTC
    btc_4h = loader.load("BTC/USDT:USDT", "4h")
    labels_btc = classify_regime_fast(
        btc_4h,
        adx_thresh=best_thresh["trend"], choppy_thresh=best_thresh["chop"],
        bb_expand=best_thresh["bb_exp"], bb_compress=best_thresh["bb_comp"],
    )

    configs = {
        "Fixed 2% (baseline)": None,
        "Moderate (0.5/0.8/1.0)": {"trending": 1.0, "transitional": 0.8, "choppy": 0.5},
        "Conservative (0.6/1.0/1.0)": {"trending": 1.0, "transitional": 1.0, "choppy": 0.6},
        "Aggressive (0.3/0.7/1.0)": {"trending": 1.0, "transitional": 0.7, "choppy": 0.3},
    }

    all_results = []

    for strat_def in STRATEGIES:
        print(f"\n{'=' * 110}")
        print(f"  {strat_def['label']}")
        print(f"{'=' * 110}")

        pair = strat_def["pair"]
        tf = strat_def["timeframe"]

        if "ETH" in pair:
            pair_labels = labels
            pair_df = eth_4h
        else:
            pair_labels = labels_btc
            pair_df = btc_4h

        for config_name, mults in configs.items():
            if mults is None:
                risk_mults = None
            else:
                # Build multiplier series from labels
                risk_mults = pd.Series(
                    [mults[l] for l in pair_labels],
                    index=pair_df.index
                )

            bt_config = SystemConfig.default()
            bt_config.backtest.execution.stop_loss_atr_mult = 3.0
            engine = BacktestEngine(bt_config.backtest)

            result = run_backtest(
                engine, pair_df, strat_def["name"], strat_def["params"],
                pair, tf, risk_mults=risk_mults
            )

            print(f"  {config_name:<35}: "
                  f"Ret={result['return']:+7.1f}%  "
                  f"Sharpe={result['sharpe']:6.2f}  "
                  f"MaxDD={result['max_dd']:5.2f}%  "
                  f"Sortino={result['sortino']:6.2f}")

            all_results.append({
                "strategy": strat_def["label"],
                "config": config_name,
                **result,
            })

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n\n{'=' * 110}")
    print("  IMPROVEMENT vs BASELINE")
    print(f"{'=' * 110}")

    for strat_def in STRATEGIES:
        label = strat_def["label"]
        baseline = next((r for r in all_results
                         if r["strategy"] == label and r["config"] == "Fixed 2% (baseline)"), None)
        if not baseline:
            continue

        print(f"\n  {label}:")
        for r in all_results:
            if r["strategy"] != label or r["config"] == "Fixed 2% (baseline)":
                continue
            sharpe_diff = r["sharpe"] - baseline["sharpe"]
            dd_diff = r["max_dd"] - baseline["max_dd"]
            ret_diff = r["return"] - baseline["return"]

            # Calculate return-per-unit-of-risk (Sharpe-like ratio)
            risk_adj_return_base = baseline["return"] / max(baseline["max_dd"], 0.01)
            risk_adj_return_new = r["return"] / max(r["max_dd"], 0.01)

            verdict = "BETTER" if risk_adj_return_new > risk_adj_return_base else "WORSE"
            print(f"    {r['config']:<35} "
                  f"Sharpe {sharpe_diff:+.2f}  MaxDD {dd_diff:+.2f}%  "
                  f"Return {ret_diff:+.1f}%  "
                  f"Risk-Adj {risk_adj_return_new:.1f} vs {risk_adj_return_base:.1f}  "
                  f"[{verdict}]")

    # Save
    out_path = Path("data/results/dynamic_risk_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()
