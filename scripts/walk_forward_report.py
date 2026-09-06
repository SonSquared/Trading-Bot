"""
Walk-Forward Validation Report

Generates a comprehensive report comparing in-sample (training) vs
out-of-sample (validation) performance for each walk-forward window.

Overfitting Detection:
- If in-sample returns are much higher than out-of-sample → overfitting
- Healthy: out-of-sample returns within 50% of in-sample
- Warning: out-of-sample returns <30% of in-sample
- Danger: out-of-sample returns are negative while in-sample is positive
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Fix Windows console encoding
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from scripts.walk_forward_optimize import (
    load_parquet,
    backtest_single,
    optimize_strategy,
    walk_forward_split,
    STRATEGY_CONFIGS,
)

RESULTS_DIR = Path("data/results")


def generate_report():
    """Generate the full walk-forward validation report."""
    print("=" * 70)
    print("WALK-FORWARD VALIDATION REPORT")
    print("=" * 70)
    print()

    # Load data
    eth_data = load_parquet("ETH_USDT_USDT")
    btc_data = load_parquet("BTC_USDT_USDT")
    min_len = min(len(eth_data), len(btc_data))
    eth_data = eth_data.tail(min_len).reset_index(drop=True)
    btc_data = btc_data.tail(min_len).reset_index(drop=True)
    print(f"Data: {min_len} candles ({eth_data['timestamp'].iloc[0]} to {eth_data['timestamp'].iloc[-1]})")
    print()

    # Split into windows
    windows = walk_forward_split(eth_data, btc_data, train_months=12, test_months=6)
    print(f"Windows: {len(windows)}")
    print()

    # Detailed results per strategy per window
    all_results = []

    for strat_name, config in STRATEGY_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"STRATEGY: {strat_name}")
        print(f"{'='*70}")

        for pair in config["pairs"]:
            pair_label = pair.replace("_USDT_USDT", "")
            data_full = eth_data if "ETH" in pair else btc_data

            print(f"\n  --- {pair_label} ---")
            print(f"  {'Window':<8} {'Train Period':<28} {'Test Period':<28} {'IS Return':>10} {'OOS Return':>11} {'IS WR':>7} {'OOS WR':>8} {'OOS DD':>7} {'Overfit?':>10}")
            print(f"  {'-'*8} {'-'*28} {'-'*28} {'-'*10} {'-'*11} {'-'*7} {'-'*8} {'-'*7} {'-'*10}")

            for w_idx, window in enumerate(windows):
                train_data = data_full.iloc[window["train_start"]:window["train_end"]].reset_index(drop=True)
                test_data = data_full.iloc[window["test_start"]:window["test_end"]].reset_index(drop=True)

                if len(train_data) < 50 or len(test_data) < 20:
                    print(f"  {w_idx+1:<8} {'Insufficient data':<28}")
                    continue

                # Optimize on train
                train_results = optimize_strategy(strat_name, pair, train_data, config["grid"], max_combos=300)

                if not train_results:
                    print(f"  {w_idx+1:<8} {'No valid results':<28}")
                    continue

                # Validate top 5 on test
                top_5 = train_results[:5]
                best_is = top_5[0]

                # Also test the default params (for comparison)
                test_results_all = []
                for tr in top_5:
                    test_r = backtest_single(test_data, strat_name, tr["params"])
                    test_results_all.append({
                        "params": tr["params"],
                        "is_return": tr["return_pct"],
                        "is_win_rate": tr["win_rate"],
                        "is_max_dd": tr["max_dd"],
                        "is_score": tr["score"],
                        "oos_return": test_r["return_pct"],
                        "oos_win_rate": test_r["win_rate"],
                        "oos_max_dd": test_r["max_dd"],
                        "oos_score": test_r["score"],
                    })

                # Honest comparison: use the TOP TRAIN candidate (rank 1 on
                # in-sample), never the one that happened to score best on the
                # test set — picking "best by OOS" makes the diagnostic
                # itself overfit to the validation data.
                best_oos = test_results_all[0]

                # Overfitting detection
                is_ret = best_oos["is_return"]
                oos_ret = best_oos["oos_return"]

                if is_ret > 0 and oos_ret < 0:
                    overfit = "DANGER"
                elif is_ret > 0 and oos_ret < is_ret * 0.3:
                    overfit = "WARNING"
                elif is_ret > 0 and oos_ret < is_ret * 0.5:
                    overfit = "CAUTION"
                else:
                    overfit = "OK"

                train_period = f"{window['train_start_date']} to {window['train_end_date']}"
                test_period = f"{window['test_start_date']} to {window['test_end_date']}"

                print(f"  {w_idx+1:<8} {train_period:<28} {test_period:<28} {is_ret:>+9.1f}% {oos_ret:>+10.1f}% {best_oos['is_win_rate']:>6.0f}% {best_oos['oos_win_rate']:>7.0f}% {best_oos['oos_max_dd']:>6.1f}% {overfit:>10}")

                all_results.append({
                    "strategy": strat_name,
                    "pair": pair,
                    "window": w_idx + 1,
                    "train_start": window["train_start_date"],
                    "train_end": window["train_end_date"],
                    "test_start": window["test_start_date"],
                    "test_end": window["test_end_date"],
                    "best_params": best_oos["params"],
                    "is_return": is_ret,
                    "oos_return": oos_ret,
                    "is_win_rate": best_oos["is_win_rate"],
                    "oos_win_rate": best_oos["oos_win_rate"],
                    "is_max_dd": best_oos["is_max_dd"],
                    "oos_max_dd": best_oos["oos_max_dd"],
                    "overfit_status": overfit,
                    "is_score": best_oos["is_score"],
                    "oos_score": best_oos["oos_score"],
                })

    # --- Summary Statistics ---
    print(f"\n{'='*70}")
    print("OVERFITTING SUMMARY")
    print(f"{'='*70}")

    total = len(all_results)
    ok = sum(1 for r in all_results if r["overfit_status"] == "OK")
    caution = sum(1 for r in all_results if r["overfit_status"] == "CAUTION")
    warning = sum(1 for r in all_results if r["overfit_status"] == "WARNING")
    danger = sum(1 for r in all_results if r["overfit_status"] == "DANGER")

    print(f"\n  Total windows tested: {total}")
    print(f"  OK (healthy):      {ok} ({ok/total*100:.0f}%)")
    print(f"  CAUTION:           {caution} ({caution/total*100:.0f}%)")
    print(f"  WARNING:           {warning} ({warning/total*100:.0f}%)")
    print(f"  DANGER (overfit):  {danger} ({danger/total*100:.0f}%)")

    # Correlation between IS and OOS returns
    is_rets = [r["is_return"] for r in all_results]
    oos_rets = [r["oos_return"] for r in all_results]
    if len(is_rets) > 2:
        corr = np.corrcoef(is_rets, oos_rets)[0, 1]
        print(f"\n  IS-OOS return correlation: {corr:.3f}")
        if corr > 0.7:
            print("  -> Strong positive correlation (low overfitting risk)")
        elif corr > 0.3:
            print("  -> Moderate correlation (some overfitting risk)")
        else:
            print("  -> Weak correlation (high overfitting risk)")

    # Degradation ratio
    avg_is = np.mean(is_rets)
    avg_oos = np.mean(oos_rets)
    if avg_is != 0:
        degradation = (1 - avg_oos / avg_is) * 100
        print(f"  Avg IS return: {avg_is:+.1f}%")
        print(f"  Avg OOS return: {avg_oos:+.1f}%")
        print(f"  Degradation: {degradation:.0f}%")

    # Per-strategy summary
    for strat_name in STRATEGY_CONFIGS:
        strat_results = [r for r in all_results if r["strategy"] == strat_name]
        if not strat_results:
            continue
        s_is = [r["is_return"] for r in strat_results]
        s_oos = [r["oos_return"] for r in strat_results]
        s_ok = sum(1 for r in strat_results if r["overfit_status"] == "OK")
        s_total = len(strat_results)
        print(f"\n  {strat_name}:")
        print(f"    Avg IS: {np.mean(s_is):+.1f}% | Avg OOS: {np.mean(s_oos):+.1f}%")
        print(f"    Healthy: {s_ok}/{s_total} windows")

    # --- Recommendations ---
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    if danger > 0:
        print(f"\n  WARNING: {danger} window(s) show classic overfitting (positive IS, negative OOS)")
        print("     -> Consider narrowing parameter grid or adding regularization")

    if corr < 0.3 if len(is_rets) > 2 else False:
        print("\n  WARNING: Weak IS-OOS correlation suggests parameters don't generalize well")
        print("     -> Consider reducing parameter count or using simpler strategies")

    if degradation > 60:
        print(f"\n  WARNING: {degradation:.0f}% degradation from IS to OOS is high")
        print("     -> Expected for mean-reversion strategies in changing regimes")

    if ok / total > 0.6:
        print("\n  OK: Most windows show healthy generalization")
        print("     -> Current strategy selection is reasonably robust")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_range": f"{eth_data['timestamp'].iloc[0]} to {eth_data['timestamp'].iloc[-1]}",
        "total_windows": len(windows),
        "total_tests": total,
        "overfitting_summary": {
            "ok": ok, "caution": caution, "warning": warning, "danger": danger,
        },
        "is_oos_correlation": float(corr) if len(is_rets) > 2 else None,
        "avg_is_return": float(avg_is),
        "avg_oos_return": float(avg_oos),
        "degradation_pct": float(degradation) if avg_is != 0 else None,
        "window_results": all_results,
    }

    out_path = RESULTS_DIR / "walk_forward_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    generate_report()
