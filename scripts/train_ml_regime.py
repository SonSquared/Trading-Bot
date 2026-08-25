"""
Train and evaluate the ML regime detector.

Loads historical data, trains the ML model with walk-forward validation,
compares against the rule-based detector, and saves the trained model.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from trading_system.bot.ml_regime_detector import MLRegimeDetector, MLRegimeConfig
from trading_system.bot.regime_detector import RegimeDetector
from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader


def evaluate_rule_based(regime_labels: pd.Series, future_return: pd.Series,
                        threshold: float = 1.5) -> dict:
    """Evaluate the rule-based detector against actual future returns."""
    # True labels: 1 = trending, 0 = non-trending
    true_labels = (future_return.abs() > threshold).astype(int)

    # Map rule-based labels
    rule_pred = regime_labels.map({
        "trending": 1,
        "choppy": 0,
        "transitional": 0.5,
    })

    # Only evaluate where both have data
    valid = true_labels.notna() & rule_pred.notna()
    if valid.sum() == 0:
        return {"accuracy": 0, "f1": 0}

    t = true_labels[valid].values
    r = rule_pred[valid].values

    # Binary: > 0.5 counts as trending
    r_binary = (r > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(t, r_binary)),
        "f1": float(f1_score(t, r_binary, zero_division=0)),
        "n_samples": int(valid.sum()),
    }


def main():
    print("=" * 70)
    print("ML REGIME DETECTOR — TRAINING AND EVALUATION")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────
    print("\n[1] Loading historical data...")
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    datasets = {}
    for pair in ["ETH/USDT:USDT", "BTC/USDT:USDT"]:
        for tf in ["4h"]:
            try:
                df = loader.load(pair, tf)
                key = f"{pair}_{tf}"
                datasets[key] = df
                print(f"  {pair} {tf}: {len(df):,} candles "
                      f"({df.index[0]} to {df.index[-1]})")
            except Exception as e:
                print(f"  {pair} {tf}: FAILED - {e}")

    if not datasets:
        print("ERROR: No data loaded!")
        return

    # Use ETH 4h as primary training data
    primary_key = "ETH/USDT:USDT_4h"
    if primary_key not in datasets:
        primary_key = list(datasets.keys())[0]

    df = datasets[primary_key]
    print(f"\n  Primary dataset: {primary_key} ({len(df):,} candles)")

    # ── Train ML detector ────────────────────────────────────────
    print("\n[2] Training ML regime detector (walk-forward)...")
    config = MLRegimeConfig(
        forward_candles=6,
        trend_threshold_pct=1.5,
        train_months=12,
        test_months=3,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_confidence=0.55,
    )

    t0 = time.time()
    ml_detector = MLRegimeDetector(config)
    metrics = ml_detector.train(df, verbose=True)
    train_time = time.time() - t0
    print(f"\n  Training completed in {train_time:.1f}s")

    # ── Save model ──────────────────────────────────────────────
    print("\n[3] Saving model...")
    ml_detector.save()
    print("  Saved to data/models/ml_regime_model.pkl")

    # ── Compare with rule-based ─────────────────────────────────
    print("\n[4] Comparing ML vs Rule-Based detector...")

    # Rule-based detector
    rule_detector = RegimeDetector(
        adx_period=14,
        adx_trend_threshold=25.0,
        adx_choppy_threshold=20.0,
    )

    # Forward returns for evaluation (6 candles ahead = 1 day on 4h)
    future_return = df["close"].pct_change(6).shift(-6) * 100

    # Rule-based regime labels
    rule_labels = rule_detector.get_regime_history(df, lookback=len(df))
    rule_series = pd.Series(
        [r["regime"] for r in rule_labels],
        index=df.index[-len(rule_labels):],
    )

    # ML predictions
    ml_preds, ml_probs = ml_detector.predict(df)

    # Evaluate both
    rule_metrics = evaluate_rule_based(rule_series, future_return.reindex(rule_series.index), 1.5)
    ml_binary = (ml_preds > 0.5).astype(int)
    ml_valid = future_return.notna() & ml_preds.notna()
    if ml_valid.sum() > 0:
        from sklearn.metrics import accuracy_score, f1_score
        ml_acc = accuracy_score(
            (future_return[ml_valid].abs() > 1.5).astype(int),
            ml_binary[ml_valid],
        )
        ml_f1 = f1_score(
            (future_return[ml_valid].abs() > 1.5).astype(int),
            ml_binary[ml_valid],
            zero_division=0,
        )
    else:
        ml_acc = ml_f1 = 0

    print(f"\n  {'Metric':<25s} {'Rule-Based':>12s} {'ML Model':>12s} {'Delta':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<25s} {rule_metrics['accuracy']:>12.3f} {ml_acc:>12.3f} {ml_acc - rule_metrics['accuracy']:>+10.3f}")
    print(f"  {'F1 Score':<25s} {rule_metrics['f1']:>12.3f} {ml_f1:>12.3f} {ml_f1 - rule_metrics['f1']:>+10.3f}")
    print(f"  {'Samples':<25s} {rule_metrics.get('n_samples', 0):>12,} {ml_valid.sum():>12,}")

    # ── Regime distribution ─────────────────────────────────────
    print("\n[5] Regime distribution comparison...")

    # Rule-based distribution
    rule_dist = rule_series.value_counts(normalize=True) * 100
    print(f"\n  Rule-Based:")
    for regime in ["trending", "choppy", "transitional"]:
        pct = rule_dist.get(regime, 0)
        print(f"    {regime:15s} {pct:>6.1f}%")

    # ML distribution
    ml_dist = ml_detector.get_regime_distribution(df)
    print(f"\n  ML Model:")
    for regime in ["trending", "choppy", "transitional"]:
        pct = ml_dist[regime]["pct"]
        conf = ml_dist[regime]["avg_confidence"]
        print(f"    {regime:15s} {pct:>6.1f}%  (confidence: {conf:.3f})")

    # ── Feature importance ──────────────────────────────────────
    print("\n[6] Feature importance (top 15):")
    importance = ml_detector.get_feature_importance_report()
    for i, (name, score) in enumerate(list(importance.items())[:15]):
        bar = "#" * int(score * 150)
        print(f"  {i+1:2d}. {name:25s} {score:.4f} {bar}")

    # ── Strategy profitability by ML regime ─────────────────────
    print("\n[7] Strategy returns by ML-predicted regime...")

    # Load strategy
    from trading_system.strategies import get_strategy
    from trading_system.backtester.engine import BacktestEngine
    from trading_system.config import BacktestConfig

    strat = get_strategy("MACD")
    signals = strat.generate_signals(df, {
        "fast_period": 12, "slow_period": 26, "signal_period": 9,
    })

    for regime_name in ["trending", "choppy", "transitional"]:
        mask = ml_preds.map({1.0: "trending", 0.0: "choppy", 0.5: "transitional"}) == regime_name
        n = mask.sum()
        if n == 0:
            continue

        # Run backtest only on candles in this regime
        regime_signals = signals.where(mask, 0)  # Zero out signals outside regime
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(df, regime_signals)

        print(f"  {regime_name:15s}: n={n:>5d} ({n/len(df)*100:>5.1f}%) "
              f"return={result.total_return*100:>+8.2f}% "
              f"sharpe={result.sharpe:>7.2f} "
              f"wr={result.win_rate*100:.1f}%")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Walk-forward accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"  Walk-forward F1:       {metrics['overall_f1']:.3f}")
    print(f"  Rule-based accuracy:   {rule_metrics['accuracy']:.3f}")
    print(f"  Improvement:           {ml_acc - rule_metrics['accuracy']:+.3f}")
    print(f"  Model saved to:        data/models/ml_regime_model.pkl")
    print(f"  Training time:         {train_time:.1f}s")

    # Also train on BTC to check cross-pair stability
    if "BTC/USDT:USDT_4h" in datasets and "BTC/USDT:USDT_4h" != primary_key:
        print(f"\n[BONUS] Cross-pair validation on BTC...")
        btc_df = datasets["BTC/USDT:USDT_4h"]
        btc_ml = MLRegimeDetector(config)
        btc_metrics = btc_ml.train(btc_df, verbose=False)
        print(f"  BTC walk-forward accuracy: {btc_metrics['overall_accuracy']:.3f}")
        print(f"  BTC walk-forward F1:       {btc_metrics['overall_f1']:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
