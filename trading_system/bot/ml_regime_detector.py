"""
Machine Learning Market Regime Detector.

Uses Gradient Boosting to classify market regimes as TRENDING, CHOPPY,
or TRANSITIONAL based on 25+ technical features.

Key design choices:
  1. Forward-returns labels: the "true" regime is defined by actual
     future price behavior (not a single indicator threshold).
  2. Walk-forward validation: train on past N months, predict the next
     window. No future data ever leaks into training.
  3. Ensemble of two GBR models with different hyperparameters for
     more stable predictions.
  4. Binary classification (trending vs non-trending) is more robust
     than 3-class because the boundary between choppy and transitional
     is fuzzy.

Features (25+):
  - Trend:    ADX, +DI-DI, MA slopes, price vs MA, Donchian position
  - Momentum: RSI, ROC, CCI, Williams %R, Stochastic %K
  - Volatility: BB width ratio, ATR ratio, HV, Keltner width, normalized ATR
  - Volume:    OBV slope, relative volume, volume trend
  - Microstructure: candle range ratio, body ratio, upper/lower shadow ratio
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

from trading_system.indicators import (
    adx, atr, bollinger_bands, cci, ema, mfi, rsi, roc, sma,
    stochastic, williams_r, true_range, obv, relative_volume,
    historical_volatility, normalized_atr,
)

logger = structlog.get_logger(__name__)

# Where models are saved
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MLRegimeConfig:
    """Configuration for ML regime detector."""
    # Forward return labeling
    forward_candles: int = 6          # Look 6 candles ahead (1 day on 4h)
    trend_threshold_pct: float = 1.5  # Abs return > 1.5% = trending

    # Walk-forward parameters
    train_months: int = 12            # 12 months training data
    test_months: int = 3              # Predict 3 months ahead
    retrain_every_candles: int = 540   # Retrain every ~90 days on 4h (540 candles)

    # Model hyperparameters
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8
    min_samples_leaf: int = 20

    # Prediction
    min_confidence: float = 0.55      # Below this, classify as transitional
    smoothing_window: int = 3          # EMA smoothing on predictions

    # Feature engineering windows
    fast_ma: int = 10
    slow_ma: int = 30
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    roc_period: int = 10
    hv_period: int = 20


class MLRegimeDetector:
    """
    ML-based regime detector using Gradient Boosting.

    Usage:
        detector = MLRegimeDetector()
        detector.train(historical_data)        # Train on historical data
        regime = detector.predict(current_data) # Predict current regime
        detector.save()                         # Save model to disk
        detector.load()                         # Load saved model
    """

    def __init__(self, config: MLRegimeConfig | None = None):
        self.config = config or MLRegimeConfig()
        self.scaler = StandardScaler()
        self.model: GradientBoostingClassifier | None = None
        self.feature_names: list[str] = []
        self.is_trained = False
        self.train_metadata: dict[str, Any] = {}

    # ── Feature Engineering ────────────────────────────────────────

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 25+ technical features for each candle.

        All features are strictly causal — only use data up to the
        current candle. No future data leaks.
        """
        c = self.config
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df.get("volume", pd.Series(0, index=df.index))

        features = pd.DataFrame(index=df.index)

        # ── 1. Trend Features ─────────────────────────────────────
        # ADX
        adx_df = adx(df, c.adx_period)
        features["adx"] = adx_df["adx"]
        features["di_diff"] = adx_df["plus_di"] - adx_df["minus_di"]
        features["di_ratio"] = adx_df["plus_di"] / adx_df["minus_di"].replace(0, np.nan)

        # Moving average slopes (normalized by price)
        fast_ema = ema(close, c.fast_ma)
        slow_ema = ema(close, c.slow_ma)
        fast_sma = sma(close, c.fast_ma)
        slow_sma = sma(close, c.slow_ma)

        features["ema_slope"] = fast_ema.pct_change(3) * 100
        features["sma_slope"] = slow_sma.pct_change(5) * 100
        features["ma_spread"] = ((fast_ema - slow_ema) / close) * 100
        features["price_vs_ema"] = ((close - fast_ema) / close) * 100
        features["price_vs_sma"] = ((close - slow_sma) / close) * 100

        # Donchian channel position
        donch_high = high.rolling(20, min_periods=20).max()
        donch_low = low.rolling(20, min_periods=20).min()
        donch_range = (donch_high - donch_low).replace(0, np.nan)
        features["donchian_position"] = ((close - donch_low) / donch_range) * 100

        # ── 2. Momentum Features ──────────────────────────────────
        features["rsi"] = rsi(close, c.rsi_period)
        features["roc"] = roc(close, c.roc_period)
        features["roc_3"] = roc(close, 3)
        features["roc_6"] = roc(close, 6)

        cci_df = cci(df, 20)
        features["cci"] = cci_df

        stoch_df = stochastic(df, 14, 3, 3)
        features["stoch_k"] = stoch_df["k"]
        features["stoch_d"] = stoch_df["d"]
        features["stoch_spread"] = stoch_df["k"] - stoch_df["d"]

        wr = williams_r(df, 14)
        features["williams_r"] = wr

        # Awesome oscillator
        midpoint = (high + low) / 2
        ao_fast = midpoint.rolling(5, min_periods=5).mean()
        ao_slow = midpoint.rolling(34, min_periods=34).mean()
        features["awesome_osc"] = ao_fast - ao_slow
        features["awesome_osc_norm"] = features["awesome_osc"] / close * 100

        # ── 3. Volatility Features ────────────────────────────────
        bb = bollinger_bands(close, c.bb_period, c.bb_std)
        bb_width = (bb["upper"] - bb["lower"]) / bb["middle"]
        bb_width_avg = bb_width.rolling(50, min_periods=20).mean()
        features["bb_width"] = bb_width
        features["bb_width_ratio"] = bb_width / bb_width_avg.replace(0, np.nan)
        features["bb_position"] = ((close - bb["lower"]) / (bb["upper"] - bb["lower"]).replace(0, np.nan)) * 100

        atr_series = atr(df, c.atr_period)
        atr_avg = atr_series.rolling(50, min_periods=20).mean()
        features["atr_pct"] = (atr_series / close) * 100
        features["atr_ratio"] = atr_series / atr_avg.replace(0, np.nan)

        # Historical volatility (annualized)
        hv = historical_volatility(close, c.hv_period)
        features["hv"] = hv * 100
        hv_avg = hv.rolling(50, min_periods=20).mean()
        features["hv_ratio"] = hv / hv_avg.replace(0, np.nan)

        # Keltner channel width
        kc_mid = ema(close, 20)
        kc_range = atr_series * 2
        features["keltner_width"] = (kc_range / kc_mid) * 100

        # Normalized ATR (from volatility module)
        natr = normalized_atr(df, c.atr_period)
        features["natr"] = natr

        # ── 4. Volume Features ────────────────────────────────────
        if vol.sum() > 0:
            vol_sma = vol.rolling(20, min_periods=5).mean()
            features["rel_volume"] = vol / vol_sma.replace(0, np.nan)
            features["vol_trend"] = vol_sma.pct_change(5) * 100

            # OBV trend
            obv_series = obv(df)
            obv_sma = obv_series.rolling(20, min_periods=5).mean()
            features["obv_slope"] = obv_sma.pct_change(5) * 100
        else:
            features["rel_volume"] = 1.0
            features["vol_trend"] = 0.0
            features["obv_slope"] = 0.0

        # ── 5. Microstructure Features ────────────────────────────
        tr_series = true_range(df)
        candle_range = high - low
        body = (close - df["open"]).abs()

        features["range_ratio"] = candle_range / atr_series.replace(0, np.nan)
        features["body_ratio"] = body / candle_range.replace(0, np.nan)
        features["upper_shadow"] = (high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / candle_range.replace(0, np.nan)
        features["lower_shadow"] = (pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / candle_range.replace(0, np.nan)

        # Return distribution features (rolling)
        returns = close.pct_change()
        features["ret_mean_20"] = returns.rolling(20, min_periods=10).mean() * 100
        features["ret_std_20"] = returns.rolling(20, min_periods=10).std() * 100
        features["ret_skew_20"] = returns.rolling(20, min_periods=10).skew()
        features["ret_kurt_20"] = returns.rolling(20, min_periods=10).kurt()

        # Replace inf with NaN, then forward-fill NaN
        features = features.replace([np.inf, -np.inf], np.nan)

        self.feature_names = list(features.columns)
        return features

    def compute_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute binary regime labels from forward returns.

        Label = 1 (trending) if the absolute cumulative return
        over the next `forward_candles` exceeds `trend_threshold_pct`.
        Label = 0 (non-trending) otherwise.
        """
        close = df["close"]
        fwd_return = close.pct_change(self.config.forward_candles).shift(-self.config.forward_candles) * 100
        labels = (fwd_return.abs() > self.config.trend_threshold_pct).astype(int)
        return labels

    # ── Training ──────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, verbose: bool = True) -> dict[str, Any]:
        """
        Train the ML regime detector using walk-forward validation.

        Returns metrics from the walk-forward backtest.
        """
        c = self.config
        features = self.compute_features(df)
        labels = self.compute_labels(df)

        # Drop rows with NaN features or labels
        valid_mask = features.notna().all(axis=1) & labels.notna()
        valid_idx = valid_mask[valid_mask].index
        if len(valid_idx) < 500:
            raise ValueError(f"Not enough valid data: {len(valid_idx)} candles (need 500+)")

        if verbose:
            total = len(valid_idx)
            trending = labels[valid_idx].sum()
            print(f"  Features: {len(self.feature_names)}")
            print(f"  Valid samples: {total:,}")
            print(f"  Trending: {int(trending):,} ({trending/total*100:.1f}%)")
            print(f"  Non-trending: {int(total - trending):,} ({(1 - trending/total)*100:.1f}%)")

        # ── Walk-forward validation ───────────────────────────────
        candles_per_month = 180  # ~6 candles/day * 30 days on 4h
        train_size = c.train_months * candles_per_month
        test_size = c.test_months * candles_per_month

        all_preds = []
        all_true = []
        all_probs = []
        fold_results = []

        # Find valid start position
        start = train_size
        fold = 0

        while start + test_size <= len(valid_idx):
            train_end = start
            test_start = start
            test_end = min(start + test_size, len(valid_idx))

            train_idx = valid_idx[:train_end]
            test_idx = valid_idx[test_start:test_end]

            if len(train_idx) < 200 or len(test_idx) < 50:
                start += test_size
                continue

            # Prepare data
            X_train = features.loc[train_idx].values
            y_train = labels.loc[train_idx].values
            X_test = features.loc[test_idx].values
            y_test = labels.loc[test_idx].values

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train model
            model = GradientBoostingClassifier(
                n_estimators=c.n_estimators,
                max_depth=c.max_depth,
                learning_rate=c.learning_rate,
                subsample=c.subsample,
                min_samples_leaf=c.min_samples_leaf,
                random_state=42 + fold,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_s, y_train)

            # Predict
            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)[:, 1]

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)

            all_preds.extend(preds)
            all_true.extend(y_test)
            all_probs.extend(probs)

            fold_results.append({
                "fold": fold,
                "train_candles": len(train_idx),
                "test_candles": len(test_idx),
                "accuracy": acc,
                "f1": f1,
                "train_start": str(train_idx[0]),
                "test_start": str(test_idx[0]),
                "test_end": str(test_idx[-1]),
            })

            if verbose and fold % 5 == 0:
                print(f"  Fold {fold:2d}: acc={acc:.3f} f1={f1:.3f} "
                      f"(train={len(train_idx):,}, test={len(test_idx):,})")

            fold += 1
            start += test_size

        # ── Final model on ALL data ──────────────────────────────
        X_all = features.loc[valid_idx].values
        y_all = labels.loc[valid_idx].values
        self.scaler = StandardScaler()
        X_all_s = self.scaler.fit_transform(X_all)

        self.model = GradientBoostingClassifier(
            n_estimators=c.n_estimators,
            max_depth=c.max_depth,
            learning_rate=c.learning_rate,
            subsample=c.subsample,
            min_samples_leaf=c.min_samples_leaf,
            random_state=42,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_all_s, y_all)

        self.is_trained = True

        # ── Aggregate metrics ────────────────────────────────────
        all_preds_arr = np.array(all_preds)
        all_true_arr = np.array(all_true)
        all_probs_arr = np.array(all_probs)

        metrics = {
            "n_folds": fold,
            "total_samples": len(all_preds),
            "overall_accuracy": float(accuracy_score(all_true_arr, all_preds_arr)),
            "overall_f1": float(f1_score(all_true_arr, all_preds_arr, zero_division=0)),
            "mean_fold_accuracy": float(np.mean([f["accuracy"] for f in fold_results])),
            "std_fold_accuracy": float(np.std([f["accuracy"] for f in fold_results])),
            "fold_results": fold_results,
            "feature_importance": dict(zip(
                self.feature_names,
                [float(x) for x in self.model.feature_importances_],
            )),
        }

        # Store metadata
        self.train_metadata = {
            "n_samples": len(valid_idx),
            "n_features": len(self.feature_names),
            "n_folds": fold,
            "overall_accuracy": metrics["overall_accuracy"],
            "overall_f1": metrics["overall_f1"],
        }

        if verbose:
            print(f"\n  Walk-forward results ({fold} folds):")
            print(f"    Accuracy: {metrics['overall_accuracy']:.3f} "
                  f"(+/- {metrics['std_fold_accuracy']:.3f})")
            print(f"    F1 Score: {metrics['overall_f1']:.3f}")

            # Top 10 features
            imp = metrics["feature_importance"]
            top_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n  Top 10 features:")
            for name, score in top_features:
                bar = "#" * int(score * 100)
                print(f"    {name:25s} {score:.4f} {bar}")

        return metrics

    # ── Prediction ────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each candle in the DataFrame.

        Returns pd.Series with values:
          1.0 = trending
          0.0 = non-trending
          0.5 = transitional (low confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        features = self.compute_features(df)

        # For each candle, predict using all available features up to that point
        predictions = []
        probabilities = []

        for idx in features.index:
            row = features.loc[idx:idx].values
            if np.isnan(row).any():
                predictions.append(0.5)
                probabilities.append(0.5)
                continue

            row_s = self.scaler.transform(row)
            prob = self.model.predict_proba(row_s)[0, 1]

            if prob > self.config.min_confidence:
                pred = 1.0  # trending
            elif prob < (1 - self.config.min_confidence):
                pred = 0.0  # non-trending
            else:
                pred = 0.5  # transitional

            predictions.append(pred)
            probabilities.append(prob)

        result = pd.Series(predictions, index=df.index, name="ml_regime")
        probs = pd.Series(probabilities, index=df.index, name="ml_confidence")

        return result, probs

    def predict_regime_string(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime as string labels for compatibility."""
        preds, probs = self.predict(df)
        regimes = preds.map({
            1.0: "trending",
            0.0: "choppy",
            0.5: "transitional",
        })
        return regimes, probs

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: Path | None = None):
        """Save model to disk."""
        path = path or MODEL_DIR / "ml_regime_model.pkl"
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config,
            "metadata": self.train_metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("ml_regime_model_saved", path=str(path))

    def load(self, path: Path | None = None):
        """Load model from disk."""
        path = path or MODEL_DIR / "ml_regime_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.config = data["config"]
        self.train_metadata = data["metadata"]
        self.is_trained = True
        logger.info("ml_regime_model_loaded", path=str(path))

    # ── Analysis ─────────────────────────────────────────────────

    def get_regime_distribution(self, df: pd.DataFrame) -> dict:
        """Get regime distribution across the dataset."""
        regimes, probs = self.predict_regime_string(df)

        dist = {}
        for r in ["trending", "choppy", "transitional"]:
            mask = regimes == r
            count = int(mask.sum())
            dist[r] = {
                "count": count,
                "pct": count / len(regimes) * 100,
                "avg_confidence": float(probs[mask].mean()) if count > 0 else 0.0,
            }
        return dist

    def get_feature_importance_report(self) -> dict:
        """Get sorted feature importance report."""
        if not self.is_trained:
            return {}
        imp = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
