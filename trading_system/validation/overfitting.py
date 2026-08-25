"""
Overfitting detection diagnostics.

Identifies common overfitting patterns in backtest results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class OverfittingDetector:
    """Detect overfitting indicators in backtest results."""

    def analyze(
        self,
        is_results: dict[str, Any] | None = None,
        oos_results: dict[str, Any] | None = None,
        param_stability: dict[str, Any] | None = None,
        full_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive overfitting analysis.

        Returns a dict of overfitting indicators and an overall overfitting score.
        """
        indicators = {}
        warnings = []

        # 1. IS/OOS performance gap
        if is_results and oos_results:
            is_sharpe = is_results.get("sharpe", 0)
            oos_sharpe = oos_results.get("sharpe", 0)

            if is_sharpe > 0:
                degradation = (is_sharpe - oos_sharpe) / is_sharpe
                indicators["is_oos_degradation"] = degradation
                if degradation > 0.5:
                    warnings.append(f"Large IS/OOS degradation: {degradation:.1%}")
            else:
                indicators["is_oos_degradation"] = 0.0

            is_return = is_results.get("total_return", 0)
            oos_return = oos_results.get("total_return", 0)
            if is_return > 0:
                indicators["return_degradation"] = (is_return - oos_return) / is_return

        # 2. Trade count
        if full_results:
            n_trades = full_results.get("total_trades", 0)
            indicators["trade_count"] = n_trades
            if n_trades < 30:
                warnings.append(f"Very few trades: {n_trades}")
            elif n_trades < 100:
                warnings.append(f"Low trade count: {n_trades}")

            # 3. Extremely high Sharpe
            sharpe = full_results.get("sharpe", 0)
            indicators["sharpe"] = sharpe
            if sharpe > 4:
                warnings.append(f"Extremely high Sharpe: {sharpe:.2f}")
            elif sharpe > 3:
                warnings.append(f"Very high Sharpe: {sharpe:.2f}")

            # 4. Win rate with low trade count
            win_rate = full_results.get("win_rate", 0)
            if win_rate > 0.85 and n_trades < 200:
                warnings.append(f"Suspiciously high win rate ({win_rate:.1%}) with {n_trades} trades")

            # 5. Profit concentration
            trades = full_results.get("trades", [])
            if trades and n_trades >= 10:
                pnls = [t["pnl"] for t in trades]
                total_pnl = sum(pnls)
                if total_pnl > 0:
                    # Top 10% of trades contribution
                    sorted_pnls = sorted(pnls, reverse=True)
                    top_10_count = max(1, n_trades // 10)
                    top_10_pnl = sum(sorted_pnls[:top_10_count])
                    concentration = top_10_pnl / total_pnl
                    indicators["profit_concentration"] = concentration
                    if concentration > 0.5:
                        warnings.append(
                            f"Profit concentrated in top 10% of trades: {concentration:.1%}"
                        )

            # 6. Drawdown analysis
            max_dd = full_results.get("max_drawdown", 0)
            if max_dd > 0.5:
                warnings.append(f"Extreme drawdown: {max_dd:.1%}")
            indicators["max_drawdown"] = max_dd

            # 7. Return to DD ratio
            total_return = full_results.get("total_return", 0)
            if max_dd > 0:
                indicators["return_to_dd_ratio"] = total_return / max_dd
                if indicators["return_to_dd_ratio"] < 1.0:
                    warnings.append(f"Poor return/drawdown ratio: {indicators['return_to_dd_ratio']:.2f}")

        # 8. Parameter stability
        if param_stability:
            stab_score = param_stability.get("stability_score", 0)
            indicators["param_stability_score"] = stab_score
            if stab_score < 0.3:
                warnings.append(f"Low parameter stability: {stab_score:.2f}")

        # Overall overfitting score (0 = likely overfit, 1 = likely robust)
        overfitting_score = self._compute_overfitting_score(indicators)

        return {
            "indicators": indicators,
            "warnings": warnings,
            "n_warnings": len(warnings),
            "overfitting_score": overfitting_score,
            "severity": self._severity(overfitting_score),
        }

    def _compute_overfitting_score(self, indicators: dict[str, float]) -> float:
        """Compute overall overfitting score (1 = robust, 0 = likely overfit)."""
        score = 1.0

        # IS/OOS degradation
        degradation = indicators.get("is_oos_degradation", 0)
        if degradation > 0.7:
            score -= 0.3
        elif degradation > 0.5:
            score -= 0.2
        elif degradation > 0.3:
            score -= 0.1

        # Trade count
        n_trades = indicators.get("trade_count", 100)
        if n_trades < 30:
            score -= 0.2
        elif n_trades < 100:
            score -= 0.1

        # Extreme Sharpe
        sharpe = indicators.get("sharpe", 0)
        if sharpe > 4:
            score -= 0.2
        elif sharpe > 3:
            score -= 0.1

        # Profit concentration
        concentration = indicators.get("profit_concentration", 0)
        if concentration > 0.6:
            score -= 0.15
        elif concentration > 0.4:
            score -= 0.05

        # Parameter stability
        stab = indicators.get("param_stability_score", 0.5)
        if stab < 0.3:
            score -= 0.15
        elif stab < 0.5:
            score -= 0.05

        # Return/DD ratio
        rdd = indicators.get("return_to_dd_ratio", 1.0)
        if rdd < 0.5:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _severity(self, score: float) -> str:
        if score >= 0.7:
            return "low"
        elif score >= 0.4:
            return "moderate"
        else:
            return "high"
