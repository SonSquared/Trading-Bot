"""
Execution models for realistic trade simulation.

Ensures no look-ahead bias — signals are generated at close of candle N,
execution happens at open of candle N+1 or later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_system.backtester.slippage import SlippageModel


@dataclass
class ExecutionConfig:
    """Execution model configuration."""
    model: str = "next_open"  # "next_open", "next_open_delay"
    delay_candles: int = 0


class NextOpenExecution:
    """Execute at the next candle's open (after signal)."""

    def __init__(
        self,
        slippage_model: SlippageModel,
        delay_candles: int = 0,
    ):
        self.slippage_model = slippage_model
        self.delay_candles = delay_candles

    def get_entry_price(
        self,
        signals: pd.Series,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        atr_series: pd.Series | None = None,
    ) -> pd.Series:
        """
        Get execution entry prices for each signal.

        Signals at time T execute at open of T+1+delay.
        """
        # Shift signals forward by 1 + delay
        exec_signals = signals.shift(1 + self.delay_candles)

        prices = pd.Series(np.nan, index=opens.index)

        for i in range(len(opens)):
            if pd.isna(exec_signals.iloc[i]) or exec_signals.iloc[i] == 0:
                continue

            is_long = exec_signals.iloc[i] > 0
            candle = opens.iloc[i]

            # Apply slippage
            adjusted = self.slippage_model.calculate_slippage(
                price=candle,
                position_size=1.0,
                is_long=is_long,
                candle_data=pd.Series({"high": highs.iloc[i], "low": lows.iloc[i], "close": closes.iloc[i]}) if i < len(highs) else None,
                atr_series=atr_series,
            )
            prices.iloc[i] = adjusted

        return prices

    def get_exit_price(
        self,
        signals: pd.Series,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        atr_series: pd.Series | None = None,
    ) -> pd.Series:
        """Get execution exit prices."""
        prices = pd.Series(np.nan, index=opens.index)

        # Detect exits: signal changes from non-zero to zero or flips direction
        prev_signal = signals.shift(1)
        is_exit = ((signals == 0) & (prev_signal != 0)) | \
                  ((signals > 0) & (prev_signal < 0)) | \
                  ((signals < 0) & (prev_signal > 0))

        for i in range(len(opens)):
            if not is_exit.iloc[i]:
                continue

            # Exit at next candle open
            exec_idx = i + 1 + self.delay_candles
            if exec_idx >= len(opens):
                continue

            was_long = prev_signal.iloc[i] > 0
            candle = opens.iloc[exec_idx]

            adjusted = self.slippage_model.calculate_slippage(
                price=candle,
                position_size=1.0,
                is_long=not was_long,  # Closing opposite side
                candle_data=pd.Series({
                    "high": highs.iloc[exec_idx],
                    "low": lows.iloc[exec_idx],
                    "close": closes.iloc[exec_idx],
                }) if exec_idx < len(highs) else None,
                atr_series=atr_series,
            )
            prices.iloc[exec_idx] = adjusted

        return prices


def create_execution_model(
    config: ExecutionConfig,
    slippage_model: SlippageModel,
) -> NextOpenExecution:
    """Factory function for execution models."""
    return NextOpenExecution(
        slippage_model=slippage_model,
        delay_candles=config.delay_candles,
    )
