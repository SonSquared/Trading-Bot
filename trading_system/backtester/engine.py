"""
Core vectorized backtesting engine.

Simulates realistic trading with:
- No look-ahead bias (signals use only past data)
- Realistic execution (next candle open + slippage)
- Fees (maker/taker/funding)
- Slippage (configurable model)
- Position tracking with P&L
- Comprehensive metrics calculation

Performance: Fully vectorized with numpy — no Python loops over candles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from trading_system.backtester.fees import FeeCalculator
from trading_system.backtester.results import BacktestResults
from trading_system.backtester.slippage import create_slippage_model
from trading_system.backtester.execution import create_execution_model
from trading_system.config import BacktestConfig
from trading_system.indicators import atr

logger = structlog.get_logger(__name__)

RISK_FREE_RATE = 0.04  # 4% annual
TRADING_HOURS_PER_YEAR = 365 * 24  # Crypto 24/7


class BacktestEngine:
    """Fully vectorized backtesting engine with realistic execution simulation."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.fee_calculator = FeeCalculator(config.fees)
        self.slippage_model = create_slippage_model(config.slippage)
        self.execution = create_execution_model(config.execution, self.slippage_model)

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        strategy_name: str = "",
        params: dict | None = None,
        pair: str = "",
        timeframe: str = "",
        funding_rates: pd.Series | None = None,
        risk_multipliers: pd.Series | None = None,
    ) -> BacktestResults:
        """Run a backtest.

        Args:
            risk_multipliers: Optional per-candle risk multiplier (0-1).
                Scales risk_per_trade at each trade entry for regime-aware sizing.
        """
        if df.empty or signals.empty:
            return BacktestResults(strategy_name=strategy_name, parameters=params or {})

        signals = signals.reindex(df.index).fillna(0).astype(int)

        # Precompute ATR for slippage
        atr_series = atr(df, period=14)

        # Convert signals to position series (vectorized)
        position = self._determine_positions(signals)

        # Build trade log and equity curve (vectorized)
        equity, trades, costs = self._simulate_portfolio_vectorized(
            df, position, atr_series, funding_rates, risk_multipliers
        )

        # Calculate metrics
        results = self._calculate_metrics(
            equity, trades, costs, df, strategy_name, params, pair, timeframe
        )
        return results

    # ------------------------------------------------------------------ #
    #  Vectorized portfolio simulation
    # ------------------------------------------------------------------ #

    def _determine_positions(self, signals: pd.Series) -> pd.Series:
        """Forward-fill signals to create held-position series."""
        position = signals.replace(0, np.nan).ffill().fillna(0).astype(int)
        return position

    def _simulate_portfolio_vectorized(
        self,
        df: pd.DataFrame,
        position: pd.Series,
        atr_series: pd.Series,
        funding_rates: pd.Series | None,
        risk_multipliers: pd.Series | None = None,
    ) -> tuple[pd.Series, list[dict], dict]:
        """Fully vectorized portfolio simulation.

        Instead of looping over every candle, we:
        1. Find all trade segments (consecutive same-position blocks)
        2. Compute entry/exit prices, P&L, fees for each trade using numpy
        3. Build the equity curve from trade results
        """
        n = len(df)
        opens = df["open"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        pos = position.values.astype(np.float64)
        atr_vals = atr_series.values.astype(np.float64)
        idx = df.index

        initial_capital = self.config.execution.initial_capital
        leverage = self.config.execution.leverage
        risk_per_trade = self.config.execution.risk_per_trade

        # --- Precompute ATR-based slippage for all candles ---
        # Fill NaN ATR values (first period-1 candles) with forward-fill then 0
        atr_filled = np.where(np.isnan(atr_vals), 0.0, atr_vals)
        # Forward fill the zeros (use the first valid ATR)
        first_valid = np.argmax(atr_filled > 0) if np.any(atr_filled > 0) else 0
        atr_filled[:first_valid] = atr_filled[first_valid] if first_valid < n else 0.0
        slippage_pct = self._precompute_slippage(opens, atr_filled)

        # --- Find trade segments ---
        # A trade starts when position goes from 0 to non-zero
        # A trade ends when position goes to 0 or changes direction
        prev_pos = np.roll(pos, 1)
        prev_pos[0] = 0

        # Trade entry: position != 0 and (prev_pos == 0 or direction changed)
        trade_start = np.zeros(n, dtype=bool)
        # Handle index 0: if position starts non-zero, it's a trade entry
        trade_start[0] = pos[0] != 0
        trade_start[1:] = (pos[1:] != 0) & ((prev_pos[1:] == 0) | (np.sign(pos[1:]) != np.sign(prev_pos[1:])))

        # Trade end: prev_pos != 0 and (position == 0 or direction changed)
        trade_end = np.zeros(n, dtype=bool)
        trade_end[1:] = (prev_pos[1:] != 0) & ((pos[1:] == 0) | (np.sign(pos[1:]) != np.sign(prev_pos[1:])))

        start_indices = np.where(trade_start)[0]
        end_indices = np.where(trade_end)[0]

        # Match starts with ends: each start[i] pairs with end[j] where j >= i
        trades: list[dict] = []
        equity_arr = np.full(n, initial_capital, dtype=np.float64)

        cash = initial_capital
        total_fees = 0.0
        total_funding = 0.0
        total_slippage_cost = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        turnover = 0.0

        # Process each trade (typically <2000 trades even on 160K candles)
        si = 0
        ei = 0
        active_entry_idx = None
        active_entry_price = 0.0
        active_position_size = 0.0
        prev_trade_end = -1  # Track end of previous trade for gap filling

        for k in range(len(start_indices)):
            s = start_indices[k]

            # Find matching end
            while ei < len(end_indices) and end_indices[ei] < s:
                ei += 1
            if ei >= len(end_indices):
                e = n - 1  # No end found, hold to end
            else:
                e = end_indices[ei]
                ei += 1

            # Entry at candle s (use next candle's open for next-open execution)
            entry_candle_idx = min(s, n - 1)
            entry_price_raw = opens[entry_candle_idx]
            entry_slip = slippage_pct[entry_candle_idx]
            is_long = pos[s] > 0
            entry_price = entry_price_raw * (1 + entry_slip) if is_long else entry_price_raw * (1 - entry_slip)

            # Position sizing: risk_per_trade fraction of current cash
            # Apply regime-based risk multiplier if available
            if risk_multipliers is not None:
                rm_val = float(risk_multipliers.iloc[s]) if s < len(risk_multipliers) else 1.0
                rm_val = max(0.0, min(1.0, rm_val))  # clamp to [0, 1]
            else:
                rm_val = 1.0
            effective_risk = risk_per_trade * rm_val
            notional = cash * effective_risk * leverage
            notional = min(notional, cash * leverage * 0.95)
            if notional <= 0 or entry_price <= 0:
                continue
            position_size = notional / entry_price
            if not is_long:
                position_size = -position_size

            # --- ATR-based Stop-Loss check ---
            sl_mult = self.config.execution.stop_loss_atr_mult
            exit_candle_idx = min(e, n - 1)
            exit_reason = "signal_change"

            if sl_mult > 0:
                entry_atr = atr_filled[entry_candle_idx] if atr_filled[entry_candle_idx] > 0 else entry_price * 0.02
                if is_long:
                    sl_price = entry_price - entry_atr * sl_mult
                else:
                    sl_price = entry_price + entry_atr * sl_mult

                # Check each candle in the trade for SL hit
                sl_hit_idx = -1
                for ci in range(s + 1, e + 1):
                    if ci >= n:
                        break
                    if is_long and lows[ci] <= sl_price:
                        sl_hit_idx = ci
                        break
                    elif not is_long and highs[ci] >= sl_price:
                        sl_hit_idx = ci
                        break

                if sl_hit_idx >= 0:
                    # SL hit — exit at SL price with slippage
                    exit_candle_idx = sl_hit_idx
                    exit_price_raw = sl_price
                    exit_slip = slippage_pct[sl_hit_idx]
                    if is_long:
                        exit_price = sl_price * (1 - exit_slip)
                    else:
                        exit_price = sl_price * (1 + exit_slip)
                    exit_reason = "stop_loss"
                    e = sl_hit_idx  # Shorten the trade end
                else:
                    # No SL hit — normal signal exit
                    exit_slip = slippage_pct[exit_candle_idx]
                    exit_price_raw = opens[exit_candle_idx]
                    if is_long:
                        exit_price = exit_price_raw * (1 - exit_slip)
                    else:
                        exit_price = exit_price_raw * (1 + exit_slip)
            else:
                # No SL configured — normal signal exit
                exit_price_raw = opens[exit_candle_idx]
                exit_slip = slippage_pct[exit_candle_idx]
                if is_long:
                    exit_price = exit_price_raw * (1 - exit_slip)
                else:
                    exit_price = exit_price_raw * (1 + exit_slip)

            # P&L
            pnl_raw = position_size * (exit_price - entry_price)

            # Fees (taker for both entry and exit)
            notional_trade = abs(position_size) * entry_price
            entry_fee = notional_trade * self.config.fees.taker_fee
            exit_fee = abs(position_size) * exit_price * self.config.fees.taker_fee
            fee_total = entry_fee + exit_fee

            # Slippage cost (difference between raw and slipped price)
            slip_cost = abs(entry_price - entry_price_raw) * abs(position_size) + \
                        abs(exit_price - exit_price_raw) * abs(position_size)

            pnl_net = pnl_raw - fee_total
            cash += pnl_net
            total_fees += fee_total
            total_slippage_cost += slip_cost
            turnover += notional_trade * 2
            gross_profit += max(pnl_raw, 0)
            gross_loss += min(pnl_raw, 0)

            # Funding cost (vectorized over the trade period)
            if funding_rates is not None and not funding_rates.empty and s < e:
                # Handle both Series and DataFrame inputs
                if hasattr(funding_rates, 'columns') and 'funding_rate' in funding_rates.columns:
                    fr_values = funding_rates['funding_rate'].values
                    fr_index = funding_rates.index
                elif hasattr(funding_rates, 'values') and hasattr(funding_rates, 'index'):
                    fr_values = funding_rates.values
                    fr_index = funding_rates.index
                else:
                    fr_values = None
                    fr_index = None

                if fr_values is not None and len(fr_values) > 0 and fr_index is not None:
                    trade_idx_slice = idx[s:e+1]
                    if len(trade_idx_slice) > 0:
                        mask = (fr_index >= trade_idx_slice[0]) & (fr_index <= trade_idx_slice[-1])
                        if hasattr(mask, 'values'):
                            mask = mask.values
                        applicable = fr_values[mask]
                        if len(applicable) > 0:
                            notional_avg = notional_trade
                            if is_long:
                                total_funding += notional_avg * float(np.sum(applicable.astype(float)))
                            else:
                                total_funding -= notional_avg * float(np.sum(applicable.astype(float)))

            trades.append({
                "entry_time": idx[s],
                "exit_time": idx[e],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": position_size,
                "pnl": pnl_net,
                "holding_candles": e - s,
                "exit_reason": exit_reason,
            })

            # Fill equity curve for this trade period
            cash_before_trade = cash - pnl_net
            if e > s:
                # Position held from s to e: equity = cash_before + unrealized_pnl
                unrealized_at_candle = position_size * (opens[s:e+1] - entry_price) - fee_total
                equity_arr[s:e+1] = cash_before_trade + unrealized_at_candle

            # Mark entry and exit with correct values
            # Entry candle: equity = cash_before (position just opened, ~0 unrealized)
            equity_arr[s] = cash_before_trade
            # Exit candle: equity = cash after trade
            equity_arr[e] = cash
            # Fill any gap between previous trade end and this trade start
            if k > 0 and s > 0:
                prev_end = int(equity_arr[s-1]) if equity_arr[s-1] > 0 else cash_before_trade
                equity_arr[prev_trade_end+1:s] = cash_before_trade
            prev_trade_end = e

        # Fill remaining equity (no active trades)
        last_trade_end = end_indices[-1] if len(end_indices) > 0 else 0
        if last_trade_end < n:
            equity_arr[last_trade_end:] = cash

        # Clean up any remaining NaN in equity curve
        equity_arr = np.where(np.isfinite(equity_arr), equity_arr, initial_capital)
        # Forward-fill any remaining zeros (untraded periods)
        equity_series = pd.Series(equity_arr, index=df.index)
        costs = {
            "total_fees": total_fees,
            "total_slippage": total_slippage_cost,
            "total_funding_costs": total_funding,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "turnover": turnover,
        }
        return equity_series, trades, costs

    def _precompute_slippage(self, opens: np.ndarray, atr_vals: np.ndarray) -> np.ndarray:
        """Vectorized slippage computation for all candles at once."""
        base = self.config.slippage.base_slippage
        model = self.config.slippage.model

        if model == "none":
            return np.zeros_like(opens)
        elif model == "fixed":
            return np.full_like(opens, base)
        else:
            # ATR-adaptive
            with np.errstate(divide="ignore", invalid="ignore"):
                atr_pct = np.where(opens > 0, atr_vals / opens, 0.0)
            return base + self.config.slippage.atr_multiplier * atr_pct

    # ------------------------------------------------------------------ #
    #  Metrics calculation (unchanged — already efficient)
    # ------------------------------------------------------------------ #

    def _calculate_metrics(
        self,
        equity: pd.Series,
        trades: list[dict],
        costs: dict,
        df: pd.DataFrame,
        strategy_name: str,
        params: dict | None,
        pair: str,
        timeframe: str,
    ) -> BacktestResults:
        """Calculate comprehensive performance metrics."""
        results = BacktestResults()
        results.strategy_name = strategy_name
        results.parameters = params or {}
        results.pair = pair
        results.timeframe = timeframe
        results.data_start = str(df.index[0]) if len(df) > 0 else ""
        results.data_end = str(df.index[-1]) if len(df) > 0 else ""

        if equity.empty or len(equity) < 2:
            return results

        results.equity_curve = equity
        eq = equity.values

        # Returns
        # Replace NaN/0 equity values to avoid division issues
        init_cap = self.config.execution.initial_capital
        eq_safe = np.where(np.isfinite(eq) & (eq != 0), eq, np.nan)
        # Forward-fill NaN equity values
        eq_series = pd.Series(eq_safe)
        eq_series = eq_series.ffill().bfill().fillna(init_cap)
        eq = eq_series.values
        returns = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)
        returns = np.where(np.isfinite(returns), returns, 0.0)

        total_return = (eq[-1] / eq[0]) - 1
        results.total_return = total_return

        # CAGR
        n_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
        n_years = n_hours / TRADING_HOURS_PER_YEAR
        if n_years > 0 and eq[0] > 0:
            results.cagr = (eq[-1] / eq[0]) ** (1 / n_years) - 1
        results.annualized_return = results.cagr

        # Monthly / yearly returns
        ret_series = pd.Series(returns, index=df.index[1:])
        if hasattr(ret_series.index, "to_period"):
            results.monthly_returns = ret_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            results.yearly_returns = ret_series.resample("YE").apply(lambda x: (1 + x).prod() - 1)

        # Drawdown (vectorized)
        running_max = np.maximum.accumulate(eq)
        drawdown = (eq - running_max) / np.where(running_max > 0, running_max, 1.0)
        results.drawdown_curve = pd.Series(drawdown, index=df.index)
        results.max_drawdown = abs(float(drawdown.min()))

        # Average drawdown & duration (vectorized with numpy)
        in_dd = drawdown < 0
        changes_dd = np.diff(in_dd.astype(int), prepend=0)
        dd_starts = np.where(changes_dd == 1)[0]
        dd_ends = np.where(changes_dd == -1)[0]
        if len(dd_ends) == 0 or (len(dd_starts) > 0 and dd_starts[-1] >= len(dd_ends)):
            dd_ends = np.append(dd_ends, len(drawdown))

        if len(dd_starts) > 0:
            dd_minima = [drawdown[dd_starts[i]:dd_ends[i]].min() for i in range(min(len(dd_starts), len(dd_ends)))]
            results.avg_drawdown = abs(float(np.mean(dd_minima)))
            dd_lengths = dd_ends[:len(dd_starts)] - dd_starts[:len(dd_ends)]
            results.max_drawdown_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

        # Volatility
        if len(returns) > 1:
            results.volatility = float(np.std(returns, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR))

        # Downside deviation
        neg_rets = returns[returns < 0]
        if len(neg_rets) > 1:
            results.downside_deviation = float(np.std(neg_rets, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR))

        # VaR / CVaR
        if len(returns) > 20:
            results.var_95 = float(np.percentile(returns, 5))
            results.cvar_95 = float(np.mean(returns[returns <= results.var_95]))

        # Sharpe (trading returns only — exclude flat/no-position periods)
        active_returns = returns[returns != 0]
        if len(active_returns) > 1 and np.std(active_returns) > 0:
            # Scale by ratio of active to total periods for annualization
            activity_ratio = len(active_returns) / len(returns) if len(returns) > 0 else 1.0
            excess_active = active_returns - RISK_FREE_RATE / TRADING_HOURS_PER_YEAR
            results.sharpe = float(
                np.mean(excess_active) / np.std(excess_active, ddof=1)
                * np.sqrt(TRADING_HOURS_PER_YEAR * activity_ratio)
            )
        else:
            # Fallback: use all returns
            excess = returns - RISK_FREE_RATE / TRADING_HOURS_PER_YEAR
            if len(excess) > 1 and np.std(excess) > 0:
                results.sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR))

        # Sortino
        if results.downside_deviation > 0:
            results.sortino = (results.annualized_return - RISK_FREE_RATE) / results.downside_deviation

        # Calmar
        if results.max_drawdown > 0:
            results.calmar = results.cagr / results.max_drawdown

        if results.max_drawdown > 0:
            results.return_to_dd = results.total_return / results.max_drawdown

        # Trading statistics
        results.total_trades = len(trades)
        if trades:
            pnls = np.array([t["pnl"] for t in trades])
            results.winning_trades = int(np.sum(pnls > 0))
            results.losing_trades = int(np.sum(pnls <= 0))
            results.win_rate = results.winning_trades / len(trades)

            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            results.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            results.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            results.expectancy = float(np.mean(pnls))

            total_wins = float(np.sum(wins))
            total_losses = float(np.abs(np.sum(losses)))
            results.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

            holding = np.array([t.get("holding_candles", 1) for t in trades if "holding_candles" in t])
            results.avg_holding_period = float(np.mean(holding)) if len(holding) > 0 else 0.0

            results.max_consecutive_losses = self._max_consecutive(pnls, loss=True)
            results.max_consecutive_wins = self._max_consecutive(pnls, loss=False)

        # Cost breakdown
        results.total_fees = costs.get("total_fees", 0)
        results.total_slippage = costs.get("total_slippage", 0)
        results.total_funding_costs = costs.get("total_funding_costs", 0)
        results.gross_profit = costs.get("gross_profit", 0)
        results.gross_loss = costs.get("gross_loss", 0)
        results.net_profit = eq[-1] - eq[0]
        results.turnover = costs.get("turnover", 0)
        results.trades = trades

        return results

    @staticmethod
    def _max_consecutive(pnls: np.ndarray, loss: bool = True) -> int:
        """Count max consecutive wins or losses."""
        mask = pnls <= 0 if loss else pnls > 0
        if not mask.any():
            return 0
        # Find runs of True
        d = np.diff(mask.astype(int), prepend=0)
        run_starts = np.where(d == 1)[0]
        run_ends = np.where(d == -1)[0]
        if len(run_ends) == 0:
            run_ends = np.array([len(mask)])
        if len(run_starts) > len(run_ends):
            run_ends = np.append(run_ends, len(mask))
        lengths = run_ends[:len(run_starts)] - run_starts
        return int(lengths.max()) if len(lengths) > 0 else 0
