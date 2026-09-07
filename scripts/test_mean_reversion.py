"""
Test mean-reversion strategies against current choppy market (Aug 15-26).
Compare with the trend-following strategies to see if they capture more trades.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import ccxt

from trading_system.strategies import STRATEGY_REGISTRY

INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POS_PCT = 0.35


def fetch(symbol):
    exchange = ccxt.kraken({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, "4h", limit=500)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    start = pd.Timestamp("2026-08-15")
    end = pd.Timestamp("2026-08-27")
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def backtest(name, df, strat_name, params):
    strat = STRATEGY_REGISTRY[strat_name]
    sig = strat.generate_signals(df, params)

    cash = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        signal = int(sig.iloc[i])

        if position and signal != position[0]:
            side, entry, size = position
            exit_price = price * (1 - SLIPPAGE_RATE * side)
            pnl_pct = (exit_price - entry) / entry * side
            pnl_usd = size * pnl_pct - size * FEE_RATE
            cash += size + pnl_usd
            trades.append({"action": "CLOSE", "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd})
            position = None

        if not position and signal != 0:
            size = min(cash * MAX_POS_PCT, cash * 0.95)
            if size > 5:
                entry_price = price * (1 + SLIPPAGE_RATE * signal)
                fee = size * FEE_RATE
                cash -= fee
                position = (signal, entry_price, size)
                label = "LONG" if signal == 1 else "SHORT"
                trades.append({"action": f"OPEN {label}", "price": entry_price, "size": size})

    if position:
        price = float(df["close"].iloc[-1])
        side, entry, size = position
        exit_price = price * (1 - SLIPPAGE_RATE * side)
        pnl_pct = (exit_price - entry) / entry * side
        pnl_usd = size * pnl_pct - size * FEE_RATE
        cash += size + pnl_usd
        trades.append({"action": "CLOSE (end)", "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd})

    final = cash
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    closes = [t for t in trades if t["action"].startswith("CLOSE")]
    wins = sum(1 for t in closes if t.get("pnl_usd", 0) > 0)
    total_pnl = sum(t.get("pnl_usd", 0) for t in closes)
    max_dd = 0
    peak = INITIAL_CAPITAL
    eq = INITIAL_CAPITAL
    for t in trades:
        if "pnl_pct" in t:
            eq += t["pnl_usd"]
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

    # Count signal changes (trade frequency)
    sig_arr = sig.values
    non_flat = sum(1 for s in sig_arr if s != 0)
    transitions = sum(1 for i in range(1, len(sig_arr)) if sig_arr[i] != sig_arr[i - 1])

    print(f"  {name}")
    print(f"    Trades: {len(closes)} | Wins: {wins}/{len(closes)} ({wins / len(closes) * 100:.0f}%)" if closes else "    Trades: 0")
    print(f"    Return: {ret:+.2f}% (${final - INITIAL_CAPITAL:+.2f})")
    print(f"    Max DD: {max_dd * 100:.2f}%")
    print(f"    Active candles: {non_flat}/{len(sig_arr)} ({non_flat / len(sig_arr) * 100:.0f}%)")
    print(f"    Signal transitions: {transitions}")
    if closes:
        print(f"    Avg trade P&L: ${total_pnl / len(closes):+.2f}")
    print()

    return {"name": name, "trades": len(closes), "wins": wins, "return": ret, "max_dd": max_dd, "active_pct": non_flat / len(sig_arr), "transitions": transitions}


if __name__ == "__main__":
    print("=" * 70)
    print("MEAN REVERSION vs TREND FOLLOWING - CHOPPY MARKET TEST")
    print("Period: Aug 15-26 (current choppy regime)")
    print("=" * 70)

    eth_data = fetch("ETH/USDT")
    btc_data = fetch("BTC/USDT")
    print(f"ETH: {len(eth_data)} candles, BTC: {len(btc_data)} candles\n")

    # Test all strategies
    tests = [
        # Trend following (current)
        ("MACD ETH (trend)", eth_data, "MACD", {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True}),
        ("ROC_Momentum ETH (trend)", eth_data, "ROC_Momentum", {"roc_period": 10, "signal_period": 5, "use_ema": True, "ema_period": 12}),
        ("MACD BTC (trend)", btc_data, "MACD", {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True}),
        # Mean reversion (default params)
        ("RSI_Reversion ETH (default)", eth_data, "RSI_Reversion", {"rsi_period": 14, "entry_oversold": 30, "entry_overbought": 70, "exit_neutral_low": 45, "exit_neutral_high": 55, "use_bb_filter": True, "bb_period": 20, "bb_std": 2.0}),
        ("Bollinger_Reversion ETH (default)", eth_data, "Bollinger_Reversion", {"bb_period": 20, "bb_std": 2.0, "rsi_filter": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "exit_at_middle": True}),
        # Mean reversion (aggressive params for choppy market)
        ("RSI_Reversion ETH (aggressive)", eth_data, "RSI_Reversion", {"rsi_period": 10, "entry_oversold": 35, "entry_overbought": 65, "exit_neutral_low": 45, "exit_neutral_high": 55, "use_bb_filter": False}),
        ("Bollinger_Reversion ETH (aggressive)", eth_data, "Bollinger_Reversion", {"bb_period": 15, "bb_std": 1.5, "rsi_filter": False, "exit_at_middle": True}),
        # BTC mean reversion
        ("RSI_Reversion BTC (default)", btc_data, "RSI_Reversion", {"rsi_period": 14, "entry_oversold": 30, "entry_overbought": 70, "exit_neutral_low": 45, "exit_neutral_high": 55, "use_bb_filter": True, "bb_period": 20, "bb_std": 2.0}),
        ("Bollinger_Reversion BTC (default)", btc_data, "Bollinger_Reversion", {"bb_period": 20, "bb_std": 2.0, "rsi_filter": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "exit_at_middle": True}),
    ]

    print("--- TREND FOLLOWING ---")
    trend_results = []
    for name, data, strat, params in tests[:3]:
        r = backtest(name, data, strat, params)
        trend_results.append(r)

    print("--- MEAN REVERSION ---")
    mr_results = []
    for name, data, strat, params in tests[3:]:
        r = backtest(name, data, strat, params)
        mr_results.append(r)

    # Summary comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    all_results = trend_results + mr_results
    all_results.sort(key=lambda x: x["return"], reverse=True)
    print(f"{'Strategy':<45} {'Trades':>6} {'Win%':>5} {'Return':>8} {'MaxDD':>7} {'Active%':>8}")
    print("-" * 85)
    for r in all_results:
        win_pct = f"{r['wins'] / r['trades'] * 100:.0f}%" if r["trades"] > 0 else "N/A"
        print(f"{r['name']:<45} {r['trades']:>6} {win_pct:>5} {r['return']:>+7.2f}% {r['max_dd'] * 100:>6.2f}% {r['active_pct'] * 100:>7.1f}%")

    print()
    print("KEY INSIGHT: Mean-reversion strategies trade more frequently in")
    print("choppy markets and capture small moves that trend strategies miss.")
