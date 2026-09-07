"""
Backtest the Aug 15-26 rally period with all 3 strategies + portfolio simulation.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import ccxt
import time

from trading_system.strategies import STRATEGY_REGISTRY

INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POS_PCT = 0.35

STRATEGIES = {
    "MACD ETH": ("ETH/USDT", "MACD", {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True}, 0.41),
    "ROC_Momentum ETH": ("ETH/USDT", "ROC_Momentum", {"roc_period": 10, "signal_period": 5, "use_ema": True, "ema_period": 12}, 0.17),
    "MACD BTC": ("BTC/USDT", "MACD", {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True}, 0.43),
}


def fetch(symbol):
    exchange = ccxt.kraken({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, "4h", limit=500)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    start = pd.Timestamp("2026-08-15")
    end = pd.Timestamp("2026-08-27")
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def backtest_strategy(name, df, strat_name, params, weight):
    strat = STRATEGY_REGISTRY[strat_name]
    sig = strat.generate_signals(df, params)

    cash = INITIAL_CAPITAL
    position = None  # (side, entry_price, size_usd)
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
            trades.append({"action": "CLOSE", "entry": entry, "exit": exit_price, "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd})
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
        trades.append({"action": "CLOSE (end)", "entry": entry, "exit": exit_price, "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd})

    final_equity = cash
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    closes = [t for t in trades if t["action"].startswith("CLOSE")]
    wins = sum(1 for t in closes if t.get("pnl_usd", 0) > 0)

    print(f"--- {name} ---")
    print(f"  Period: Aug 15-26 ({len(df)} candles)")
    print(f"  Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    print(f"  Trades: {len(trades)} ({len(closes)} closed)")
    if closes:
        print(f"  Wins: {wins}/{len(closes)} ({wins / len(closes) * 100:.0f}%)")
    print(f"  Final equity: ${final_equity:.2f}")
    print(f"  Return: {total_return:+.2f}% (${final_equity - INITIAL_CAPITAL:+.2f})")
    if trades:
        print("  Trade details:")
        for t in trades:
            if "pnl_pct" in t:
                print(f"    {t['action']}: {t['pnl_pct']:+.2f}% (${t['pnl_usd']:+.2f})")
            else:
                print(f"    {t['action']}: ${t['price']:,.2f} size=${t['size']:.2f}")
    print()
    return trades, final_equity


def backtest_portfolio(eth_data, btc_data):
    """Portfolio-level simulation with weighted aggregation."""
    macd_eth_sig = STRATEGY_REGISTRY["MACD"].generate_signals(eth_data, STRATEGIES["MACD ETH"][2])
    roc_eth_sig = STRATEGY_REGISTRY["ROC_Momentum"].generate_signals(eth_data, STRATEGIES["ROC_Momentum ETH"][2])
    macd_btc_sig = STRATEGY_REGISTRY["MACD"].generate_signals(btc_data, STRATEGIES["MACD BTC"][2])

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []

    for i in range(len(eth_data)):
        eth_price = float(eth_data["close"].iloc[i])
        btc_price = float(btc_data["close"].iloc[i])

        eth_score = (int(macd_eth_sig.iloc[i]) * 0.41 + int(roc_eth_sig.iloc[i]) * 0.17) / (0.41 + 0.17)
        eth_signal = 1 if eth_score > 0.3 else (-1 if eth_score < -0.3 else 0)
        btc_signal = int(macd_btc_sig.iloc[i])

        for pair, signal, price in [("ETH_USDT_USDT", eth_signal, eth_price), ("BTC_USDT_USDT", btc_signal, btc_price)]:
            current = positions.get(pair)
            current_side = current["side"] if current else 0

            if signal != current_side:
                if current_side != 0:
                    exit_p = price * (1 - SLIPPAGE_RATE * current_side)
                    pnl_pct = (exit_p - current["entry"]) / current["entry"] * current_side
                    pnl_usd = current["size"] * pnl_pct - current["size"] * FEE_RATE
                    cash += current["size"] + pnl_usd
                    trades.append({"pair": pair, "action": "CLOSE", "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd})
                    del positions[pair]

                if signal != 0:
                    size = min(cash * MAX_POS_PCT, cash * 0.95)
                    if size > 5:
                        entry_p = price * (1 + SLIPPAGE_RATE * signal)
                        fee = size * FEE_RATE
                        cash -= fee
                        positions[pair] = {"side": signal, "entry": entry_p, "size": size}
                        label = "LONG" if signal == 1 else "SHORT"
                        trades.append({"pair": pair, "action": f"OPEN {label}", "price": entry_p, "size": size})

    for pair, pos in list(positions.items()):
        price = float(eth_data["close"].iloc[-1]) if "ETH" in pair else float(btc_data["close"].iloc[-1])
        exit_p = price * (1 - SLIPPAGE_RATE * pos["side"])
        pnl_pct = (exit_p - pos["entry"]) / pos["entry"] * pos["side"]
        pnl_usd = pos["size"] * pnl_pct - pos["size"] * FEE_RATE
        cash += pos["size"] + pnl_usd
        trades.append({"pair": pair, "action": "CLOSE (end)", "pnl_pct": pnl_pct * 100, "pnl_usd": pnl_usd})

    final = cash
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    closes = [t for t in trades if t["action"].startswith("CLOSE")]
    wins = sum(1 for t in closes if t["pnl_usd"] > 0)

    print("--- PORTFOLIO (weighted) ---")
    print(f"  Trades: {len(trades)} ({len(closes)} closed)")
    if closes:
        print(f"  Wins: {wins}/{len(closes)} ({wins / len(closes) * 100:.0f}%)")
    print(f"  Final equity: ${final:.2f}")
    print(f"  Return: {ret:+.2f}% (${final - INITIAL_CAPITAL:+.2f})")
    for t in trades:
        if "pnl_pct" in t:
            print(f"    {t['pair']} {t['action']}: {t['pnl_pct']:+.2f}% (${t['pnl_usd']:+.2f})")
        else:
            print(f"    {t['pair']} {t['action']}: ${t['price']:,.2f} size=${t['size']:.2f}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("BACKTEST: AUG 15 - AUG 26 RALLY PERIOD")
    print("=" * 70)
    print()

    eth_data = fetch("ETH/USDT")
    btc_data = fetch("BTC/USDT")

    print(f"ETH data: {len(eth_data)} candles ({eth_data['timestamp'].iloc[0]} to {eth_data['timestamp'].iloc[-1]})")
    print(f"BTC data: {len(btc_data)} candles ({btc_data['timestamp'].iloc[0]} to {btc_data['timestamp'].iloc[-1]})")
    print()

    for name, (symbol, strat_name, params, weight) in STRATEGIES.items():
        df = eth_data if "ETH" in symbol else btc_data
        backtest_strategy(name, df, strat_name, params, weight)
        time.sleep(0.3)

    print("=" * 70)
    print("PORTFOLIO SIMULATION (weighted combination)")
    print("=" * 70)
    backtest_portfolio(eth_data, btc_data)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The rally from Aug 15-22 saw ETH go from ~$1,880 to ~$2,440 (+30%)")
    print("and BTC from ~$63,000 to ~$77,000 (+22%).")
    print("The strategies captured some of this move but were whipsawed")
    print("by the rapid price changes and short-lived signals.")
