"""
Telegram Command Bot

Long-running bot that polls for Telegram commands and responds with
paper trading status, positions, and equity information.

Commands:
  /status   - Full portfolio status (equity, P&L, positions, win rate)
  /equity   - Current equity and return
  /positions - Open positions with P&L
  /trades   - Recent trade history
  /signals  - Current strategy signals
  /help     - List all commands

Usage:
  python scripts/telegram_bot.py

Note: This script must be running continuously to respond to commands.
Run it locally, on a VPS, or via GitHub Actions cron (see weekly_runner.yml).
"""

import os
import sys
import json
import time
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

# --- Config ---
STATE_FILE = Path("data/results/paper_state.json")
TRADE_LOG = Path("data/results/paper_trades.jsonl")
SUMMARY_FILE = Path("data/results/paper_summary.json")
POLL_INTERVAL = 3  # seconds between polls
INITIAL_CAPITAL = 10000.0

running = True


def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- Telegram API ---
def load_config():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        return {"bot_token": token, "chat_id": chat_id}
    try:
        import yaml
        cfg_path = Path("configs/bot_live.yaml")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            tg = cfg.get("bot", {}).get("telegram", {})
            if tg.get("bot_token"):
                return {"bot_token": tg["bot_token"], "chat_id": str(tg.get("chat_id", ""))}
    except Exception:
        pass
    return {}


def api_call(token: str, method: str, params: dict = None) -> dict:
    """Make a Telegram Bot API call via curl."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    result = subprocess.run(
        ["curl", "-s", "-m", "15", "-X", "POST", url,
         "-H", "Content-Type: application/json",
         "-d", json.dumps(params or {})],
        capture_output=True, text=True, timeout=20,
    )
    if result.stdout:
        return json.loads(result.stdout)
    return {"ok": False, "description": "No response"}


def send_message(token: str, chat_id: str, text: str) -> bool:
    resp = api_call(token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    })
    return resp.get("ok", False)


def send_photo(token: str, chat_id: str, photo_path: str, caption: str = "") -> bool:
    """Send a photo file via Telegram."""
    result = subprocess.run(
        ["curl", "-s", "-m", "30", "-X", "POST",
         f"https://api.telegram.org/bot{token}/sendPhoto",
         "-F", f"chat_id={chat_id}",
         "-F", f"photo=@{photo_path}",
         "-F", f"caption={caption}",
         "-F", "parse_mode=HTML"],
        capture_output=True, text=True, timeout=35,
    )
    if result.stdout:
        data = json.loads(result.stdout)
        return data.get("ok", False)
    return False


# --- State Reading ---
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": INITIAL_CAPITAL, "positions": {}, "total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "peak_equity": INITIAL_CAPITAL, "max_drawdown": 0.0}


def load_trades(limit: int = 10) -> list:
    trades = []
    if TRADE_LOG.exists():
        with open(TRADE_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return trades[-limit:]


def get_signals() -> dict:
    """Run strategies and return current signals."""
    try:
        from trading_system.strategies import STRATEGY_REGISTRY

        STRATEGIES = {
            "MACD ETH": {"strategy": "MACD", "pair": "ETH_USDT_USDT", "timeframe": "4h"},
            "ROC_Momentum ETH": {"strategy": "ROC_Momentum", "pair": "ETH_USDT_USDT", "timeframe": "4h"},
            "MACD BTC": {"strategy": "MACD", "pair": "BTC_USDT_USDT", "timeframe": "4h"},
        }

        signals = {}
        for name, cfg in STRATEGIES.items():
            try:
                for pattern in [f"data/raw/{cfg['pair']}/{cfg['timeframe']}.parquet",
                               f"data/raw/{cfg['pair']}/klines_{cfg['timeframe']}.parquet"]:
                    p = Path(pattern)
                    if p.exists():
                        import pandas as pd
                        df = pd.read_parquet(p)
                        strat = STRATEGY_REGISTRY[cfg["strategy"]]
                        # Load params from the summary or use defaults
                        params = {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True}
                        if "ROC" in cfg["strategy"]:
                            params = {"roc_period": 10, "signal_period": 5, "use_ema": True, "ema_period": 12}
                        sig = strat.generate_signals(df, params)
                        latest = int(sig.iloc[-1])
                        price = float(df["close"].iloc[-1])
                        direction = "LONG" if latest == 1 else "FLAT" if latest == 0 else "SHORT"
                        signals[name] = {"signal": latest, "direction": direction, "price": price}
                        break
            except Exception as e:
                signals[name] = {"signal": 0, "direction": "ERROR", "price": 0, "error": str(e)}
        return signals
    except Exception as e:
        return {"error": str(e)}


# --- Command Handlers ---
def handle_status(token: str, chat_id: str):
    """Full portfolio status."""
    state = load_state()
    equity = state["cash"]
    # Add unrealized for open positions (simplified)
    for pair, pos in state.get("positions", {}).items():
        equity += pos.get("size_usd", 0)

    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wr = state["wins"] / state["total_trades"] * 100 if state["total_trades"] > 0 else 0

    positions_text = ""
    for pair, pos in state.get("positions", {}).items():
        side = "LONG" if pos.get("side") == 1 else "SHORT"
        positions_text += f"  {'🟢' if pos.get('side') == 1 else '🔴'} {pair.replace('_', '/')} {side} @ ${pos.get('entry_price', 0):,.2f}\n"
    if not positions_text:
        positions_text = "  No open positions\n"

    msg = (
        f"📋 <b>PORTFOLIO STATUS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Equity: <b>${equity:,.2f}</b>\n"
        f"📈 Return: <b>{total_return:+.2f}%</b>\n"
        f"💵 Cash: ${state['cash']:,.2f}\n"
        f"📊 Trades: {state['total_trades']} (W:{state['wins']} / L:{state['losses']})\n"
        f"🏆 Win Rate: {wr:.1f}%\n"
        f"📉 Realized P&L: ${state['total_pnl']:+,.2f}\n"
        f"🔻 Max Drawdown: {state.get('max_drawdown', 0) * 100:.2f}%\n"
        f"\n📍 <b>Open Positions:</b>\n{positions_text}"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    send_message(token, chat_id, msg)


def handle_equity(token: str, chat_id: str):
    """Quick equity check."""
    state = load_state()
    equity = state["cash"]
    for pair, pos in state.get("positions", {}).items():
        equity += pos.get("size_usd", 0)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    msg = (
        f"💰 <b>EQUITY</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Current: <b>${equity:,.2f}</b>\n"
        f"Return: <b>{total_return:+.2f}%</b>\n"
        f"Cash: ${state['cash']:,.2f}"
    )
    send_message(token, chat_id, msg)


def handle_positions(token: str, chat_id: str):
    """Show open positions."""
    state = load_state()
    positions = state.get("positions", {})

    if not positions:
        send_message(token, chat_id, "📍 No open positions")
        return

    msg = "📍 <b>OPEN POSITIONS</b>\n━━━━━━━━━━━━━━━━━━━━━\n"
    for pair, pos in positions.items():
        side = "LONG" if pos.get("side") == 1 else "SHORT"
        emoji = "🟢" if pos.get("side") == 1 else "🔴"
        msg += (
            f"\n{emoji} <b>{pair.replace('_', '/')}</b> {side}\n"
            f"  Entry: ${pos.get('entry_price', 0):,.2f}\n"
            f"  Size: ${pos.get('size_usd', 0):,.2f}\n"
            f"  Strategy: {pos.get('strategy', 'Unknown')}\n"
            f"  Opened: {pos.get('entry_time', 'Unknown')[:16]}\n"
        )
    send_message(token, chat_id, msg)


def handle_trades(token: str, chat_id: str):
    """Show recent trades."""
    trades = load_trades(10)
    if not trades:
        send_message(token, chat_id, "📋 No trades yet")
        return

    msg = "📋 <b>RECENT TRADES</b>\n━━━━━━━━━━━━━━━━━━━━━\n"
    for t in reversed(trades):
        action = t.get("action", "?")
        pair = t.get("pair", "?").replace("_", "/")
        if action == "CLOSE":
            pnl = t.get("pnl_usd", 0)
            emoji = "✅" if pnl >= 0 else "❌"
            msg += f"\n{emoji} {pair} CLOSED @ ${t.get('exit_price', 0):,.2f}\n"
            msg += f"  P&L: ${pnl:+.2f} ({t.get('pnl_pct', 0):+.2f}%)\n"
        elif action.startswith("OPEN"):
            msg += f"\n🟢 {pair} {action} @ ${t.get('price', 0):,.2f}\n"
        else:
            msg += f"\n📝 {pair} {action}\n"

    send_message(token, chat_id, msg)


def handle_signals(token: str, chat_id: str):
    """Show current strategy signals."""
    signals = get_signals()
    msg = "🧠 <b>CURRENT SIGNALS</b>\n━━━━━━━━━━━━━━━━━━━━━\n"

    if "error" in signals:
        msg += f"❌ Error: {signals['error']}"
    else:
        for name, data in signals.items():
            emoji = "🟢" if data.get("signal") == 1 else "🔴" if data.get("signal") == -1 else "⚪"
            price = data.get("price", 0)
            direction = data.get("direction", "?")
            msg += f"\n{emoji} <b>{name}</b>\n"
            msg += f"  Signal: {direction} (${price:,.2f})\n"

    send_message(token, chat_id, msg)


def handle_restart(token: str, chat_id: str):
    """Trigger a bot run via GitHub Actions API."""
    import subprocess

    send_message(token, chat_id, "Triggering bot run... please wait 30s.")

    # Try GitHub Actions API
    github_token = os.getenv("GITHUB_TOKEN", "")
    if not github_token:
        # Try to read from config
        try:
            import yaml
            cfg_path = Path("configs/bot_live.yaml")
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                github_token = cfg.get("github", {}).get("token", "")
        except Exception:
            pass

    if github_token:
        result = subprocess.run(
            ["curl", "-s", "-m", "15", "-X", "POST",
             "https://api.github.com/repos/SonSquared/Trading-Bot/actions/workflows/bot.yml/dispatches",
             "-H", f"Authorization: token {github_token}",
             "-H", "Accept: application/vnd.github.v3+json",
             "-d", '{"ref":"main"}'],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            send_message(token, chat_id, "Bot run triggered! Check /status in 2 minutes.")
            return

    # Fallback: run locally
    send_message(token, chat_id, "GitHub API not configured. Running locally...")
    try:
        result = subprocess.run(
            [sys.executable, "-u", "scripts/paper_trader.py"],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent.parent),
        )
        if result.returncode == 0:
            send_message(token, chat_id, "Bot run completed! Check /status for results.")
        else:
            send_message(token, chat_id, f"Bot run failed: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        send_message(token, chat_id, "Bot run timed out (>2 min).")
    except Exception as e:
        send_message(token, chat_id, f"Error: {e}")


def handle_dashboard(token: str, chat_id: str):
    """Show monitoring dashboard summary."""
    RUN_LOG = Path("data/results/run_history.jsonl")
    STATE_FILE = Path("data/results/paper_state.json")
    SUMMARY_FILE = Path("data/results/paper_summary.json")

    runs = []
    if RUN_LOG.exists():
        with open(RUN_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    state = {}
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)

    summary = {}
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE) as f:
            summary = json.load(f)

    if not runs:
        send_message(token, chat_id, "No run history yet.")
        return

    total = len(runs)
    successful = sum(1 for r in runs if r.get("status") == "success")
    failed = sum(1 for r in runs if r.get("status") == "failed")
    success_rate = successful / total * 100 if total > 0 else 0

    last = runs[-1]
    status_color = {"success": "GREEN", "partial": "YELLOW", "failed": "RED"}.get(last.get("status", ""), "UNKNOWN")

    # Consecutive failures
    consec = 0
    for r in reversed(runs):
        if r.get("status") == "failed":
            consec += 1
        else:
            break

    equity = summary.get("equity", state.get("cash", 10000))
    total_return = summary.get("total_return_pct", 0)

    msg = (
        f"📊 <b>BOT DASHBOARD</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Status: [{status_color}] <b>{last.get('status', '?').upper()}</b>\n"
        f"Runs: {total} ({success_rate:.0f}% success)\n"
        f"Failed: {failed}\n"
        f"Consec. fails: {consec}\n\n"
        f"💰 Equity: <b>${equity:,.2f}</b>\n"
        f"📈 Return: <b>{total_return:+.2f}%</b>\n"
        f"⏱ Last run: {last.get('duration_seconds', 0):.1f}s\n"
        f"🕐 {last.get('timestamp', '?')[:19]}"
    )
    send_message(token, chat_id, msg)


def handle_help(token: str, chat_id: str):
    """Show available commands."""
    msg = (
        "🤖 <b>TRADING BOT COMMANDS</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "/status - Full portfolio status\n"
        "/equity - Quick equity check\n"
        "/positions - Open positions\n"
        "/trades - Recent trade history\n"
        "/signals - Current strategy signals\n"
        "/restart - Trigger a bot run now\n"
        "/dashboard - Bot monitoring dashboard\n"
        "/help - This message\n\n"
        f"Mode: <b>PAPER</b>\n"
        f"Initial: ${INITIAL_CAPITAL:,.0f}"
    )
    send_message(token, chat_id, msg)


COMMANDS = {
    "/status": handle_status,
    "/equity": handle_equity,
    "/positions": handle_positions,
    "/trades": handle_trades,
    "/signals": handle_signals,
    "/restart": handle_restart,
    "/dashboard": handle_dashboard,
    "/help": handle_help,
}


# --- Main Loop ---
def main():
    config = load_config()
    if not config.get("bot_token"):
        print("ERROR: No Telegram bot token configured.")
        print("Set TELEGRAM_BOT_TOKEN env var or configure in configs/bot_live.yaml")
        sys.exit(1)

    token = config["bot_token"]
    chat_id = config["chat_id"]

    print("=" * 50)
    print("TELEGRAM COMMAND BOT")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Commands: {', '.join(COMMANDS.keys())}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print("=" * 50)
    print("Listening for commands... (Ctrl+C to stop)")

    # Get last update_id to skip old messages
    resp = api_call(token, "getUpdates", {"limit": 1, "timeout": 0})
    offset = 0
    if resp.get("ok") and resp.get("result"):
        offset = resp["result"][-1].get("update_id", 0) + 1

    while running:
        try:
            resp = api_call(token, "getUpdates", {
                "offset": offset,
                "timeout": POLL_INTERVAL,
                "allowed_updates": json.dumps(["message"]),
            })

            if resp.get("ok") and resp.get("result"):
                for update in resp["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip().lower()
                    msg_chat_id = str(msg.get("chat", {}).get("id", ""))

                    # Only respond to the configured chat
                    if msg_chat_id != chat_id:
                        continue

                    if text in COMMANDS:
                        print(f"  Command: {text}")
                        COMMANDS[text](token, chat_id)
                    elif text.startswith("/"):
                        send_message(token, chat_id, f"Unknown command: {text}\nType /help for available commands.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5)

    print("Bot stopped.")


if __name__ == "__main__":
    main()
