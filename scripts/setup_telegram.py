"""
Telegram Bot Setup & Test Script

Interactive guide to set up Telegram notifications.

Usage:
  python scripts/setup_telegram.py          # Interactive setup
  python scripts/setup_telegram.py --test   # Test existing config
  python scripts/setup_telegram.py --send-test  # Send a test message
"""

import sys
import os
sys.path.insert(0, ".")

import yaml
import requests
from pathlib import Path


CONFIG_FILE = Path("configs/bot_live.yaml")


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_me(token: str) -> dict | None:
    """Get bot info from Telegram API."""
    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("result")
    except Exception:
        pass
    return None


def send_message(token: str, chat_id: str, text: str) -> bool:
    """Send a test message."""
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def setup():
    """Interactive setup."""
    print("=" * 60)
    print("  TELEGRAM BOT SETUP")
    print("=" * 60)
    print()
    print("Step 1: Create a Telegram Bot")
    print("-" * 40)
    print("  1. Open Telegram and search for @BotFather")
    print("  2. Send /newbot")
    print("  3. Choose a name (e.g., 'My Trading Bot')")
    print("  4. Choose a username (e.g., 'my_trading_bot_123')")
    print("  5. Copy the bot token (looks like: 123456:ABC-DEF...)")
    print()

    token = input("Paste your bot token here: ").strip()
    if not token:
        print("No token provided. Exiting.")
        return

    # Validate token
    bot_info = get_me(token)
    if not bot_info:
        print("❌ Invalid token! Make sure you copied it correctly.")
        return

    print(f"✅ Bot connected: @{bot_info['username']} ({bot_info['first_name']})")
    print()

    print("Step 2: Get Your Chat ID")
    print("-" * 40)
    print("  1. Open Telegram and search for @userinfobot")
    print("  2. Send /start")
    print("  3. Copy your chat ID (a number like 123456789)")
    print()
    print("  Alternative: Search for @getmyid_bot and send /getid")
    print()

    chat_id = input("Paste your chat ID here: ").strip()
    if not chat_id:
        print("No chat ID provided. Exiting.")
        return

    # Test connection
    print()
    print("Testing connection...")
    test_msg = (
        "✅ <b>Trading Bot Connected!</b>\n\n"
        "Bot: @" + bot_info['username'] + "\n"
        "You will receive:\n"
        "  • Trade alerts (open/close)\n"
        "  • Stop-loss triggers\n"
        "  • Daily summaries\n"
        "  • Error notifications"
    )

    if send_message(token, chat_id, test_msg):
        print("✅ Test message sent successfully!")
        print("  Check your Telegram for the message.")
    else:
        print("❌ Failed to send test message.")
        print("  Make sure you:")
        print("  1. Started a chat with your bot (send /start)")
        print("  2. Copied the correct chat ID")
        return

    # Save to config
    print()
    config = load_config()
    if "bot" not in config:
        config["bot"] = {}
    if "telegram" not in config["bot"]:
        config["bot"]["telegram"] = {}

    config["bot"]["telegram"]["enabled"] = True
    config["bot"]["telegram"]["bot_token"] = token
    config["bot"]["telegram"]["chat_id"] = chat_id

    save_config(config)
    print(f"✅ Config saved to {CONFIG_FILE}")
    print()
    print("Telegram notifications are now enabled!")
    print("The bot will send alerts for:")
    print("  • New position opened")
    print("  • Position closed (with P&L)")
    print("  • Stop-loss triggered")
    print("  • Daily summary")
    print("  • Errors and emergency stops")


def test_existing():
    """Test existing configuration."""
    config = load_config()
    telegram = config.get("bot", {}).get("telegram", {})

    if not telegram.get("enabled"):
        print("❌ Telegram is not enabled in config.")
        print("  Run: python scripts/setup_telegram.py")
        return

    token = telegram.get("bot_token", "")
    chat_id = telegram.get("chat_id", "")

    if not token or not chat_id:
        print("❌ Missing bot_token or chat_id in config.")
        return

    # Test
    print("Testing Telegram connection...")
    bot_info = get_me(token)
    if bot_info:
        print(f"✅ Bot: @{bot_info['username']}")
    else:
        print("❌ Invalid bot token")
        return

    test_msg = (
        "🧪 <b>Test Message</b>\n\n"
        "Telegram notifications are working correctly.\n"
        "You will receive trade alerts here."
    )

    if send_message(token, chat_id, test_msg):
        print("✅ Test message sent! Check your Telegram.")
    else:
        print("❌ Failed to send message. Check chat_id.")


def send_test_message():
    """Send a sample trade notification."""
    config = load_config()
    telegram = config.get("bot", {}).get("telegram", {})

    if not telegram.get("enabled"):
        print("❌ Telegram is not enabled.")
        return

    token = telegram["bot_token"]
    chat_id = telegram["chat_id"]

    # Send sample notifications
    from trading_system.bot.telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier(token, chat_id, enabled=True)

    print("Sending sample notifications...")
    print()

    notifier.notify_trade_open(
        pair="ETH/USDT:USDT", side="buy", price=2500.00,
        amount=0.5, confidence=0.85, strategy="portfolio", mode="paper"
    )
    print("  ✅ Trade open notification sent")

    import time
    time.sleep(1)

    notifier.notify_trade_close(
        pair="ETH/USDT:USDT", side="buy", entry_price=2500.00,
        exit_price=2575.00, pnl_pct=3.0, pnl_usd=37.50,
        reason="signal_exit", mode="paper"
    )
    print("  ✅ Trade close notification sent")

    time.sleep(1)

    notifier.notify_daily_summary(
        equity=10150.00, daily_pnl=150.00, daily_pnl_pct=1.5,
        trades_today=3, open_positions=[
            {"pair": "ETH/USDT:USDT", "side": "long", "unrealized_pnl_pct": 2.1},
        ],
        win_rate=68.5, total_trades=2911
    )
    print("  ✅ Daily summary sent")

    print()
    print("Check your Telegram for the sample notifications!")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_existing()
    elif "--send-test" in sys.argv:
        send_test_message()
    else:
        setup()
