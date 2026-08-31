"""
Bot Scheduler Daemon

Continuously runs the paper trading bot and position tracker on a schedule.
Designed for Railway (or any always-on server) — replaces GitHub Actions cron.

Features:
  - Runs trading bot every 15 minutes
  - Runs position tracker every 15 minutes (offset by 2 min)
  - Health check endpoint via HTTP on port 8080
  - Graceful shutdown on SIGTERM/SIGINT
  - Error recovery (crashes don't kill the scheduler)
  - Memory-efficient (no state accumulation)
"""

import os
import sys
import time
import signal
import threading
import traceback
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration ---
BOT_INTERVAL = 15 * 60       # 15 minutes in seconds
TRACKER_INTERVAL = 15 * 60   # 15 minutes
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8080"))

# --- Graceful Shutdown ---
_shutdown = False

def _signal_handler(sig, frame):
    global _shutdown
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Received signal {sig}, shutting down...")
    _shutdown = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# --- Health Check HTTP Server ---
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Trading Bot is running")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


def start_health_server():
    """Start health check HTTP server in a background thread."""
    try:
        server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"Health server listening on port {HEALTH_PORT}")
        return server
    except Exception as e:
        print(f"WARNING: Health server failed to start: {e}")
        return None


# --- Scheduler ---
def run_bot():
    """Run the paper trading bot."""
    now = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*50}")
    print(f"[{now}] Running trading bot...")
    print(f"{'='*50}")
    try:
        from scripts.paper_trader import main
        main()
    except Exception as e:
        print(f"[{now}] Bot error: {e}")
        traceback.print_exc()
        # Try to send error notification
        try:
            from scripts.paper_trader import load_telegram_config, tg_send_message
            cfg = load_telegram_config()
            if cfg.get("bot_token"):
                tg_send_message(
                    cfg["bot_token"], str(cfg["chat_id"]),
                    f"Bot error at {now}:\n{str(e)[:200]}"
                )
        except Exception:
            pass


def run_tracker():
    """Run the position tracker."""
    now = datetime.now(timezone.utc).isoformat()
    print(f"\n[{now}] Running position tracker...")
    try:
        from scripts.position_tracker import check_positions
        check_positions()
    except Exception as e:
        print(f"[{now}] Tracker error: {e}")
        traceback.print_exc()


def main():
    """Main scheduler loop."""
    print("=" * 50)
    print("TRADING BOT SCHEDULER")
    print(f"Bot interval: {BOT_INTERVAL // 60} min")
    print(f"Tracker interval: {TRACKER_INTERVAL // 60} min")
    print(f"Health port: {HEALTH_PORT}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 50)

    # Start health server
    start_health_server()

    # Run bot immediately on start
    run_bot()

    # Schedule loops
    last_bot_run = time.time()
    last_tracker_run = time.time()

    # Initial tracker run (2 min after bot start)
    time.sleep(120)
    run_tracker()
    last_tracker_run = time.time()

    print("\nScheduler running. Waiting for next cycle...")

    while not _shutdown:
        try:
            now = time.time()

            # Run bot every 15 minutes
            if now - last_bot_run >= BOT_INTERVAL:
                run_bot()
                last_bot_run = time.time()

            # Run tracker every 15 minutes (offset from bot)
            if now - last_tracker_run >= TRACKER_INTERVAL:
                run_tracker()
                last_tracker_run = time.time()

            # Sleep 30 seconds between checks
            time.sleep(30)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Scheduler error: {e}")
            traceback.print_exc()
            time.sleep(60)  # Wait a minute before retrying

    print("Scheduler stopped.")


if __name__ == "__main__":
    main()
