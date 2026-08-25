"""
Bot Monitoring Dashboard

Reads run history and displays:
  - Uptime and success rate
  - Error history
  - Performance metrics
  - Can be viewed as HTML or sent via Telegram

Usage:
  python scripts/bot_dashboard.py          # Print to console
  python scripts/bot_dashboard.py --html   # Generate HTML dashboard
  python scripts/bot_dashboard.py --telegram  # Send summary via Telegram
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")

RUN_LOG = Path("data/results/run_history.jsonl")
STATE_FILE = Path("data/results/paper_state.json")
SUMMARY_FILE = Path("data/results/paper_summary.json")
TRADE_LOG = Path("data/results/paper_trades.jsonl")
HTML_OUTPUT = Path("data/results/monitoring_dashboard.html")


def load_runs() -> list:
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
    return runs


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def load_summary() -> dict:
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE) as f:
            return json.load(f)
    return {}


def load_trades() -> list:
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
    return trades


def compute_stats(runs: list) -> dict:
    if not runs:
        return {
            "total_runs": 0, "successful": 0, "failed": 0, "partial": 0,
            "success_rate": 0, "uptime_hours": 0, "avg_duration": 0,
            "last_run": None, "last_status": "unknown", "errors_today": 0,
            "runs_today": 0, "consecutive_failures": 0,
        }

    total = len(runs)
    successful = sum(1 for r in runs if r.get("status") == "success")
    failed = sum(1 for r in runs if r.get("status") == "failed")
    partial = sum(1 for r in runs if r.get("status") == "partial")

    # Time stats
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    runs_today = [r for r in runs if datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) >= today_start]
    errors_today = sum(1 for r in runs_today if r.get("errors"))

    # Uptime
    if len(runs) >= 2:
        first_run = datetime.fromisoformat(runs[0]["timestamp"].replace("Z", "+00:00"))
        uptime = (now - first_run).total_seconds() / 3600
    else:
        uptime = 0

    # Consecutive failures
    consecutive_failures = 0
    for r in reversed(runs):
        if r.get("status") == "failed":
            consecutive_failures += 1
        else:
            break

    # Average duration
    durations = [r.get("duration_seconds", 0) for r in runs]
    avg_duration = sum(durations) / len(durations) if durations else 0

    last_run = runs[-1] if runs else None

    return {
        "total_runs": total,
        "successful": successful,
        "failed": failed,
        "partial": partial,
        "success_rate": successful / total * 100 if total > 0 else 0,
        "uptime_hours": uptime,
        "avg_duration": avg_duration,
        "last_run": last_run,
        "last_status": last_run.get("status", "unknown") if last_run else "unknown",
        "errors_today": errors_today,
        "runs_today": len(runs_today),
        "consecutive_failures": consecutive_failures,
    }


def print_dashboard():
    runs = load_runs()
    stats = compute_stats(runs)
    state = load_state()
    summary = load_summary()

    print("=" * 60)
    print("BOT MONITORING DASHBOARD")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Status indicator
    status_emoji = {"success": "GREEN", "partial": "YELLOW", "failed": "RED"}.get(stats["last_status"], "UNKNOWN")
    print(f"\nStatus: [{status_emoji}] {stats['last_status'].upper()}")

    # Core metrics
    print(f"\n--- RUN STATS ---")
    print(f"Total runs:      {stats['total_runs']}")
    print(f"Successful:      {stats['successful']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Partial:         {stats['partial']}")
    print(f"Success rate:    {stats['success_rate']:.1f}%")
    print(f"Uptime:          {stats['uptime_hours']:.1f} hours")
    print(f"Avg duration:    {stats['avg_duration']:.1f}s")

    # Today
    print(f"\n--- TODAY ---")
    print(f"Runs today:      {stats['runs_today']}")
    print(f"Errors today:    {stats['errors_today']}")
    print(f"Consec. fails:   {stats['consecutive_failures']}")

    # Performance
    equity = summary.get("equity", state.get("cash", 10000))
    total_return = summary.get("total_return_pct", 0)
    total_trades = summary.get("total_trades", state.get("total_trades", 0))
    win_rate = summary.get("win_rate", 0)
    realized_pnl = summary.get("realized_pnl", state.get("total_pnl", 0))

    print(f"\n--- PERFORMANCE ---")
    print(f"Equity:          ${equity:,.2f}")
    print(f"Return:          {total_return:+.2f}%")
    print(f"Total trades:    {total_trades}")
    print(f"Win rate:        {win_rate:.1f}%")
    print(f"Realized P&L:    ${realized_pnl:+,.2f}")
    print(f"Open positions:  {len(state.get('positions', {}))}")

    # Recent errors
    recent_errors = []
    for r in reversed(runs[-20:]):
        if r.get("errors"):
            for e in r["errors"][:2]:
                recent_errors.append(f"  [{r['timestamp'][:16]}] {e}")
    if recent_errors:
        print(f"\n--- RECENT ERRORS ---")
        for e in recent_errors[-5:]:
            print(e)
    else:
        print(f"\n--- RECENT ERRORS ---")
        print("  None")

    # Last run details
    if stats["last_run"]:
        lr = stats["last_run"]
        print(f"\n--- LAST RUN ---")
        print(f"Time:     {lr['timestamp'][:19]}")
        print(f"Status:   {lr.get('status', '?')}")
        print(f"Duration: {lr.get('duration_seconds', 0):.1f}s")
        print(f"Trades:   {lr.get('trades', 0)}")
        if lr.get("errors"):
            print(f"Errors:   {len(lr['errors'])}")

    print("\n" + "=" * 60)


def generate_html():
    runs = load_runs()
    stats = compute_stats(runs)
    state = load_state()
    summary = load_summary()

    equity = summary.get("equity", state.get("cash", 10000))
    total_return = summary.get("total_return_pct", 0)
    total_trades = summary.get("total_trades", 0)
    win_rate = summary.get("win_rate", 0)

    status_color = {"success": "#00ff88", "partial": "#ffaa00", "failed": "#ff4444"}.get(stats["last_status"], "#888888")

    # Recent runs for chart
    recent_runs = runs[-50:] if runs else []
    run_rows = ""
    for r in reversed(recent_runs):
        sc = {"success": "#00ff88", "partial": "#ffaa00", "failed": "#ff4444"}.get(r.get("status", ""), "#888888")
        err_text = "; ".join(r.get("errors", [])[:2]) if r.get("errors") else "-"
        run_rows += f"""
        <tr>
            <td>{r.get('timestamp', '?')[:19]}</td>
            <td style="color:{sc}">{r.get('status', '?').upper()}</td>
            <td>{r.get('duration_seconds', 0):.1f}s</td>
            <td>{r.get('trades', 0)}</td>
            <td class="error-text">{err_text}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Bot Dashboard</title>
<style>
    body {{ background: #0d1117; color: #e6e6e6; font-family: -apple-system, sans-serif; margin: 20px; }}
    h1 {{ color: #00ff88; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
    .card .value {{ font-size: 28px; font-weight: bold; color: #00ff88; }}
    .card .label {{ color: #8b949e; font-size: 13px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }}
    th {{ color: #8b949e; font-weight: 600; }}
    .error-text {{ color: #ff6666; font-size: 12px; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 14px; }}
</style></head><body>
<h1>Bot Monitoring Dashboard</h1>
<p>Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="grid">
    <div class="card"><div class="value">{stats['success_rate']:.0f}%</div><div class="label">Success Rate</div></div>
    <div class="card"><div class="value">{stats['total_runs']}</div><div class="label">Total Runs</div></div>
    <div class="card"><div class="value">{stats['uptime_hours']:.0f}h</div><div class="label">Uptime</div></div>
    <div class="card"><div class="value" style="color:{status_color}">{stats['last_status'].upper()}</div><div class="label">Current Status</div></div>
</div>

<div class="grid">
    <div class="card"><div class="value">${equity:,.0f}</div><div class="label">Equity</div></div>
    <div class="card"><div class="value">{total_return:+.1f}%</div><div class="label">Return</div></div>
    <div class="card"><div class="value">{total_trades}</div><div class="label">Total Trades</div></div>
    <div class="card"><div class="value">{win_rate:.0f}%</div><div class="label">Win Rate</div></div>
</div>

<h2>Recent Runs</h2>
<table>
<tr><th>Time</th><th>Status</th><th>Duration</th><th>Trades</th><th>Errors</th></tr>
{run_rows}
</table>

</body></html>"""

    with open(HTML_OUTPUT, "w") as f:
        f.write(html)
    print(f"Dashboard saved to {HTML_OUTPUT}")


def send_telegram_summary():
    runs = load_runs()
    stats = compute_stats(runs)
    state = load_state()
    summary = load_summary()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        try:
            import yaml
            cfg_path = Path("configs/bot_live.yaml")
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                tg = cfg.get("bot", {}).get("telegram", {})
                token = tg.get("bot_token", "")
                chat_id = str(tg.get("chat_id", ""))
        except Exception:
            pass

    if not token or not chat_id:
        print("Telegram not configured")
        return

    equity = summary.get("equity", state.get("cash", 10000))
    total_return = summary.get("total_return_pct", 0)
    wr = summary.get("win_rate", 0)

    status_emoji = {"success": "GREEN", "partial": "YELLOW", "failed": "RED"}.get(stats["last_status"], "UNKNOWN")

    msg = (
        f"📊 <b>BOT MONITORING DASHBOARD</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Status: [{status_emoji}] <b>{stats['last_status'].upper()}</b>\n"
        f"Uptime: {stats['uptime_hours']:.1f}h\n"
        f"Runs: {stats['total_runs']} ({stats['success_rate']:.0f}% success)\n"
        f"Today: {stats['runs_today']} runs, {stats['errors_today']} errors\n"
        f"Consec. fails: {stats['consecutive_failures']}\n\n"
        f"💰 Equity: <b>${equity:,.2f}</b>\n"
        f"📈 Return: <b>{total_return:+.2f}%</b>\n"
        f"🏆 Win Rate: {wr:.1f}%\n"
        f"📊 Trades: {summary.get('total_trades', 0)}\n\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    result = subprocess.run(
        ["curl", "-s", "-m", "15", "-X", "POST",
         f"https://api.telegram.org/bot{token}/sendMessage",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"})],
        capture_output=True, text=True, timeout=20,
    )
    if result.stdout:
        data = json.loads(result.stdout)
        if data.get("ok"):
            print("Dashboard sent to Telegram")
        else:
            print(f"Failed: {data.get('description', '')}")
    else:
        print("Failed to send")


if __name__ == "__main__":
    if "--html" in sys.argv:
        generate_html()
    elif "--telegram" in sys.argv:
        send_telegram_summary()
    else:
        print_dashboard()
