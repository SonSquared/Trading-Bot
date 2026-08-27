"""
Trading Dashboard

Generates a comprehensive HTML dashboard showing:
  - Current positions with live P&L
  - Strategy signals and recent trades
  - Equity curve chart (using Chart.js)
  - Trade history table
  - Risk metrics and drawdown
  - Bot run history

Run locally: python scripts/trading_dashboard.py
Auto-refreshes every 60 seconds when opened in browser.
"""

import json
import os
import sys
import io
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

RESULTS_DIR = Path("data/results")
DASHBOARD_FILE = RESULTS_DIR / "dashboard.html"


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def generate_dashboard():
    """Generate the HTML dashboard."""
    state = load_json(RESULTS_DIR / "paper_state.json")
    summary = load_json(RESULTS_DIR / "paper_summary.json")
    tracker = load_json(RESULTS_DIR / "position_tracker.json")
    run_history = []
    if (RESULTS_DIR / "run_history.jsonl").exists():
        with open(RESULTS_DIR / "run_history.jsonl") as f:
            for line in f:
                try:
                    run_history.append(json.loads(line.strip()))
                except:
                    pass

    # Trade history
    trades = []
    if (RESULTS_DIR / "paper_trades.jsonl").exists():
        with open(RESULTS_DIR / "paper_trades.jsonl") as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except:
                    pass

    cash = state.get("cash", 97.0)
    positions = state.get("positions", {})
    
    # Calculate unrealized P&L from position tracker
    position_checks = tracker.get("checks", [])
    latest_prices = {}
    for c in position_checks:
        sym = c.get("symbol", "")
        latest_prices[sym] = c.get("current_price", 0)
    
    unrealized_pnl = 0
    unrealized_pnl_pct = 0
    for sym, pos in positions.items():
        entry = pos.get("entry_price", 0)
        size = pos.get("size_usd", pos.get("size", 0))
        side = pos.get("side", 0)
        current = latest_prices.get(sym, entry)
        if entry > 0:
            if side == 1:
                upnl = size * (current - entry) / entry
            else:
                upnl = size * (entry - current) / entry
            unrealized_pnl += upnl
    
    # Real equity = cash + position values + unrealized P&L
    position_value = sum(p.get("size_usd", p.get("size", 0)) for p in positions.values())
    equity = cash + position_value + unrealized_pnl
    peak = state.get("peak_equity", equity)
    dd = (peak - equity) / peak * 100 if peak > 0 else 0
    total_trades = state.get("total_trades", 0)
    wins = state.get("wins", 0)
    losses = state.get("losses", 0)
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    pnl = state.get("total_pnl", 0)
    ret = (equity - 97.0) / 97.0 * 100

    # Build equity curve data from trades + current unrealized
    equity_points = [{"x": 0, "y": 97.0}]
    running_equity = 97.0
    for t in trades:
        running_equity = t.get("cash_after", running_equity)
        equity_points.append({
            "x": len(equity_points),
            "y": round(running_equity, 2)
        })
    # Add current equity (including unrealized) as last point
    if equity_points:
        equity_points[-1]["y"] = round(equity, 2)

    # Position tracker data for live P&L
    position_checks = tracker.get("checks", [])
    tracker_equity = []
    for c in position_checks:
        tracker_equity.append({
            "time": c.get("time", ""),
            "symbol": c.get("symbol", ""),
            "price": c.get("current_price", 0),
            "pnl_pct": c.get("pnl_pct", 0),
        })

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Position rows with unrealized P&L
    pos_rows = ""
    for symbol, pos in positions.items():
        pair = symbol.replace("_USDT_USDT", "")
        side = "LONG" if pos.get("side") == 1 else "SHORT"
        side_color = "#22c55e" if side == "LONG" else "#ef4444"
        entry = pos.get("entry_price", 0)
        size = pos.get("size_usd", pos.get("size", 0))
        strat = pos.get("strategy", "Unknown")
        entry_time = pos.get("entry_time", "N/A")
        if len(entry_time) > 19:
            entry_time = entry_time[:19]
        current = latest_prices.get(symbol, entry)
        if entry > 0 and pos.get("side", 0) == 1:
            upnl_pct = (current - entry) / entry * 100
            upnl_usd = size * (current - entry) / entry
        elif entry > 0:
            upnl_pct = (entry - current) / entry * 100
            upnl_usd = size * (entry - current) / entry
        else:
            upnl_pct = 0
            upnl_usd = 0
        upnl_color = "#22c55e" if upnl_pct > 0 else "#ef4444" if upnl_pct < 0 else "#9ca3af"

        pos_rows += f"""
        <tr>
            <td><strong>{pair}</strong></td>
            <td style="color: {side_color}; font-weight: bold;">{side}</td>
            <td>${entry:,.2f}</td>
            <td>${current:,.2f}</td>
            <td>${size:,.2f}</td>
            <td style="color: {upnl_color}; font-weight: bold;">{upnl_pct:+.2f}% (${upnl_usd:+.2f})</td>
            <td>{strat}</td>
            <td>{entry_time}</td>
        </tr>"""

    if not pos_rows:
        pos_rows = '<tr><td colspan="8" style="text-align: center; color: #9ca3af;">No open positions</td></tr>'

    # Trade rows
    trade_rows = ""
    for t in reversed(trades[-20:]):  # Last 20 trades
        pair = t.get("pair", "").replace("_USDT_USDT", "")
        action = t.get("action", "")
        price = t.get("price", t.get("exit_price", 0))
        size = t.get("size_usd", 0)
        pnl_val = t.get("pnl_usd", 0)
        pnl_pct = t.get("pnl_pct", 0)
        reason = t.get("reason", "")
        strategy = t.get("strategy", "")
        ts = t.get("timestamp", "")
        if len(ts) > 19:
            ts = ts[:19]

        if action == "CLOSE":
            color = "#22c55e" if pnl_val > 0 else "#ef4444"
            trade_rows += f"""
        <tr>
            <td>{ts}</td>
            <td>{pair}</td>
            <td><span style="color: #6b7280;">{action}</span></td>
            <td>${price:,.2f}</td>
            <td style="color: {color}; font-weight: bold;">${pnl_val:+.2f}</td>
            <td style="color: {color};">{pnl_pct:+.1f}%</td>
            <td>{reason}</td>
            <td>{strategy}</td>
        </tr>"""
        else:
            trade_rows += f"""
        <tr>
            <td>{ts}</td>
            <td>{pair}</td>
            <td><span style="color: #3b82f6;">{action}</span></td>
            <td>${price:,.2f}</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>{strategy}</td>
        </tr>"""

    if not trade_rows:
        trade_rows = '<tr><td colspan="8" style="text-align: center; color: #9ca3af;">No trades yet</td></tr>'

    # Run history rows
    run_rows = ""
    for r in reversed(run_history[-10:]):  # Last 10 runs
        status = r.get("status", "unknown")
        status_color = "#22c55e" if status == "success" else "#f59e0b" if status == "partial" else "#ef4444"
        ts = r.get("timestamp", "")
        if len(ts) > 19:
            ts = ts[:19]
        run_rows += f"""
        <tr>
            <td>{ts}</td>
            <td style="color: {status_color}; font-weight: bold;">{status.upper()}</td>
            <td>{r.get('trades', 0)}</td>
            <td>{r.get('duration_seconds', r.get('duration', 0)):.1f}s</td>
            <td>{len(r.get('errors', []))}</td>
        </tr>"""

    if not run_rows:
        run_rows = '<tr><td colspan="5" style="text-align: center; color: #9ca3af;">No run history</td></tr>'

    # Position tracker data
    tracker_rows = ""
    for c in reversed(position_checks[-10:]):
        ts = c.get("time", "")
        if len(ts) > 19:
            ts = ts[:19]
        pair = c.get("symbol", "").replace("_USDT_USDT", "")
        price = c.get("current_price", 0)
        pnl_pct = c.get("pnl_pct", 0)
        color = "#22c55e" if pnl_pct > 0 else "#ef4444" if pnl_pct < 0 else "#9ca3af"
        tracker_rows += f"""
        <tr>
            <td>{ts}</td>
            <td>{pair}</td>
            <td>${price:,.2f}</td>
            <td style="color: {color}; font-weight: bold;">{pnl_pct:+.2f}%</td>
        </tr>"""

    if not tracker_rows:
        tracker_rows = '<tr><td colspan="4" style="text-align: center; color: #9ca3af;">No tracker data</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <meta http-equiv="refresh" content="60">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ font-size: 28px; color: #f8fafc; }}
        .header .subtitle {{ color: #94a3b8; margin-top: 5px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
        .card .label {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
        .card .value {{ font-size: 28px; font-weight: 700; margin-top: 5px; }}
        .card .change {{ font-size: 14px; margin-top: 3px; }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .neutral {{ color: #94a3b8; }}
        .section {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }}
        .section h2 {{ font-size: 18px; margin-bottom: 15px; color: #f8fafc; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; padding: 10px; border-bottom: 2px solid #334155; color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 10px; border-bottom: 1px solid #1e293b; font-size: 14px; }}
        tr:hover {{ background: #1e293b; }}
        .chart-container {{ height: 300px; position: relative; }}
        .risk-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
        .risk-low {{ background: #065f46; color: #34d399; }}
        .risk-med {{ background: #78350f; color: #fbbf24; }}
        .risk-high {{ background: #7f1d1d; color: #f87171; }}
        .footer {{ text-align: center; color: #475569; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Trading Bot Dashboard</h1>
        <div class="subtitle">Paper Mode | $97 Starting Capital | Updated: {now}</div>
    </div>

    <!-- Key Metrics -->
    <div class="grid">
        <div class="card">
            <div class="label">Equity</div>
            <div class="value">${equity:,.2f}</div>
            <div class="change {'positive' if ret > 0 else 'negative'}">{ret:+.2f}%</div>
        </div>
        <div class="card">
            <div class="label">Cash</div>
            <div class="value">${cash:,.2f}</div>
            <div class="change neutral">{cash/max(equity,1)*100:.0f}% of equity</div>
        </div>
        <div class="card">
            <div class="label">Unrealized P&L</div>
            <div class="value {'positive' if unrealized_pnl > 0 else 'negative'}">${unrealized_pnl:+,.2f}</div>
            <div class="change neutral">{len(positions)} open positions</div>
        </div>
        <div class="card">
            <div class="label">Total Trades</div>
            <div class="value">{total_trades}</div>
            <div class="change neutral">{wins}W / {losses}L</div>
        </div>
        <div class="card">
            <div class="label">Win Rate</div>
            <div class="value {'positive' if wr > 50 else 'negative'}">{wr:.1f}%</div>
            <div class="change neutral">Target: 60%+</div>
        </div>
        <div class="card">
            <div class="label">Realized P&L</div>
            <div class="value {'positive' if pnl > 0 else 'negative'}">${pnl:+.2f}</div>
            <div class="change neutral">From {total_trades} trades</div>
        </div>
        <div class="card">
            <div class="label">Max Drawdown</div>
            <div class="value {'positive' if dd < 5 else 'negative'}">{dd:.1f}%</div>
            <div class="change {'risk-low' if dd < 5 else 'risk-med' if dd < 10 else 'risk-high'}">
                {'LOW RISK' if dd < 5 else 'MEDIUM' if dd < 10 else 'HIGH RISK'}
            </div>
        </div>
    </div>

    <!-- Open Positions -->
    <div class="section">
        <h2>📋 Open Positions ({len(positions)})</h2>
        <table>
            <thead>
                <tr><th>Pair</th><th>Side</th><th>Entry</th><th>Current</th><th>Size</th><th>Unrealized P&L</th><th>Strategy</th><th>Opened</th></tr>
            </thead>
            <tbody>{pos_rows}</tbody>
        </table>
    </div>

    <!-- Equity Curve -->
    <div class="section">
        <h2>📈 Equity Curve</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <!-- Position Tracker -->
    <div class="section">
        <h2>🔍 Position Tracker (Last 10 Checks)</h2>
        <table>
            <thead>
                <tr><th>Time</th><th>Pair</th><th>Price</th><th>P&L</th></tr>
            </thead>
            <tbody>{tracker_rows}</tbody>
        </table>
    </div>

    <!-- Trade History -->
    <div class="section">
        <h2>📊 Trade History (Last 20)</h2>
        <table>
            <thead>
                <tr><th>Time</th><th>Pair</th><th>Action</th><th>Price</th><th>P&L ($)</th><th>P&L (%)</th><th>Reason</th><th>Strategy</th></tr>
            </thead>
            <tbody>{trade_rows}</tbody>
        </table>
    </div>

    <!-- Bot Runs -->
    <div class="section">
        <h2>⚙️ Bot Runs (Last 10)</h2>
        <table>
            <thead>
                <tr><th>Time</th><th>Status</th><th>Trades</th><th>Duration</th><th>Errors</th></tr>
            </thead>
            <tbody>{run_rows}</tbody>
        </table>
    </div>

    <div class="footer">
        Auto-refreshes every 60 seconds | Bot runs every 15 minutes on GitHub Actions
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        const equityData = {json.dumps(equity_points)};
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: equityData.map(d => d.x),
                datasets: [{{
                    label: 'Equity ($)',
                    data: equityData.map(d => d.y),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: equityData.length > 50 ? 0 : 3,
                }}, {{
                    label: 'Starting ($97)',
                    data: equityData.map(() => 97),
                    borderColor: '#6b7280',
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#94a3b8' }} }}
                }},
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        ticks: {{ color: '#94a3b8', callback: v => '$' + v.toFixed(0) }},
                        grid: {{ color: '#1e293b' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    DASHBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard generated: {DASHBOARD_FILE}")
    print(f"Open in browser: file:///{DASHBOARD_FILE.resolve()}")


if __name__ == "__main__":
    generate_dashboard()
