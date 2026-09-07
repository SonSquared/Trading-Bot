"""
Trading Dashboard

Generates a comprehensive HTML dashboard showing:
  - Current positions with live P&L
  - Equity curve chart (using Chart.js)
  - Trade history table
  - Risk metrics and drawdown
  - Bot run history

Run locally: python scripts/trading_dashboard.py
Auto-refreshes every 60 seconds when opened in browser.
"""

import json
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
INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005


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

    trades = []
    if (RESULTS_DIR / "paper_trades.jsonl").exists():
        with open(RESULTS_DIR / "paper_trades.jsonl") as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except:
                    pass

    cash = state.get("cash", INITIAL_CAPITAL)
    positions = state.get("positions", {})
    
    # Get latest prices from position tracker
    position_checks = tracker.get("checks", [])
    tracker_data = tracker.get("position_data", {})
    latest_prices = {}
    for sym, data in tracker_data.items():
        latest_prices[sym] = data.get("last_price", 0)
    
    # Fallback to last check if tracker_data empty
    if not latest_prices:
        for c in position_checks:
            sym = c.get("symbol", "")
            latest_prices[sym] = c.get("current_price", 0)
    
    # Calculate unrealized P&L with fees
    unrealized_pnl = 0
    unrealized_pnl_pct = 0
    total_position_value = 0
    
    for sym, pos in positions.items():
        entry = pos.get("entry_price", 0)
        size = pos.get("size_usd", pos.get("size", 0))
        side = pos.get("side", 0)
        current = latest_prices.get(sym, entry)
        
        if entry > 0 and current > 0:
            qty = size / entry
            if side == 1:  # LONG
                upnl = qty * (current - entry)
            else:  # SHORT
                upnl = qty * (entry - current)
            
            # Deduct fees (entry + estimated exit)
            fees = size * FEE_RATE * 2
            net_pnl = upnl - fees
            
            unrealized_pnl += net_pnl
            total_position_value += size + net_pnl
    
    # Real equity = cash + position market value (cost + unrealized)
    equity = cash + total_position_value
    peak = state.get("peak_equity", max(equity, INITIAL_CAPITAL))
    dd = (peak - equity) / peak * 100 if peak > 0 else 0
    
    # Trade stats
    total_trades = state.get("total_trades", 0)
    wins = state.get("wins", 0)
    losses = state.get("losses", 0)
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    realized_pnl = state.get("total_pnl", 0)
    total_pnl = realized_pnl + unrealized_pnl
    ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Build equity curve data from trades
    equity_points = []
    running_equity = INITIAL_CAPITAL
    
    # Add starting point
    equity_points.append({
        "time": trades[0].get("timestamp", "")[:16] if trades else "",
        "equity": INITIAL_CAPITAL,
    })
    
    for t in trades:
        running_equity = t.get("cash_after", running_equity)
        ts = t.get("timestamp", "")[:16]
        equity_points.append({
            "time": ts,
            "equity": round(running_equity, 2),
        })
    
    # Add current equity as last point
    equity_points.append({
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
        "equity": round(equity, 2),
    })
    
    # Deduplicate times (keep last value per unique time)
    seen_times = {}
    for p in equity_points:
        t = p["time"]
        if t:
            seen_times[t] = p
    equity_points = list(seen_times.values()) if seen_times else equity_points

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Position rows
    pos_rows = ""
    for symbol, pos in positions.items():
        pair = symbol.replace("_USDT_USDT", "")
        side = "LONG" if pos.get("side") == 1 else "SHORT"
        side_color = "#22c55e" if side == "LONG" else "#ef4444"
        entry = pos.get("entry_price", 0)
        size = pos.get("size_usd", pos.get("size", 0))
        strat = pos.get("strategy", "Unknown")
        entry_time = pos.get("entry_time", "N/A")[:19] if pos.get("entry_time") else "N/A"
        current = latest_prices.get(symbol, entry)
        
        if entry > 0 and current > 0:
            qty = size / entry
            if pos.get("side", 0) == 1:
                upnl_pct = (current - entry) / entry * 100
                upnl_usd = qty * (current - entry) - size * FEE_RATE * 2
            else:
                upnl_pct = (entry - current) / entry * 100
                upnl_usd = qty * (entry - current) - size * FEE_RATE * 2
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
    for t in reversed(trades[-20:]):
        pair = t.get("pair", "").replace("_USDT_USDT", "")
        action = t.get("action", "")
        price = t.get("price", t.get("exit_price", 0))
        pnl_val = t.get("pnl_usd", 0)
        pnl_pct = t.get("pnl_pct", 0)
        reason = t.get("reason", "")
        strategy = t.get("strategy", "")
        ts = t.get("timestamp", "")[:19]

        if action == "CLOSE":
            color = "#22c55e" if pnl_val > 0 else "#ef4444"
            trade_rows += f"""
        <tr>
            <td>{ts}</td>
            <td>{pair}</td>
            <td style="color: #6b7280;">CLOSE</td>
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
            <td style="color: #3b82f6;">OPEN</td>
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
    for r in reversed(run_history[-10:]):
        status = r.get("status", "unknown")
        status_color = "#22c55e" if status == "success" else "#f59e0b" if status == "partial" else "#ef4444"
        ts = r.get("timestamp", "")[:19]
        run_rows += f"""
        <tr>
            <td>{ts}</td>
            <td style="color: {status_color}; font-weight: bold;">{status.upper()}</td>
            <td>{r.get('trades', 0)}</td>
            <td>{r.get('duration_seconds', 0):.1f}s</td>
            <td>{len(r.get('errors', []))}</td>
        </tr>"""

    if not run_rows:
        run_rows = '<tr><td colspan="5" style="text-align: center; color: #9ca3af;">No run history</td></tr>'

    # Risk level
    if dd < 3:
        risk_level = "LOW"
        risk_class = "risk-low"
    elif dd < 7:
        risk_level = "MEDIUM"
        risk_class = "risk-med"
    else:
        risk_level = "HIGH"
        risk_class = "risk-high"

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
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
        .card .label {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
        .card .value {{ font-size: 26px; font-weight: 700; margin-top: 5px; }}
        .card .change {{ font-size: 13px; margin-top: 3px; }}
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
        <h1>Trading Bot Dashboard</h1>
        <div class="subtitle">Paper Mode | ${INITIAL_CAPITAL:.0f} Starting Capital | Updated: {now}</div>
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
            <div class="change neutral">{len(positions)} open position{'s' if len(positions) != 1 else ''}</div>
        </div>
        <div class="card">
            <div class="label">Total P&L</div>
            <div class="value {'positive' if total_pnl > 0 else 'negative'}">${total_pnl:+,.2f}</div>
            <div class="change neutral">Realized + Unrealized</div>
        </div>
        <div class="card">
            <div class="label">Trades</div>
            <div class="value">{total_trades}</div>
            <div class="change neutral">{wins}W / {losses}L ({wr:.0f}% win)</div>
        </div>
        <div class="card">
            <div class="label">Max Drawdown</div>
            <div class="value {'positive' if dd < 5 else 'negative'}">{dd:.1f}%</div>
            <div class="change {risk_class}">{risk_level} RISK</div>
        </div>
    </div>

    <!-- Open Positions -->
    <div class="section">
        <h2>Open Positions ({len(positions)})</h2>
        <table>
            <thead>
                <tr><th>Pair</th><th>Side</th><th>Entry</th><th>Current</th><th>Size</th><th>P&L</th><th>Strategy</th><th>Opened</th></tr>
            </thead>
            <tbody>{pos_rows}</tbody>
        </table>
    </div>

    <!-- Equity Curve -->
    <div class="section">
        <h2>Equity Curve</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <!-- Trade History -->
    <div class="section">
        <h2>Trade History (Last 20)</h2>
        <table>
            <thead>
                <tr><th>Time</th><th>Pair</th><th>Action</th><th>Price</th><th>P&L ($)</th><th>P&L (%)</th><th>Reason</th><th>Strategy</th></tr>
            </thead>
            <tbody>{trade_rows}</tbody>
        </table>
    </div>

    <!-- Bot Runs -->
    <div class="section">
        <h2>Bot Runs (Last 10)</h2>
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
        
        // Create labels from timestamps
        const labels = equityData.map(d => {{
            if (!d.time) return '';
            const dt = new Date(d.time + 'Z');
            return dt.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
        }});
        
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Equity ($)',
                    data: equityData.map(d => d.equity),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: equityData.length > 50 ? 0 : 3,
                }}, {{
                    label: 'Starting (${INITIAL_CAPITAL:.0f})',
                    data: equityData.map(() => {INITIAL_CAPITAL}),
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
                    legend: {{ labels: {{ color: '#94a3b8' }} }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return '$' + context.parsed.y.toFixed(2);
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#94a3b8', maxTicksLimit: 10 }},
                        grid: {{ color: '#1e293b' }}
                    }},
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
