"""
Trading Bot Dashboard

A lightweight HTML dashboard that reads bot state and trade logs
to show current positions, recent trades, and performance metrics.

Usage:
  python -m trading_system.bot.dashboard  # Start on port 8080
  python -m trading_system.bot.dashboard --port 3000  # Custom port
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

STATE_FILE = Path("data/bot_state.json")
TRADE_LOG = Path("data/results/trade_log.jsonl")
DASHBOARD_DATA = Path("data/results/dashboard_data.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_trades(limit: int = 100) -> list[dict]:
    trades = []
    if TRADE_LOG.exists():
        with open(TRADE_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        pass
    return trades[-limit:]


def compute_metrics(trades: list[dict], state: dict) -> dict:
    """Compute performance metrics from trade data."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "avg_pnl": 0, "best_trade": 0, "worst_trade": 0,
            "sharpe": 0, "max_dd": 0, "equity": 10000,
        }

    pnls = [t.get("pnl_pct", 0) for t in trades if "pnl_pct" in t]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    equity = state.get("equity", 10000)
    total_pnl = sum(pnls)

    # Drawdown
    peak = equity
    max_dd = 0
    running = equity
    for pnl in pnls:
        running *= (1 + pnl / 100)
        peak = max(peak, running)
        dd = (peak - running) / peak
        max_dd = max(max_dd, dd)

    # Sharpe (simplified)
    if len(pnls) > 1:
        import numpy as np
        arr = np.array(pnls)
        sharpe = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252)) if np.std(arr) > 0 else 0
    else:
        sharpe = 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
        "total_pnl": total_pnl,
        "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
        "best_trade": max(pnls) if pnls else 0,
        "worst_trade": min(pnls) if pnls else 0,
        "sharpe": sharpe,
        "max_dd": max_dd * 100,
        "equity": equity,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Trading Bot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid #1a1a2e;
        }
        .header h1 {
            font-size: 24px;
            color: #00d4ff;
        }
        .status-badge {
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        .status-paper { background: #1a3a1a; color: #4ade80; border: 1px solid #4ade80; }
        .status-live { background: #3a1a1a; color: #f87171; border: 1px solid #f87171; }
        .status-off { background: #1a1a1a; color: #888; border: 1px solid #555; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: #111118;
            border: 1px solid #1a1a2e;
            border-radius: 12px;
            padding: 20px;
        }
        .card h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #1a1a2e;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #888; }
        .metric-value { font-weight: 600; font-variant-numeric: tabular-nums; }
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        .neutral { color: #e0e0e0; }

        .big-number {
            font-size: 36px;
            font-weight: 700;
            margin: 8px 0;
        }

        .strategies {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .strategy-card {
            background: #111118;
            border: 1px solid #1a1a2e;
            border-radius: 12px;
            padding: 16px;
        }
        .strategy-card .name {
            font-weight: 600;
            margin-bottom: 8px;
        }
        .signal-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .signal-long { background: #1a3a1a; color: #4ade80; }
        .signal-short { background: #3a1a1a; color: #f87171; }
        .signal-flat { background: #1a1a1a; color: #888; }

        .trades-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .trades-table th {
            text-align: left;
            padding: 10px 12px;
            background: #0a0a0f;
            color: #888;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid #1a1a2e;
        }
        .trades-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #111118;
        }
        .trades-table tr:hover { background: #111118; }

        .footer {
            text-align: center;
            color: #555;
            font-size: 12px;
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid #1a1a2e;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Trading Bot Dashboard</h1>
        <div id="status-badge" class="status-badge status-off">OFFLINE</div>
    </div>

    <div class="grid" id="metrics-grid">
        <div class="card">
            <h3>Equity</h3>
            <div class="big-number" id="equity">$10,000.00</div>
            <div class="metric">
                <span class="metric-label">Total P&L</span>
                <span class="metric-value" id="total-pnl">+0.00%</span>
            </div>
        </div>
        <div class="card">
            <h3>Performance</h3>
            <div class="metric">
                <span class="metric-label">Win Rate</span>
                <span class="metric-value" id="win-rate">0%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Sharpe Ratio</span>
                <span class="metric-value" id="sharpe">0.00</span>
            </div>
            <div class="metric">
                <span class="metric-label">Max Drawdown</span>
                <span class="metric-value" id="max-dd">0.00%</span>
            </div>
        </div>
        <div class="card">
            <h3>Trade Stats</h3>
            <div class="metric">
                <span class="metric-label">Total Trades</span>
                <span class="metric-value" id="total-trades">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Avg P&L</span>
                <span class="metric-value" id="avg-pnl">0.00%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Best Trade</span>
                <span class="metric-value" id="best-trade">+0.00%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Worst Trade</span>
                <span class="metric-value" id="worst-trade">-0.00%</span>
            </div>
        </div>
    </div>

    <h2 style="font-size: 18px; margin-bottom: 12px; color: #888;">🧠 Active Strategies</h2>
    <div class="strategies" id="strategies">
        <div class="strategy-card">
            <div class="name">MACD ETH 4h</div>
            <span class="signal-badge signal-flat">FLAT</span>
        </div>
        <div class="strategy-card">
            <div class="name">ROC_Momentum ETH 4h</div>
            <span class="signal-badge signal-flat">FLAT</span>
        </div>
        <div class="strategy-card">
            <div class="name">MACD BTC 4h</div>
            <span class="signal-badge signal-flat">FLAT</span>
        </div>
    </div>

    <h2 style="font-size: 18px; margin-bottom: 12px; color: #888;">📋 Recent Trades</h2>
    <div class="card" style="overflow-x: auto;">
        <table class="trades-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Pair</th>
                    <th>Side</th>
                    <th>Price</th>
                    <th>Amount</th>
                    <th>P&L</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody id="trades-body">
                <tr><td colspan="7" style="text-align: center; color: #555;">No trades yet</td></tr>
            </tbody>
        </table>
    </div>

    <div class="footer">
        Trading Bot Dashboard • Auto-refreshes every 30s • <span id="last-update">—</span>
    </div>

    <script>
        // Data will be injected by the server
        const DATA = __DASHBOARD_DATA__;

        function formatCurrency(n) {
            return '$' + n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        function formatPct(n) {
            const sign = n >= 0 ? '+' : '';
            return sign + n.toFixed(2) + '%';
        }
        function pnlClass(n) {
            if (n > 0) return 'positive';
            if (n < 0) return 'negative';
            return 'neutral';
        }

        // Update status badge
        const mode = DATA.state.mode || 'paper';
        const badge = document.getElementById('status-badge');
        if (mode === 'live') {
            badge.textContent = '🔴 LIVE';
            badge.className = 'status-badge status-live';
        } else if (mode === 'paper') {
            badge.textContent = '📝 PAPER';
            badge.className = 'status-badge status-paper';
        } else {
            badge.textContent = '⏸ ' + mode.toUpperCase();
            badge.className = 'status-badge status-off';
        }

        // Update metrics
        const m = DATA.metrics;
        document.getElementById('equity').textContent = formatCurrency(m.equity);
        const pnlEl = document.getElementById('total-pnl');
        pnlEl.textContent = formatPct(m.total_pnl);
        pnlEl.className = 'metric-value ' + pnlClass(m.total_pnl);
        document.getElementById('win-rate').textContent = m.win_rate.toFixed(1) + '%';
        document.getElementById('sharpe').textContent = m.sharpe.toFixed(2);
        document.getElementById('max-dd').textContent = '-' + m.max_dd.toFixed(2) + '%';
        document.getElementById('total-trades').textContent = m.total_trades;
        const avgPnl = document.getElementById('avg-pnl');
        avgPnl.textContent = formatPct(m.avg_pnl);
        avgPnl.className = 'metric-value ' + pnlClass(m.avg_pnl);
        const bestEl = document.getElementById('best-trade');
        bestEl.textContent = formatPct(m.best_trade);
        bestEl.className = 'metric-value ' + pnlClass(m.best_trade);
        const worstEl = document.getElementById('worst-trade');
        worstEl.textContent = formatPct(m.worst_trade);
        worstEl.className = 'metric-value ' + pnlClass(m.worst_trade);

        // Update strategies
        const stratDiv = document.getElementById('strategies');
        stratDiv.innerHTML = '';
        DATA.strategies.forEach(s => {
            const sig = s.last_signal;
            let sigClass = 'signal-flat';
            let sigText = 'FLAT';
            if (sig === 1) { sigClass = 'signal-long'; sigText = '🟢 LONG'; }
            else if (sig === -1) { sigClass = 'signal-short'; sigText = '🔴 SHORT'; }
            stratDiv.innerHTML += `
                <div class="strategy-card">
                    <div class="name">${s.label}</div>
                    <span class="signal-badge ${sigClass}">${sigText}</span>
                    <div style="color: #888; font-size: 12px; margin-top: 8px;">Weight: ${(s.weight * 100).toFixed(0)}%</div>
                </div>
            `;
        });

        // Update trades
        const tbody = document.getElementById('trades-body');
        if (DATA.trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #555;">No trades yet</td></tr>';
        } else {
            tbody.innerHTML = '';
            DATA.trades.slice().reverse().slice(0, 20).forEach(t => {
                const pnl = t.pnl_pct || 0;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${t.timestamp || '—'}</td>
                    <td>${t.pair || '—'}</td>
                    <td>${t.side === 'buy' ? '🟢 Long' : '🔴 Short'}</td>
                    <td>${t.price ? formatCurrency(t.price) : '—'}</td>
                    <td>${t.amount ? t.amount.toFixed(6) : '—'}</td>
                    <td class="${pnlClass(pnl)}">${pnl ? formatPct(pnl) : '—'}</td>
                    <td>${t.reason || t.action || '—'}</td>
                `;
                tbody.appendChild(row);
            });
        }

        document.getElementById('last-update').textContent = DATA.last_update;
    </script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve the dashboard with injected data."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

            state = load_state()
            trades = load_trades(100)
            metrics = compute_metrics(trades, state)

            strategies = state.get("last_signals", {})
            strategy_list = []
            for pair, data in strategies.items():
                if isinstance(data, dict):
                    for sig_name, sig_val in data.get("signals", {}).items():
                        strategy_list.append({
                            "label": sig_name,
                            "pair": pair,
                            "weight": 0.33,
                            "last_signal": sig_val,
                        })

            dashboard_data = {
                "state": {"mode": state.get("mode", "paper")},
                "metrics": metrics,
                "strategies": strategy_list if strategy_list else [
                    {"label": "MACD ETH 4h", "pair": "ETH", "weight": 0.41, "last_signal": 0},
                    {"label": "ROC_Momentum ETH 4h", "pair": "ETH", "weight": 0.17, "last_signal": 0},
                    {"label": "MACD BTC 4h", "pair": "BTC", "weight": 0.43, "last_signal": 0},
                ],
                "trades": trades[-20:],
                "last_update": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }

            html = DASHBOARD_HTML.replace(
                "__DASHBOARD_DATA__",
                json.dumps(dashboard_data, default=str)
            )

            self.wfile.write(html.encode())

        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            state = load_state()
            trades = load_trades(100)
            metrics = compute_metrics(trades, state)
            data = {"state": state, "metrics": metrics, "trades": trades[-20:]}
            self.wfile.write(json.dumps(data, default=str).encode())

        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = 8080
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{port}")
    print(f"  • Main view:  http://localhost:{port}/")
    print(f"  • API status: http://localhost:{port}/api/status")
    print("  • Auto-refreshes every 30 seconds")
    print("  • Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
