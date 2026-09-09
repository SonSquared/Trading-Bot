"""Operations reporting (plan MD Task 7): read-only aggregates, redacted.

Reports expose costs, funding, leverage, drawdown, exposure, health,
vetoes, and halts — never secrets. Rendering goes through the ledger's
redaction helper as a final safety net.
"""

from __future__ import annotations

from typing import Any, Mapping

from crypto_system.audit.ledger import redact
from crypto_system.research.statistics import max_drawdown


def render_weekly_report(state: Mapping[str, Any]) -> str:
    """Render the weekly operations report (pure function, no I/O)."""
    safe = redact(dict(state))
    equity = float(safe.get("equity", 0.0))
    start = float(safe.get("start_equity", equity))
    pnl = equity - start
    pnl_pct = (pnl / start * 100.0) if start else 0.0
    positions = safe.get("positions", {}) or {}
    n_trades = int(safe.get("n_trades", 0))
    halts = safe.get("halts", []) or []
    vetoes = safe.get("vetoes", []) or []

    gross = sum(
        float(p.get("qty", 0.0)) * float(p.get("price", p.get("entry", 0.0)))
        for p in positions.values()
    )
    dd = 0.0
    equity_curve = safe.get("equity_curve")
    if equity_curve:
        from pandas import Series

        dd = max_drawdown(Series([float(x) for x in equity_curve]))

    lines = [
        "=== WEEKLY REPORT (paper) ===",
        f"Equity:        ${equity:,.2f}  (start ${start:,.2f}, "
        f"P&L ${pnl:+,.2f} / {pnl_pct:+.2f}%)",
        f"Trades:        {n_trades}",
        f"Exposure:      gross ${gross:,.2f} across {len(positions)} position(s)",
        f"Worst DD:      {dd:.2%}" if equity_curve else "Worst DD:      n/a",
        "Positions:",
    ]
    if positions:
        for symbol, p in sorted(positions.items()):
            lines.append(
                f"  {symbol:<12} {str(p.get('side', '?')):<5} "
                f"qty {float(p.get('qty', 0.0)):.6g} @ {float(p.get('entry', 0.0)):.2f}"
            )
    else:
        lines.append("  (flat)")
    lines.append(f"Halts:        {'; '.join(map(str, halts)) or 'none'}")
    lines.append(f"Vetoes:       {'; '.join(map(str, vetoes)) or 'none'}")
    lines.append(
        "NOTE: paper trading only. No profitability is promised or implied."
    )
    return "\n".join(lines)
