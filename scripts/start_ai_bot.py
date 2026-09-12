#!/usr/bin/env python3
"""AI Trading Bot — CLI entry point.

Commands:
    once [NAME]   Run a single wakeup now (NAME = schedule entry, optional)
    run           Run the continuous 6x/day schedule (Ctrl+C to stop)
    next          Wait for and run the next scheduled wakeup
    status        Show current bot state (equity, positions, last wakeup)
    journal [N]   Show the last N journal entries (default 5)
    telegram-test Send a test alert to verify Telegram delivery

Examples:
    python scripts/start_ai_bot.py once
    python scripts/start_ai_bot.py once us_open
    python scripts/start_ai_bot.py run
    python scripts/start_ai_bot.py status
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_system.utils.logging import setup_logging

console = Console()

# Windows consoles default to cp1252 and crash on non-ASCII prints.
# Force a UTF-8-tolerant stdout/stderr before anything prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_config(path: str) -> dict:
    import yaml
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _build_agent(config_path: str, mode_override: str | None = None,
                 data_dir_override: str | None = None):
    """Load .env, build exchange + agent. Raises a clean error if misconfigured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv optional; env vars can be set manually

    import os

    cfg = _load_config(config_path)
    bot_cfg = cfg.get("bot", {})

    mode = mode_override or bot_cfg.get("mode", "paper")
    data_dir = data_dir_override or bot_cfg.get("data_dir", "data/ai_bot")

    if mode == "live" and not os.environ.get("AI_BOT_LIVE_CONFIRMED"):
        raise click.ClickException(
            "LIVE mode requires AI_BOT_LIVE_CONFIRMED=1 in the environment.\n"
            "Paper mode is the safe default: run without --live first."
        )

    from trading_system.bot.ai_agent import AIAgent
    from trading_system.bot.exchange import ExchangeInterface
    from trading_system.bot.scheduler import DEFAULT_SCHEDULE, Scheduler
    from trading_system.config import ExchangeConfig

    ex_cfg = ExchangeConfig()
    sandbox = bool(bot_cfg.get("sandbox", False))
    if sandbox:
        # Testnet rehearsal: keys from BINANCE_TESTNET_* env vars, sandbox on.
        ex_cfg.api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        ex_cfg.api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        ex_cfg.sandbox = True
    else:
        ex_cfg.api_key = os.environ.get("BINANCE_API_KEY", "")
        ex_cfg.api_secret = os.environ.get("BINANCE_API_SECRET", "")
        # Paper mode still needs market data; public endpoints work without keys.
        ex_cfg.sandbox = False

    exchange = ExchangeInterface(ex_cfg)
    if not exchange.connect():
        raise click.ClickException(
            "Could not connect to Binance. Check your internet connection."
        )

    agent = AIAgent(
        exchange=exchange,
        data_dir=data_dir,
        config=cfg,
        mode=mode,
    )

    schedule = cfg.get("schedule") or DEFAULT_SCHEDULE
    scheduler = Scheduler(
        agent_fn=agent.run_wakeup,
        schedule=schedule,
        tz_offset_hours=float(bot_cfg.get("tz_offset_hours", 0)),
    )
    return agent, scheduler, cfg


def _paper_account(agent) -> tuple[float, list[dict]]:
    """Paper-mode account summary without running a wakeup."""
    if agent.ledger is None:
        return 0.0, []
    prices = agent._current_prices(list(agent.ledger.positions.keys()) or agent.pairs)
    equity = agent.ledger.equity(prices)
    positions = agent.ledger.open_positions_list()
    for pos in positions:
        price = prices.get(pos["pair"], pos["entry_price"])
        direction = 1.0 if pos["side"] == "long" else -1.0
        pos["unrealized_pnl"] = round(
            direction * (price - pos["entry_price"]) * pos["size"], 2
        )
    return equity, positions


def _print_status(agent, cfg: dict) -> None:
    mode = agent.mode
    if mode == "paper":
        equity, positions = _paper_account(agent)
        start = agent.ledger.data.get("start_equity", equity)
        peak = agent.ledger.data.get("peak_equity", equity)
        ret = (equity - start) / start * 100 if start else 0.0
        dd = (peak - equity) / peak * 100 if peak else 0.0
    else:
        equity, positions, _ = agent._get_equity_and_positions(agent.pairs)
        start = peak = equity
        ret = dd = 0.0

    console.print(Panel.fit(
        f"[bold]AI Trading Bot — {mode.upper()} mode[/bold]\n"
        f"Model: {agent.ai.model} | Pairs: {', '.join(agent.pairs)}",
        border_style="blue",
    ))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Metric ")
    table.add_column("Value ", justify="right")
    table.add_row("Equity", f"${equity:,.2f}", "Return", f"{ret:+.2f}%")
    if mode == "paper":
        table.add_row("Peak equity", f"${peak:,.2f}", "Drawdown", f"{dd:.2f}%")
        closed = agent.ledger.data.get("closed_trades", [])
        wins = [t for t in closed if t.get("net_pnl", 0) > 0]
        wr = len(wins) / len(closed) * 100 if closed else 0.0
        table.add_row("Closed trades", str(len(closed)), "Win rate", f"{wr:.0f}%")
    console.print(table)

    if positions:
        pt = Table(show_header=True, header_style="bold")
        pt.add_column("Pair")
        pt.add_column("Side")
        pt.add_column("Entry", justify="right")
        pt.add_column("Size", justify="right")
        pt.add_column("P&L", justify="right")
        for p in positions:
            side = "LONG" if p.get("side") == "long" else "SHORT"
            pnl = p.get("unrealized_pnl", 0)
            color = "green" if pnl >= 0 else "red"
            pt.add_row(
                p["pair"], side, f"${p['entry_price']:,.2f}",
                f"${abs(p.get('notional', 0)):,.2f}",
                f"[{color}]${pnl:+,.2f}[/]",
            )
        console.print(pt)
    else:
        console.print("No open positions.")

    progress = agent._read_progress()
    if progress.get("last_wakeup"):
        console.print(f"\nLast wakeup: {progress['last_wakeup']}")
        console.print(f"Notes: {progress.get('notes', '')}")

    journal = agent.data_dir / "journal.jsonl"
    if journal.exists():
        lines = journal.read_text().strip().splitlines()
        console.print(f"Journal entries: {len(lines)} (at {journal})")
    console.print(f"Data dir: {agent.data_dir}")


def _print_journal(agent, n: int) -> None:
    journal = agent.data_dir / "journal.jsonl"
    if not journal.exists():
        console.print("[yellow]No journal yet — run a wakeup first.[/yellow]")
        return
    lines = journal.read_text().strip().splitlines()[-n:]
    for line in lines:
        import json
        e = json.loads(line)
        style = "green" if e.get("status") == "success" else "red"
        console.print(Panel(
            f"Outlook: {e.get('market_outlook', '?')} | "
            f"Equity: ${e.get('equity', 0):,.2f} | "
            f"Approved/Executed: {e.get('actions_approved', 0)}/{e.get('actions_executed', 0)}\n"
            f"Reasoning: {e.get('ai_reasoning', '')[:200]}\n"
            f"Errors: {e.get('errors') or 'none'}",
            title=f"[{style}]'{e.get('wakeup_id', '?')}' [{e.get('status', '?')}][/{style}]",
        ))


@click.group()
@click.option("--config", default="configs/ai_bot.yaml", help="Config file path")
@click.pass_context
def cli(ctx, config: str):
    """AI Trading Bot — LLM-driven crypto trading with hard risk limits."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.argument("name", required=False)
@click.option("--live", is_flag=True, help="Use live mode (requires AI_BOT_LIVE_CONFIRMED=1)")
@click.pass_context
def once(ctx, name: str | None, live: bool):
    """Run a single wakeup now (optionally a named schedule entry)."""
    setup_logging()
    mode = "live" if live else None
    agent, scheduler, cfg = _build_agent(ctx.obj["config"], mode_override=mode)
    try:
        result = scheduler.run_once(name)
    except ValueError as e:
        raise click.ClickException(str(e))
    sys.exit(0 if result.get("status") == "success" else 1)


@cli.command()
@click.option("--live", is_flag=True, help="Use live mode (requires AI_BOT_LIVE_CONFIRMED=1)")
@click.pass_context
def run(ctx, live: bool):
    """Run the continuous daily schedule until Ctrl+C."""
    setup_logging()
    mode = "live" if live else None
    agent, scheduler, cfg = _build_agent(ctx.obj["config"], mode_override=mode)
    scheduler.run_forever()


@cli.command("next")
@click.option("--live", is_flag=True, help="Use live mode (requires AI_BOT_LIVE_CONFIRMED=1)")
@click.pass_context
def next_(ctx, live: bool):
    """Wait for and run the next scheduled wakeup."""
    setup_logging()
    mode = "live" if live else None
    agent, scheduler, cfg = _build_agent(ctx.obj["config"], mode_override=mode)
    scheduler.run_next()


@cli.command()
@click.pass_context
def status(ctx):
    """Show current bot state (no wakeup, no AI call)."""
    setup_logging()
    agent, _, _ = _build_agent(ctx.obj["config"])
    _print_status(agent, {})


@cli.command()
@click.argument("n", default=5, type=int)
@click.pass_context
def journal(ctx, n: int):
    """Show the last N journal entries."""
    setup_logging()
    agent, _, _ = _build_agent(ctx.obj["config"])
    _print_journal(agent, n)


@cli.command("telegram-test")
@click.pass_context
def telegram_test(ctx):
    """Send a test alert to verify Telegram is configured correctly."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    import os

    from trading_system.bot.telegram_notifier import TelegramNotifier

    cfg = _load_config(ctx.obj["config"])
    tg_cfg = cfg.get("telegram", {})
    # AI_-prefixed names take precedence so the AI bot and the main bot can
    # use different Telegram bots without clashing.
    token = (
        os.environ.get("AI_TELEGRAM_BOT_TOKEN")
        or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    chat_id = (
        os.environ.get("AI_TELEGRAM_CHAT_ID")
        or os.environ.get("TELEGRAM_CHAT_ID", "")
    )

    problems = []
    if not token:
        problems.append(
            "AI_TELEGRAM_BOT_TOKEN (or TELEGRAM_BOT_TOKEN) is not set (env or .env)"
        )
    if not chat_id:
        problems.append(
            "AI_TELEGRAM_CHAT_ID (or TELEGRAM_CHAT_ID) is not set (env or .env)"
        )
    if not tg_cfg.get("enabled", False):
        problems.append(
            f"telegram.enabled is false in {ctx.obj['config']}"
        )
    if problems:
        for p in problems:
            console.print(f"[red]- {p}[/red]")
        raise click.ClickException("Fix the above, then re-run this command.")

    notifier = TelegramNotifier(bot_token=token, chat_id=chat_id, enabled=True)

    if not notifier.test_connection():
        raise click.ClickException(
            "Bot token rejected by Telegram (getMe failed). Double-check "
            "AI_TELEGRAM_BOT_TOKEN — it looks like 123456:ABC-DEF..."
        )

    console.print(f"Bot token OK. Sending test message to chat {chat_id}...")
    if notifier.send_test_message():
        console.print("[green]Test alert sent — check your Telegram.[/green]")
        console.print(
            "If nothing arrived: open your bot in Telegram and send /start once "
            "(bots cannot initiate a chat first), then verify the chat id."
        )
    else:
        raise click.ClickException(
            "Token is valid but the message failed — the chat id is likely "
            "wrong, or you have not sent /start to the bot yet."
        )


if __name__ == "__main__":
    cli(obj={})
