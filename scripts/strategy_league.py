"""
Strategy league: an honest, recurring tournament for the portfolio's slots.

Every candidate is evaluated with the SAME walk-forward folds and the SAME
honest selection machinery as the monthly re-optimizer
(scripts.monthly_reoptimize.select_params_honest: train-only selection, a
full out-of-sample matrix, no window cherry-picking). The incumbent is
evaluated on the identical test windows. A challenger is promoted ONLY if
it beats the incumbent OOS on ALL of:

  1. mean out-of-sample return
  2. profitable-window fraction (>= incumbent's)
  3. worst-window return (robustness floor, and never worse than -15%)

If nothing clears the gate, the incumbent stays and the report says so.
This makes "constantly finding better setups" safe by construction: the
portfolio can only ever hold configs with positive multi-window OOS
evidence — never a curve-fit hopeful.

Usage:
    python scripts/strategy_league.py [--days 950] [--max-combos 300]
                                      [--dry-run] [--pair ETH_USDT_USDT]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

import scripts.paper_trader as pt
from scripts.monthly_reoptimize import (
    backtest_single,
    select_params_honest,
    walk_forward_windows,
)
from scripts.watchdog import send_telegram

RESULTS_DIR = Path("data/results")
REPORT_MD = RESULTS_DIR / "strategy_league_report.md"
REPORT_JSON = RESULTS_DIR / "strategy_league_report.json"
PARAMS_FILE = pt.OPTIMIZED_PARAMS_FILE
PARAMS_BACKUP = RESULTS_DIR / "bot_strategy_params.backup.json"

PAIRS = ["ETH_USDT_USDT", "BTC_USDT_USDT"]
TIMEFRAME = "4h"

# Five challengers from five distinct families for genuine diversity.
# Grids are trimmed to ~36-144 combos each: enough to optimize honestly,
# small enough that the tournament fits comfortably in CI time.
CANDIDATES: dict[str, dict] = {
    "Supertrend": {
        "family": "trend",
        "grid": {
            "period": [10, 14, 21],
            "multiplier": [2.0, 3.0, 4.0],
        },
    },
    "Donchian_Breakout": {
        "family": "trend_breakout",
        "grid": {
            "channel_period": [20, 40, 55],
            "exit_period": [10, 20],
            "atr_filter": [False, True],
            "atr_period": [14],
        },
    },
    "Keltner_Breakout": {
        "family": "volatility",
        "grid": {
            "ema_period": [20, 30],
            "atr_period": [10, 20],
            "multiplier": [1.5, 2.0, 2.5],
            "exit_multiplier": [0.5, 1.0],
        },
    },
    "BB_Squeeze": {
        "family": "vol_compression_mr",
        "grid": {
            "bb_period": [20],
            "bb_std": [2.0],
            "kc_period": [20],
            "kc_atr_mult": [1.0, 1.5],
            "squeeze_lookback": [60, 120],
            "momentum_period": [10, 20],
        },
    },
    "ZScore_Reversion": {
        "family": "statistical_mr",
        "grid": {
            "lookback": [30, 50, 100],
            "entry_threshold": [1.5, 2.0, 2.5],
            "exit_threshold": [0.5],
            "use_sma_baseline": [False, True],
        },
    },
}

# Robustness floor: no deployed config may have a window worse than this.
WORST_WINDOW_FLOOR_PCT = -15.0
DEFAULT_WEIGHTS = [0.40, 0.35, 0.25]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def fetch_data(pair: str, days: int) -> pd.DataFrame | None:
    """Fetch `days` of 4h candles, paginated.

    Kraken's OHLC endpoint serves at most ~720 candles regardless of
    ``since`` — deep history must come from Binance. Kraken is only
    consulted when the requested window fits inside its cap.
    """
    import ccxt

    symbol = pair.replace("_USDT_USDT", "/USDT").replace("_", "/")
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    max_candles = int(days * 24 * 3600 / (4 * 3600)) + 60

    # Kraken first only for short windows; Binance paginates honestly.
    if max_candles <= 700:
        exchanges = ((ccxt.kraken, 720), (ccxt.binance, 1000))
    else:
        exchanges = ((ccxt.binance, 1000), (ccxt.kraken, 720))

    for exchange_cls, limit in exchanges:
        try:
            exchange = exchange_cls({"enableRateLimit": True})
            out: list[list] = []
            since = since_ms
            while len(out) < max_candles:
                batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=limit)
                if not batch:
                    break
                out.extend(batch)
                if len(batch) < limit:
                    break
                since = batch[-1][0] + 1
                _time.sleep((getattr(exchange, "rateLimit", 0) or 500) / 1000.0)
            if len(out) < 60:
                continue
            df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
            print(f"  {exchange_cls.__name__}: {len(df)} candles for {pair} "
                  f"({(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days}d span)")
            return df
        except Exception as e:
            print(f"  {exchange_cls.__name__} failed for {pair}: {e}")
    return None


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_oos(strat_name: str, params: dict, data: pd.DataFrame,
                 windows: list[dict]) -> dict | None:
    """Evaluate FIXED params on every test window (the honest OOS matrix)."""
    returns: list[float] = []
    total_trades = 0
    for w in windows:
        test = data.iloc[w["test_start"]:w["test_end"]].reset_index(drop=True)
        if len(test) < 20:
            continue
        try:
            r = backtest_single(test, strat_name, params)
        except Exception:
            return None
        returns.append(r["return_pct"])
        total_trades += int(r.get("trades", 0))
    if not returns:
        return None
    n = len(returns)
    profitable = sum(1 for r in returns if r > 0)
    return {
        "mean": float(pd.Series(returns).mean()),
        "median": float(pd.Series(returns).median()),
        "min": float(pd.Series(returns).min()),
        "max": float(pd.Series(returns).max()),
        "profitable_windows": profitable,
        "total_windows": n,
        "profitable_frac": profitable / n,
        "window_returns": [round(r, 3) for r in returns],
        "n_trades": total_trades,
    }


def load_incumbent() -> dict:
    """The deployed config (optimized file if present, else defaults)."""
    incumbent = pt.load_optimized_strategies()
    if not incumbent:
        incumbent = pt.load_active_strategies()
    return incumbent or {}


def evaluate_incumbent_for_pair(cfg: dict, data: pd.DataFrame,
                                windows: list[dict]) -> dict | None:
    return evaluate_oos(cfg["strategy"], cfg.get("params", {}), data, windows)


def pick_benchmark_incumbent(incumbent: dict, pair: str,
                             data: pd.DataFrame,
                             windows: list[dict]) -> tuple[str | None, dict | None, dict | None]:
    """Choose the incumbent config a challenger must beat for `pair`.

    A pair may hold several configs (e.g. Bollinger 0.35 + RSI 0.25).
    Benchmarking against only the first (or the weakest) would let a
    challenger promote by beating a config it would never actually
    replace. The honest benchmark is the pair's BEST config by OOS mean.

    Returns (slot_key, cfg, eval). slot_key is the config the winner may
    replace; eval is None when no config for the pair is evaluable.
    """
    candidates = [(k, c) for k, c in incumbent.items()
                  if c.get("pair") == pair]
    if not candidates:
        return None, None, None
    best_key, best_cfg, best_eval = None, None, None
    evals = []
    for k, c in candidates:
        ev = evaluate_incumbent_for_pair(c, data, windows)
        evals.append((k, c, ev))
        if ev and (best_eval is None or ev["mean"] > best_eval["mean"]):
            best_key, best_cfg, best_eval = k, c, ev
    if best_key is None:
        # None evaluable — keep the first config as the reported benchmark.
        best_key, best_cfg = candidates[0]
        best_eval = None
    else:
        others = ", ".join(f"{c['strategy']} {e['mean']:+.2f}%" if e
                           else f"{c['strategy']} n/a" for k, c, e in evals)
        print(f"  benchmark: {best_cfg['strategy']} (best of: {others})")
    return best_key, best_cfg, best_eval


def challenger_wins(challenger: dict, incumbent: dict | None) -> tuple[bool, str]:
    """The promotion gate. Deliberately conservative: every metric must
    favor the challenger, not just the mean."""
    if challenger["mean"] <= 0:
        return False, "mean OOS return not positive"
    # A candidate that never trades in ANY OOS window is not a strategy,
    # it is a vacancy. Its 0.00% must never dress up as a result.
    if challenger.get("n_trades", 0) == 0:
        return False, "no trades in any OOS window (inactive)"
    if challenger["min"] <= WORST_WINDOW_FLOOR_PCT:
        return False, (f"worst window {challenger['min']:.1f}% breaches the "
                       f"{WORST_WINDOW_FLOOR_PCT:.0f}% floor")
    # Strict majority: 4 windows -> 3, 3 -> 2, 2 -> both. A challenger
    # that loses half its out-of-sample windows has no business trading.
    majority = challenger["total_windows"] // 2 + 1
    if challenger["profitable_windows"] < majority:
        return False, (f"profitable in only {challenger['profitable_windows']}/"
                       f"{challenger['total_windows']} windows (need majority)")
    if incumbent is None:
        return True, "no incumbent — first qualifying config"
    if challenger["mean"] <= incumbent["mean"]:
        return False, (f"mean {challenger['mean']:.2f}% <= incumbent "
                       f"{incumbent['mean']:.2f}%")
    if challenger["profitable_frac"] < incumbent["profitable_frac"]:
        return False, (f"profitable fraction {challenger['profitable_frac']:.2f} < "
                       f"incumbent {incumbent['profitable_frac']:.2f}")
    if challenger["min"] <= incumbent["min"]:
        return False, (f"worst window {challenger['min']:.2f}% <= incumbent "
                       f"{incumbent['min']:.2f}%")
    return True, "beats incumbent on mean, consistency, and worst case"


# --------------------------------------------------------------------------
# League round for one pair
# --------------------------------------------------------------------------

def run_league_for_pair(pair: str, data: pd.DataFrame, windows: list[dict],
                        incumbent_cfg: dict | None,
                        incumbent_eval: dict | None,
                        max_combos: int,
                        slot_key: str | None = None) -> dict:
    print(f"\n--- {pair}: {len(windows)} walk-forward window(s) ---")
    if incumbent_cfg:
        if incumbent_eval:
            print(f"  incumbent {incumbent_cfg['strategy']}: "
                  f"OOS mean {incumbent_eval['mean']:+.2f}% "
                  f"({incumbent_eval['profitable_windows']}/"
                  f"{incumbent_eval['total_windows']} profitable, "
                  f"worst {incumbent_eval['min']:+.2f}%)")
        else:
            print("  incumbent: not evaluable on these windows")

    results = []
    for name, spec in CANDIDATES.items():
        t0 = _time.time()
        try:
            best_params, evaluation, ok = select_params_honest(
                name, pair, data, spec["grid"], windows, max_combos=max_combos)
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            results.append({"strategy": name, "family": spec["family"],
                            "error": str(e), "ok_to_deploy": False})
            continue
        if not evaluation or best_params is None:
            results.append({"strategy": name, "family": spec["family"],
                            "error": "no candidates produced", "ok_to_deploy": False})
            continue
        # Re-evaluate the selected params ourselves for the comparable
        # per-window numbers (select_params_honest aggregates only).
        oos = evaluate_oos(name, best_params, data, windows)
        entry = {
            "strategy": name, "family": spec["family"],
            "params": best_params, "ok_to_deploy": bool(ok),
            "mean": evaluation.get("mean_test_return"),
            "profitable_windows": evaluation.get("profitable_windows"),
            "total_windows": evaluation.get("total_windows"),
            "oos": oos,
        }
        if oos:
            activity = f"{oos['n_trades']} trades" if oos.get("n_trades") else "INACTIVE"
            print(f"  {name:20s}: mean {oos['mean']:+6.2f}%  "
                  f"({oos['profitable_windows']}/{oos['total_windows']} prof, "
                  f"worst {oos['min']:+6.2f}%)  {activity}  [{_time.time()-t0:.0f}s]")
        results.append(entry)

    # Pick the winner among candidates that clear the absolute gate AND
    # beat the incumbent; fall back to incumbent retention.
    winner = None
    for entry in results:
        if not entry.get("oos"):
            continue
        ok, why = challenger_wins(entry["oos"], incumbent_eval)
        entry["gate"] = {"passed": ok, "reason": why}
        if ok and (winner is None or entry["oos"]["mean"] > winner["oos"]["mean"]):
            winner = entry

    if winner:
        print(f"  >>> WINNER: {winner['strategy']} — promoting "
              f"(OOS mean {winner['oos']['mean']:+.2f}%)")
    else:
        print("  >>> incumbent retained — no challenger cleared the gate")

    return {
        "pair": pair,
        "slot_key": slot_key,
        "windows": [{k: w[k] for k in ("train_start_date", "test_start_date",
                                       "test_end_date")} for w in windows],
        "incumbent": ({
            "strategy": incumbent_cfg["strategy"],
            "params": incumbent_cfg.get("params", {}),
            "oos": incumbent_eval,
        } if incumbent_cfg else None),
        "candidates": results,
        "winner": winner,
        "promoted": winner is not None,
    }


# --------------------------------------------------------------------------
# Deployment
# --------------------------------------------------------------------------

def deploy_winner(pair: str, round_result: dict, incumbent: dict) -> dict:
    """Write the winning config for `pair` into the params file the bot
    loads. Other pairs' entries are preserved; the previous file is backed
    up first.

    Only the BENCHMARK slot for the pair — the incumbent config the winner
    actually out-tested (round_result["slot_key"]) — is replaced. Other
    configs for the same pair were not evaluated by this league round;
    deleting unevaluated configs would be changing things without evidence.
    """
    winner = round_result["winner"]
    slot_key = round_result.get("slot_key")
    weight = None
    primary_key = None
    if slot_key and slot_key in incumbent:
        cfg = incumbent[slot_key]
        if cfg.get("pair") == pair:
            weight = float(cfg.get("weight", DEFAULT_WEIGHTS[0]))
            primary_key = slot_key
    if primary_key is None:
        # Fallback: first config for the pair (e.g. legacy round results).
        for key, cfg in incumbent.items():
            if cfg.get("pair") == pair:
                weight = float(cfg.get("weight", DEFAULT_WEIGHTS[0]))
                primary_key = key
                break
    if weight is None:
        weight = DEFAULT_WEIGHTS[0]

    new_config = {k: v for k, v in incumbent.items() if k != primary_key}
    new_config[f"{winner['strategy']}_{pair}"] = {
        "strategy": winner["strategy"],
        "pair": pair,
        "timeframe": TIMEFRAME,
        "weight": weight,
        "params": winner["params"],
    }
    return new_config


def apply_config(new_config: dict, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run] config NOT written")
        return
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PARAMS_FILE.exists():
        shutil.copy2(PARAMS_FILE, PARAMS_BACKUP)
    tmp = PARAMS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_config, indent=2), encoding="utf-8")
    tmp.replace(PARAMS_FILE)
    print(f"Deployed config -> {PARAMS_FILE} (backup: {PARAMS_BACKUP.name})")


def validate_deployed(new_config: dict) -> bool:
    """The deployed file must load cleanly through the bot's own loader."""
    saved = pt.OPTIMIZED_PARAMS_FILE
    try:
        pt.OPTIMIZED_PARAMS_FILE = PARAMS_FILE
        loaded = pt.load_active_strategies()
        return bool(loaded) and all(
            all(k in cfg for k in ("strategy", "pair", "timeframe", "weight", "params"))
            for cfg in loaded.values())
    finally:
        pt.OPTIMIZED_PARAMS_FILE = saved


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def fmt_oos(oos: dict | None) -> str:
    if not oos:
        return "n/a"
    return (f"mean {oos['mean']:+6.2f}% | {oos['profitable_windows']}/"
            f"{oos['total_windows']} profitable | worst {oos['min']:+6.2f}%")


def build_report(rounds: list[dict], days: int, promoted_any: bool) -> str:
    lines = [
        "# Strategy League Report",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        f"Data: trailing {days}d of 4h candles (Kraken) | "
        f"Walk-forward 12/6 months, anchored | Costs: 0.05% fee/side, "
        f"0.02% slippage/side, 8h funding",
        f"\n**Outcome: {'PROMOTED' if promoted_any else 'INCUMBENT RETAINED'}**",
    ]
    for r in rounds:
        lines.append(f"\n## {r['pair']}")
        if r.get("error"):
            lines.append(f"\n**SKIPPED**: {r['error']}")
            continue
        for w in r.get("windows", []):
            lines.append(f"- window: train {w['train_start_date']} → "
                         f"test {w['test_start_date']} .. {w['test_end_date']}")
        inc = r.get("incumbent")
        if inc:
            lines.append(f"\n**Incumbent** {inc['strategy']}: {fmt_oos(inc.get('oos'))}")
        lines.append("\n| Candidate | Family | OOS result | Gate |")
        lines.append("|---|---|---|---|")
        for c in r["candidates"]:
            gate = c.get("gate") or {"passed": False, "reason": c.get("error", "—")}
            lines.append(f"| {c['strategy']} | {c['family']} | "
                         f"{fmt_oos(c.get('oos'))} | "
                         f"{'PASS' if gate['passed'] else 'fail'}: {gate['reason']} |")
        if r.get("winner"):
            w = r["winner"]
            lines.append(f"\n**Winner: {w['strategy']}** — promoted. "
                         f"Params: `{json.dumps(w['params'])}`")
        else:
            lines.append("\n**No winner** — incumbent retained (nothing beat it "
                         "out-of-sample).")
    return "\n".join(lines)


def summarize_telegram(rounds: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"🏆 <b>STRATEGY LEAGUE</b> — {now}"]
    for r in rounds:
        if r.get("promoted"):
            w = r["winner"]
            lines.append(f"\n✅ <b>{r['pair']}: {w['strategy']} PROMOTED</b>\n"
                         f"OOS: {fmt_oos(w['oos'])}")
        else:
            inc = r.get("incumbent") or {}
            lines.append(f"\n⏸ {r['pair']}: incumbent retained "
                         f"({inc.get('strategy', 'n/a')}) — no challenger "
                         f"beat it OOS")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Honest strategy league")
    ap.add_argument("--days", type=int, default=950)
    ap.add_argument("--max-combos", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; never write the params file")
    ap.add_argument("--pair", action="append", dest="pairs",
                    help="restrict to specific pair(s), repeatable")
    args = ap.parse_args()
    pairs = args.pairs or PAIRS

    print("=" * 70)
    print("STRATEGY LEAGUE — honest walk-forward tournament")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    incumbent = load_incumbent()
    print(f"Incumbent configs: {list(incumbent.keys())}")

    rounds: list[dict] = []
    working_config: dict | None = None
    promoted_any = False

    for pair in pairs:
        data = fetch_data(pair, args.days)
        if data is None or len(data) < 400:
            print(f"  {pair}: insufficient data, skipping")
            rounds.append({"pair": pair, "error": "insufficient data",
                           "candidates": [], "promoted": False})
            continue
        windows = walk_forward_windows(data, train_months=12, test_months=6)
        if not windows:
            print(f"  {pair}: not enough span for walk-forward windows")
            rounds.append({"pair": pair, "error": "no walk-forward windows",
                           "candidates": [], "promoted": False})
            continue
        inc_slot, inc_cfg, inc_eval = pick_benchmark_incumbent(
            incumbent, pair, data, windows)
        r = run_league_for_pair(pair, data, windows, inc_cfg, inc_eval,
                                args.max_combos, slot_key=inc_slot)
        rounds.append(r)
        if r.get("promoted"):
            promoted_any = True
            # Chain promotions: each deploy builds on the previous one so
            # promoting pair B never overwrites pair A's promotion.
            base = working_config if working_config is not None else incumbent
            working_config = deploy_winner(pair, r, base)

    # Deploy once, after all rounds, if anything was promoted.
    if working_config is not None:
        apply_config(working_config, args.dry_run)
        if not args.dry_run and not validate_deployed(working_config):
            print("ERROR: deployed config failed validation — restoring backup")
            if PARAMS_BACKUP.exists():
                shutil.copy2(PARAMS_BACKUP, PARAMS_FILE)

    report = build_report(rounds, args.days, promoted_any)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(rounds, indent=2, default=str),
                           encoding="utf-8")
    print(f"\nReport: {REPORT_MD}")

    # Telegram
    token = (pt.load_telegram_config() or {})
    if token.get("bot_token") and token.get("chat_id"):
        try:
            send_telegram(token["bot_token"], str(token["chat_id"]),
                          summarize_telegram(rounds))
        except Exception as e:
            print(f"Telegram send failed: {e}")

    return 0 if rounds and not all(r.get("error") for r in rounds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
