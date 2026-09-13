"""
AI Trading Engine — LLM reasoning for crypto trading.

Defaults to Google Gemini's FREE tier (gemini-flash-latest alias, no
credit card); any OpenAI model works too via OPENAI_API_KEY.
Each call is stateless: the agent provides all context via the prompt,
and the engine returns a structured JSON trading decision.

Safety: the engine NEVER places orders directly. It returns a
TradingDecision dataclass that the agent must validate before execution.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeAction:
    """One atomic trade instruction."""
    pair: str
    side: str  # "long" | "short" | "close"
    size_pct: float  # % of equity to allocate (0-100)
    stop_loss_pct: float  # stop-loss distance as % from entry
    take_profit_pct: float  # take-profit distance as % from entry
    confidence: float  # 0-100, AI's self-assessed confidence
    reasoning: str  # human-readable why


@dataclass
class TradingDecision:
    """Structured output from the AI engine."""
    actions: list[TradeAction]
    market_outlook: str  # "bullish" | "bearish" | "neutral" | "volatile"
    risk_assessment: str  # brief risk note
    reasoning: str  # overall reasoning summary
    raw_response: str = ""  # raw model output for debugging
    model: str = "gpt-4o"
    ok: bool = True  # False when the engine call/parse failed — never trade on this
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def has_trades(self) -> bool:
        return len(self.actions) > 0


# ---------------------------------------------------------------------------
# AI Engine
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert crypto trading analyst and risk manager.
You analyze market data, technical indicators, and portfolio state to make
precise trading decisions for Binance USDⓈ-M perpetual futures.

## RULES — you MUST follow every one:

1. NEVER risk more than 2% of equity on a single trade.
2. ALWAYS set a stop-loss (max 5% from entry).
3. ALWAYS set a take-profit (at least 1.5x the stop-loss distance).
4. Maximum 3 open positions at any time.
5. Close losing positions that breach their stop-loss.
6. Do NOT open a new position if drawdown exceeds 10% from peak equity.
7. Prefer high-liquidity pairs (BTC, ETH) during volatile markets.
8. Confidence below 60 means DO NOT trade — return empty actions.
9. Never exceed 50% total portfolio exposure.

## OUTPUT FORMAT — respond with ONLY valid JSON (no markdown, no explanation):

{
  "actions": [
    {
      "pair": "BTC/USDT:USDT",
      "side": "long",
      "size_pct": 5.0,
      "stop_loss_pct": 3.0,
      "take_profit_pct": 6.0,
      "confidence": 75,
      "reasoning": "RSI oversold bounce at support, MACD crossover bullish"
    }
  ],
  "market_outlook": "bullish",
  "risk_assessment": "Low volatility, strong uptrend on 4h",
  "reasoning": "BTC showing accumulation pattern..."
}

If no trades should be taken, return: {"actions": [], "market_outlook": "...", ...}
"""


class AIEngine:
    """LLM trading decision engine.

    Defaults to Google's Gemini FREE tier via its OpenAI-compatible
    endpoint — free AI Studio key, no credit card, and 6 wakeups/day fit
    far inside the free rate limits. The primary model is the
    'gemini-flash-latest' alias, which always tracks the newest flash
    model; see MODEL_FALLBACKS for why a single retired/overloaded model
    can never kill a wakeup anymore. Any OpenAI model also works: set
    ai.model in the config and OPENAI_API_KEY.
    """

    # Gemini's OpenAI-compatible endpoint: same chat.completions API surface.
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Tried in order on ANY error (retired model 404, 429 rate limit, 503
    # "high demand", ...). History: gemini-2.5-flash was retired for new
    # keys (404), and gemini-flash-latest itself once returned 503 under
    # load — a wakeup dying on either is a silent trading gap. The first
    # entry is the evergreen alias; the bare 'gemini-flash' alias and the
    # long-lived gemini-2.0-flash are the safety net.
    MODEL_FALLBACKS: tuple[str, ...] = (
        "gemini-flash-latest",
        "gemini-flash",
        "gemini-2.0-flash",
    )

    # Backoff (seconds) consumed between consecutive attempts. Transient
    # 503/429 spikes usually clear within seconds; the spread keeps the
    # total retry window (~10s worst case) well inside the job budget.
    RETRY_BACKOFF: tuple[float, ...] = (2, 4, 4)

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy-init OpenAI client (avoids import at module level)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
            kwargs = {"api_key": ""}
            if self.model.startswith("gemini"):
                kwargs["api_key"] = os.environ.get("GEMINI_API_KEY", "")
                kwargs["base_url"] = self.GEMINI_BASE_URL
                if not kwargs["api_key"]:
                    raise ValueError(
                        "GEMINI_API_KEY is required for gemini models. Get a "
                        "FREE key at https://aistudio.google.com/apikey and set "
                        "it in your .env file."
                    )
            else:
                kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "")
                if not kwargs["api_key"]:
                    raise ValueError(
                        "OPENAI_API_KEY is required for OpenAI models — or set "
                        "ai.model to gemini-2.5-flash to use a FREE Gemini key "
                        "(https://aistudio.google.com/apikey)."
                    )
            self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _error_decision(reasoning: str, raw: str = "") -> TradingDecision:
        """An explicitly-failed decision. Callers must check .ok before trading."""
        return TradingDecision(
            actions=[],
            market_outlook="unknown",
            risk_assessment="engine failure",
            reasoning=reasoning,
            raw_response=raw,
            ok=False,
        )

    def decide(
        self,
        market_context: str,
        portfolio_context: str,
        strategy_context: str,
        risk_context: str,
    ) -> TradingDecision:
        """Generate a trading decision from assembled context.

        All context is pre-formatted strings — the engine does not fetch data.
        This keeps it pure and testable.

        Resilience: the configured model is tried first, then every entry of
        MODEL_FALLBACKS (gemini configs only), two attempts each, with real
        backoff between attempts. A retired model (404) or a demand spike
        (503/429) therefore degrades to a slower wakeup, never a dead one.
        Only when every model and attempt is exhausted does this return an
        explicitly-failed decision (ok=False) for the agent to report.
        """
        user_prompt = self._build_prompt(
            market_context, portfolio_context, strategy_context, risk_context
        )

        if self.model.startswith("gemini"):
            chain = (self.model,) + tuple(
                m for m in self.MODEL_FALLBACKS if m != self.model
            )
        else:
            # Non-gemini (e.g. OpenAI) config: fallback list is gemini-only.
            chain = (self.model,)

        sleeps = iter(self.RETRY_BACKOFF)

        def _nap() -> None:
            time.sleep(next(sleeps, 4))

        errors: list[str] = []
        for model_i, model in enumerate(chain):
            for attempt in (1, 2):
                logger.info("ai_decision_request", model=model, attempt=attempt)
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                    )

                    raw = (response.choices[0].message.content or "").strip()
                    logger.info(
                        "ai_decision_response",
                        model=model,
                        attempt=attempt,
                        raw_length=len(raw),
                    )

                    if not raw:
                        logger.error("ai_empty_response", model=model, attempt=attempt)
                        errors.append(f"{model}: empty response")
                    else:
                        decision = self._parse_response(raw)
                        decision.model = model
                        if model != self.model:
                            logger.warning(
                                "ai_model_fallback", failed=self.model, working=model
                            )
                            # Stick with the working model for any later
                            # calls in this process (one wakeup = one run).
                            self.model = model
                        return decision

                except Exception as e:  # noqa: BLE001 — every error retries/falls back
                    errors.append(f"{model}: {e}")
                    logger.warning(
                        "ai_decision_attempt_failed",
                        model=model,
                        attempt=attempt,
                        error=str(e),
                    )
                _nap()  # between the two attempts, and before the next model
            if model_i < len(chain) - 1:
                logger.warning("ai_model_exhausted", failed_model=model)

        logger.error("ai_decision_failed", attempts=len(errors), last=errors[-1] if errors else "?")
        return self._error_decision(
            f"Failed to get AI decision after trying {len(chain)} model(s) "
            f"x2 attempts — last: {errors[-1] if errors else 'unknown'}"
        )

    def _build_prompt(
        self,
        market: str,
        portfolio: str,
        strategies: str,
        risk: str,
    ) -> str:
        """Assemble the user prompt from pre-formatted context blocks."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return f"""## Current Time: {now}

## Market Data & Indicators
{market}

## Portfolio State
{portfolio}

## Strategy Signals
{strategies}

## Risk Status
{risk}

Based on the above, provide your trading decision as JSON.
Remember: only trade if confidence >= 60. Always use stop-losses.
"""

    def _parse_response(self, raw: str) -> TradingDecision:
        """Parse the JSON response into a TradingDecision."""
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("response is not a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("ai_json_parse_failed", error=str(e), raw=raw[:500])
            decision = self._error_decision(f"JSON parse error: {e}", raw=raw)
            return decision

        actions = []
        for act in data.get("actions", []):
            try:
                actions.append(TradeAction(
                    pair=str(act.get("pair", "")),
                    side=str(act.get("side", "")),
                    size_pct=float(act.get("size_pct", 0)),
                    stop_loss_pct=float(act.get("stop_loss_pct", 3.0)),
                    take_profit_pct=float(act.get("take_profit_pct", 6.0)),
                    confidence=float(act.get("confidence", 0)),
                    reasoning=str(act.get("reasoning", "")),
                ))
            except (ValueError, TypeError) as e:
                logger.warning("ai_action_parse_skip", error=str(e), action=act)

        # Filter out low-confidence actions
        actions = [a for a in actions if a.confidence >= 60]

        return TradingDecision(
            actions=actions,
            market_outlook=str(data.get("market_outlook", "unknown")),
            risk_assessment=str(data.get("risk_assessment", "")),
            reasoning=str(data.get("reasoning", "")),
            raw_response=raw,
            model=self.model,
        )
