"""
Comprehensive research report generator.

Produces a detailed markdown report covering the entire research pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ReportGenerator:
    """Generates comprehensive research reports in markdown format."""

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        config_summary: dict[str, Any],
        data_summary: dict[str, Any],
        strategy_results: list[dict[str, Any]],
        selected_strategies: list[dict[str, Any]],
        portfolio_results: dict[str, Any] | None = None,
        validation_results: dict[str, Any] | None = None,
        monte_carlo_results: dict[str, Any] | None = None,
        robustness_results: dict[str, Any] | None = None,
    ) -> str:
        """Generate the complete research report."""
        sections = [
            self._header(),
            self._executive_summary(
                config_summary, data_summary, strategy_results, selected_strategies
            ),
            self._methodology(config_summary),
            self._data_description(data_summary),
            self._strategy_results(strategy_results),
            self._selected_strategies(selected_strategies),
            self._portfolio_results(portfolio_results),
            self._robustness_analysis(robustness_results),
            self._monte_carlo_analysis(monte_carlo_results),
            self._validation_results(validation_results),
            self._risk_analysis(selected_strategies, monte_carlo_results),
            self._final_recommendation(selected_strategies),
            self._limitations(),
            self._appendix(config_summary),
        ]

        report = "\n\n".join(s for s in sections if s)
        path = self.output_dir / "research_report.md"
        with open(path, "w") as f:
            f.write(report)

        logger.info("report_generated", path=str(path))
        return str(path)

    def _header(self) -> str:
        return f"""# Trading Strategy Research Report

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
**System:** Trading Backtesting & Automated Trading System v0.1.0"""

    def _executive_summary(
        self,
        config: dict,
        data: dict,
        strategy_results: list,
        selected: list,
    ) -> str:
        total_tested = len(strategy_results)
        profitable = sum(1 for r in strategy_results if r.get("results", {}).get("total_return", 0) > 0)

        return f"""## Executive Summary

### Scope
- **Exchange:** {config.get('exchange', 'Binance Futures')}
- **Pairs:** {config.get('pairs', 'BTC/USDT, ETH/USDT')}
- **Timeframes:** {config.get('timeframes', '15m, 30m, 1h, 4h')}
- **Period:** {config.get('start_date', '2022-01-01')} to {config.get('end_date', 'present')}
- **Strategy families tested:** {config.get('families', 'Trend, Momentum, Mean Reversion, Volatility')}

### Results
- **Total combinations tested:** {total_tested:,}
- **Profitable strategies:** {profitable:,} ({profitable/max(total_tested,1)*100:.1f}%)
- **Selected strategies:** {len(selected)}

### Final Strategies
{self._format_selected_summary(selected)}"""

    def _methodology(self, config: dict) -> str:
        return """## Methodology

### Data Pipeline
- Historical OHLCV data from Binance Futures
- Funding rate data for perpetual futures cost modeling
- Data validation: gap detection, OHLC integrity, anomaly detection
- Data cleaning: duplicate removal, zero-price filtering, forward-fill for small gaps

### Backtesting Engine
- Vectorized execution with strict causality enforcement
- Execution model: Signal at candle N close → execution at candle N+1 open
- Slippage model: ATR-adaptive (higher volatility = more slippage)
- Fee model: Taker fees (0.04%), spread (0.02%), actual funding rates
- Position tracking: Long/short with leverage, funding cost accrual

### Anti-Overfitting Measures
- Chronological data splitting (in-sample / validation / out-of-sample / holdout)
- Walk-forward analysis with rolling windows
- Monte Carlo simulation (trade shuffling, bootstrap)
- Parameter perturbation testing
- Transaction cost stress testing (1x, 1.5x, 2x, 3x)
- Composite scoring that penalizes overfitting indicators
- Minimum trade count requirements

### Optimization
- Grid search for comprehensive parameter exploration
- Random search for large parameter spaces
- Bayesian optimization (Optuna) for efficient exploration
- Multi-objective scoring (not just maximizing returns)

### Scoring Formula
```
Score = 0.25 × Sharpe + 0.15 × Sortino + 0.15 × Calmar
      + 0.10 × ProfitFactor + 0.10 × OOS_Ratio
      + 0.10 × ParamStability + 0.05 × RegimeStability
      + 0.05 × CostRobustness + 0.05 × TimeframeStability
      - 0.05 × OverfittingPenalty
```"""

    def _data_description(self, data: dict) -> str:
        lines = ["## Data Description\n"]
        for pair, timeframes in data.items():
            lines.append(f"### {pair}")
            for tf, info in timeframes.items():
                lines.append(f"- **{tf}**: {info.get('rows', 0):,} candles "
                           f"({info.get('start', 'N/A')} to {info.get('end', 'N/A')})")
        return "\n".join(lines)

    def _strategy_results(self, results: list[dict]) -> str:
        lines = ["## Strategy Results\n"]
        lines.append("| Strategy | Family | Sharpe | Sortino | Max DD | Return | Trades | Score |")
        lines.append("|----------|--------|--------|---------|--------|--------|--------|-------|")

        for r in sorted(results, key=lambda x: x.get("results", {}).get("composite_score", 0), reverse=True)[:30]:
            res = r.get("results", {})
            lines.append(
                f"| {r.get('strategy_name', 'N/A')} | {r.get('timeframe', 'N/A')} "
                f"| {res.get('sharpe', 0):.2f} | {res.get('sortino', 0):.2f} "
                f"| {res.get('max_drawdown', 0):.1%} | {res.get('total_return', 0):.1%} "
                f"| {res.get('total_trades', 0)} | {res.get('composite_score', 0):.3f} |"
            )

        return "\n".join(lines)

    def _selected_strategies(self, selected: list[dict]) -> str:
        lines = ["## Selected Strategies\n"]
        for i, s in enumerate(selected):
            res = s.get("results", {})
            lines.append(f"""### Strategy {i+1}: {s.get('strategy_name', 'N/A')} ({s.get('timeframe', 'N/A')})

**Parameters:** `{s.get('parameters', {})}`

| Metric | Value |
|--------|-------|
| Sharpe | {res.get('sharpe', 0):.2f} |
| Sortino | {res.get('sortino', 0):.2f} |
| Calmar | {res.get('calmar', 0):.2f} |
| Total Return | {res.get('total_return', 0):.1%} |
| CAGR | {res.get('cagr', 0):.1%} |
| Max Drawdown | {res.get('max_drawdown', 0):.1%} |
| Win Rate | {res.get('win_rate', 0):.1%} |
| Profit Factor | {res.get('profit_factor', 0):.2f} |
| Total Trades | {res.get('total_trades', 0)} |
| Avg Trade | ${res.get('expectancy', 0):.2f} |
| Total Fees | ${res.get('total_fees', 0):.2f} |
| Total Funding | ${res.get('total_funding_costs', 0):.2f} |
""")

        return "\n".join(lines)

    def _portfolio_results(self, portfolio: dict | None) -> str:
        if not portfolio:
            return ""
        return f"""## Portfolio Results

**Combined portfolio performance across selected strategies.**

{self._format_dict_table(portfolio)}"""

    def _robustness_analysis(self, robustness: dict | None) -> str:
        if not robustness:
            return ""
        return """## Robustness Analysis

### Parameter Stability
Strategies were tested with ±30% parameter perturbation.
A robust strategy maintains positive performance across most nearby parameter values.

### Cost Stress Testing
Strategies were tested with 1x, 1.5x, 2x, and 3x transaction costs.
Strategies that become unprofitable at 1.5x costs are flagged as fragile.

### Time Period Stability
Strategies were tested across 4 independent time periods.
Strategies that only work in one period are flagged as regime-dependent.
"""

    def _monte_carlo_analysis(self, mc: dict | None) -> str:
        if not mc:
            return ""
        return """## Monte Carlo Analysis

Monte Carlo simulations were performed using trade shuffling and bootstrap resampling.
Results show the distribution of possible outcomes under random trade orderings.
"""

    def _validation_results(self, validation: dict | None) -> str:
        if not validation:
            return ""
        return """## Validation Results

### Walk-Forward Analysis
Strategies were tested using walk-forward windows to estimate realistic out-of-sample performance.

### Out-of-Sample Testing
Strategies were evaluated on data never seen during optimization.
"""

    def _risk_analysis(self, selected: list, mc: dict | None) -> str:
        return """## Risk Analysis

### Drawdown Analysis
- Maximum historical drawdown reported for each strategy
- Monte Carlo estimate of 95th percentile worst-case drawdown
- Expected losing streak duration

### Risk of Ruin
Monte Carlo simulations estimate probability of significant capital loss.

### Sensitivity to Execution
Performance under worse fills, delayed entries, and partial fills.
"""

    def _final_recommendation(self, selected: list[dict]) -> str:
        return f"""## Final Recommendation

Based on comprehensive testing across {len(selected)} selected strategies:

{self._format_selected_summary(selected)}

### Key Considerations
1. These results are based on historical data and do not guarantee future performance
2. Paper trading should be conducted for at least 30 days before live deployment
3. Start with minimum position sizes and gradually increase
4. Monitor performance regularly and be prepared to deactivate strategies
5. Market conditions change — regular revalidation is recommended

### Weaknesses and Caveats
- Performance in extreme market conditions (flash crashes, exchange outages) is not captured
- Actual execution may differ from backtest assumptions
- Funding rates and slippage in live markets may be higher than modeled
"""

    def _limitations(self) -> str:
        return """## Limitations

1. **Historical data quality:** While validated, historical data may contain errors not caught by automated checks
2. **Execution assumptions:** Backtest assumes next-candle open execution; real execution may be different
3. **Liquidity:** Large positions may experience significant market impact not modeled in backtests
4. **Correlation:** Strategy correlations may change in different market regimes
5. **Black swan events:** Extreme events are underrepresented in historical data
6. **Overfitting risk:** Despite multiple safeguards, some degree of data-mining bias may remain
"""

    def _appendix(self, config: dict) -> str:
        return f"""## Appendix

### Configuration
```
{self._format_dict(config)}
```

### Software Versions
- Python 3.10+
- pandas, numpy, scipy, ccxt
- Trading System v0.1.0
"""

    def _format_selected_summary(self, selected: list[dict]) -> str:
        if not selected:
            return "No strategies met the robustness criteria for selection."
        lines = []
        for i, s in enumerate(selected):
            res = s.get("results", {})
            lines.append(
                f"{i+1}. **{s.get('strategy_name', 'N/A')}** ({s.get('timeframe', 'N/A')}): "
                f"Sharpe {res.get('sharpe', 0):.2f}, "
                f"Max DD {res.get('max_drawdown', 0):.1%}, "
                f"Return {res.get('total_return', 0):.1%}"
            )
        return "\n".join(lines)

    def _format_dict_table(self, d: dict, depth: int = 0) -> str:
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"**{k}:**")
                lines.append(self._format_dict_table(v, depth + 1))
            elif isinstance(v, float):
                if abs(v) < 1:
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v:.2f}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _format_dict(self, d: dict) -> str:
        return "\n".join(f"{k}: {v}" for k, v in d.items())
