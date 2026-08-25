# Trading Strategy Backtesting & Automated Trading System

A professional-grade quantitative trading research and backtesting system for crypto perpetual futures on Binance.

## Overview

This system implements a complete research pipeline:

1. **Data Pipeline** — Download, validate, and clean historical market data
2. **Strategy Framework** — 17+ strategies across 4 families (trend, momentum, mean reversion, volatility)
3. **Backtesting Engine** — Vectorized engine with realistic execution, fees, slippage, and funding costs
4. **Optimization** — Grid search, random search, and Bayesian optimization across millions of parameter combinations
5. **Validation** — Walk-forward analysis, Monte Carlo simulation, robustness testing, overfitting detection
6. **Strategy Selection** — Composite scoring that prioritizes robustness over historical returns
7. **Portfolio Construction** — Multi-strategy portfolio with various allocation methods
8. **Trading Bot** — Paper trading and live trading modes with strict risk management
9. **Research Report** — Comprehensive markdown report of all findings

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### 1. Download Historical Data

```bash
python scripts/download_data.py --start 2022-01-01
```

### 2. Run a Single Backtest

```bash
python scripts/run_backtest.py --strategy MA_Crossover --pair "BTC/USDT:USDT" --timeframe 1h
```

### 3. Run Optimization

```bash
python scripts/run_optimization.py --strategy MACD --pair "BTC/USDT:USDT" --timeframe 1h
```

### 4. Run Full Research Pipeline

```bash
python scripts/run_full_research.py --pair "BTC/USDT:USDT"
```

### 5. Run Individual Validation Tests

```bash
python scripts/run_walk_forward.py --strategy MA_Crossover --pair "BTC/USDT:USDT"
python scripts/run_monte_carlo.py --strategy MA_Crossover --pair "BTC/USDT:USDT"
python scripts/run_robustness.py --strategy MA_Crossover --pair "BTC/USDT:USDT"
```

### 6. Generate Charts & Report

```bash
python scripts/generate_charts.py --strategy MA_Crossover
python scripts/generate_report.py
```

### 7. Start Paper Trading

```bash
python scripts/start_paper_bot.py --strategies "MA_Crossover,MACD"
```

### 8. Start Live Trading (requires API keys and --confirm)

```bash
python scripts/start_live_bot.py --strategies "MA_Crossover" --mode live --confirm
```

## Running Tests

```bash
make test
make test-cov
```

## Project Structure

```
trading_system/
├── data/           # Data download, validation, cleaning, storage
├── indicators/     # 30+ technical indicators
├── strategies/     # 17 strategies in 4 families
├── backtester/     # Vectorized backtesting engine
├── optimization/   # Parameter search and scoring
├── validation/     # Walk-forward, Monte Carlo, robustness
├── ranking/        # Strategy scoring and selection
├── portfolio/      # Multi-strategy allocation
├── bot/            # Paper and live trading bot
├── dashboard/      # Chart generation
├── reports/        # Research report generation
└── utils/          # Logging, timing, hashing
```

## Configuration

All settings are in `configs/default.yaml`. Key parameters:

- **Exchange**: Binance Futures (USDⓈ-M)
- **Pairs**: BTC/USDT:USDT, ETH/USDT:USDT
- **Timeframes**: 15m, 30m, 1h, 4h
- **Fees**: Taker 0.04%, Maker 0.02%
- **Slippage**: ATR-adaptive
- **Execution**: Next candle open
- **Leverage**: 1x (conservative default)

## Strategy Families

| Family | Strategies |
|--------|-----------|
| **Trend Following** | MA Crossover, MACD, ADX Trend, Donchian Breakout, Supertrend |
| **Momentum** | RSI Momentum, ROC, Stochastic Momentum, Multi-Momentum |
| **Mean Reversion** | RSI Reversion, Bollinger Reversion, Z-Score, BB Squeeze |
| **Volatility** | ATR Breakout, Volatility Expansion, Keltner Breakout, Regime Volatility |

## Anti-Overfitting Measures

- Chronological data splitting (train / validation / out-of-sample / holdout)
- Walk-forward analysis with rolling windows
- Monte Carlo simulation (trade shuffling, bootstrap)
- Parameter perturbation testing
- Transaction cost stress testing (1x to 3x)
- Composite scoring that penalizes overfitting indicators
- Minimum trade count requirements

## Scoring Formula

```
Score = 0.25 × Sharpe + 0.15 × Sortino + 0.15 × Calmar
      + 0.10 × ProfitFactor + 0.10 × OOS_Ratio
      + 0.10 × ParamStability + 0.05 × RegimeStability
      + 0.05 × CostRobustness + 0.05 × TimeframeStability
      - 0.05 × OverfittingPenalty
```

## Available Strategies (List)

```bash
python scripts/run_backtest.py --list-strategies
```

## Environment Variables

For live trading, set these in a `.env` file:

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

**Never commit API keys to source control.**

## Limitations

- Historical data may contain errors not caught by automated validation
- Backtest assumes next-candle open execution; real execution may differ
- Large positions may experience significant market impact
- Extreme market conditions (flash crashes, exchange outages) are not fully modeled
- Past performance does not guarantee future results
