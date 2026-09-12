# ASX Stock Screener

A research-first ASX stock screener and backtesting platform for short-term trades with a maximum holding period of 20 trading days (approximately one month).

## Goals

- Scan ASX-listed equities for systematic trading opportunities.
- Backtest strategies over a five-year historical period.
- Evaluate large-, mid-, small- and micro-cap stocks separately.
- Include realistic transaction costs, including configurable $11 brokerage per side.
- Model slippage and liquidity rather than assuming perfect fills.
- Avoid survivorship and look-ahead bias wherever the available data permits.
- Use walk-forward / out-of-sample validation before a strategy is considered viable.

## Current milestone

**Phase 1: Data foundation**

The first implementation establishes a local research data layer using Parquet files and DuckDB, plus a Yahoo Finance adapter for downloading adjusted historical OHLCV data for a supplied ASX ticker universe.

This is intentionally separate from strategy development. We will validate data quality and the backtesting engine before optimising trading rules.

## Local setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy the example environment file if required:

```bash
cp .env.example .env
```

Run tests:

```bash
pytest
```

## Download sample ASX data

Create a CSV containing a `ticker` column, for example:

```csv
ticker
CBA
BHP
CSL
WBC
```

Then run:

```bash
python -m app.cli download-prices --universe data/universe_sample.csv --start 2021-01-01 --end 2026-01-01
```

The downloader converts ASX tickers to Yahoo Finance's `.AX` format and stores clean per-ticker Parquet files under `data/prices/`.

## Important research rules

The production backtest must eventually account for:

1. Corporate actions and adjusted prices.
2. Securities that delisted during the test period.
3. Point-in-time market-cap classifications.
4. Suspensions and missing trading days.
5. Brokerage and slippage.
6. Trading only on information available at the time of the signal.
7. A maximum holding period of 20 trading sessions.

No strategy is considered profitable merely because it performs well in-sample.
