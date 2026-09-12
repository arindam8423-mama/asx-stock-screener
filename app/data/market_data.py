from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def yahoo_symbol(asx_ticker: str) -> str:
    """Convert an ASX ticker such as BHP to Yahoo Finance's BHP.AX symbol."""
    ticker = asx_ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    return ticker if ticker.endswith(".AX") else f"{ticker}.AX"


def normalise_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise a yfinance OHLCV frame into the project's stable schema."""
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # yfinance can return a MultiIndex even for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    missing = [column for column in rename if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns for {ticker}: {missing}")

    frame = raw.rename(columns=rename)[list(rename.values())].copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None)
    frame = frame.rename_axis("date").reset_index()
    frame["ticker"] = ticker.upper().removesuffix(".AX")

    frame = frame[REQUIRED_COLUMNS]
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0).astype("int64")
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return frame.reset_index(drop=True)


def download_history(
    ticker: str,
    start: str,
    end: str,
    output_dir: Path,
) -> Path:
    """Download one ASX ticker and persist it as Parquet."""
    clean_ticker = ticker.strip().upper().removesuffix(".AX")
    symbol = yahoo_symbol(clean_ticker)

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    frame = normalise_ohlcv(raw, clean_ticker)
    if frame.empty:
        raise ValueError(f"No historical data returned for {clean_ticker} ({symbol})")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{clean_ticker}.parquet"
    frame.to_parquet(path, index=False)
    return path


def load_universe(path: Path) -> list[str]:
    """Load tickers from a CSV containing a `ticker` column."""
    universe = pd.read_csv(path)
    if "ticker" not in universe.columns:
        raise ValueError("Universe CSV must contain a 'ticker' column")

    tickers = (
        universe["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.removesuffix(".AX")
    )
    return sorted(set(ticker for ticker in tickers if ticker))
