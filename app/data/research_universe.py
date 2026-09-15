from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.universe import UNIVERSE_COLUMNS, download_asx_isin_directory, equity_universe, save_universe

RESEARCH_UNIVERSE_COLUMNS = [
    "ticker",
    "company_name",
    "isin",
    "instrument_type",
    "source",
]


def build_current_research_universe() -> pd.DataFrame:
    """Build the broad current ASX ordinary-equity research candidate universe.

    This deliberately starts from the live ASX ISIN directory and applies the
    project's conservative equity classification. It is a *candidate* universe,
    not yet a point-in-time historical universe: listing/delisting dates and
    historical membership must be added before using it for final backtests.
    """
    directory = download_asx_isin_directory()
    candidates = equity_universe(directory)
    return candidates[RESEARCH_UNIVERSE_COLUMNS].drop_duplicates(
        subset=["ticker", "isin"], keep="first"
    ).sort_values(["ticker", "isin"]).reset_index(drop=True)


def save_current_research_universe(
    path: Path,
) -> pd.DataFrame:
    """Build and save the current broad research candidate universe."""
    universe = build_current_research_universe()
    save_universe(universe, path)
    return universe


def load_research_universe(path: Path) -> pd.DataFrame:
    """Load a previously generated research-universe CSV."""
    frame = pd.read_csv(path)
    missing = [column for column in RESEARCH_UNIVERSE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Research universe is missing columns: {missing}")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["company_name"] = frame["company_name"].astype(str).str.strip()
    frame["isin"] = frame["isin"].astype(str).str.strip().str.upper()
    return frame[RESEARCH_UNIVERSE_COLUMNS].drop_duplicates(
        subset=["ticker", "isin"], keep="first"
    ).sort_values(["ticker", "isin"]).reset_index(drop=True)
