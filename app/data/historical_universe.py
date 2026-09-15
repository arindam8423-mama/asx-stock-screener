from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from app.data.historical_reference import CODE_CHANGE_COLUMNS, DELISTING_COLUMNS
from app.data.security_master import SECURITY_COLUMNS, build_security_master, point_in_time_universe, update_security_dates

HISTORICAL_METADATA_COLUMNS = [
    "security_id",
    "listing_date",
    "delisting_date",
    "status",
    "source",
]

SECURITY_ALIAS_COLUMNS = [
    "security_id",
    "ticker",
    "effective_from",
    "effective_to",
    "source",
]


def _date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def normalise_historical_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize externally sourced point-in-time listing/delisting metadata.

    The metadata must identify a security by the project's stable ``security_id``
    (currently the ISIN). Dates are never inferred from today's ticker snapshot.
    """
    missing = [column for column in ("security_id", "listing_date", "delisting_date") if column not in raw.columns]
    if missing:
        raise ValueError(f"Historical metadata is missing columns: {missing}")

    frame = raw.copy()
    frame["security_id"] = frame["security_id"].astype(str).str.strip().str.upper()
    frame["listing_date"] = _date_series(frame["listing_date"])
    frame["delisting_date"] = _date_series(frame["delisting_date"])
    frame["status"] = frame["status"].astype(str).str.strip().str.lower() if "status" in frame else "active"
    frame["source"] = frame["source"].astype(str).str.strip() if "source" in frame else "external historical metadata"
    frame = frame[frame["security_id"].ne("")]
    return frame[HISTORICAL_METADATA_COLUMNS].drop_duplicates("security_id", keep="last").reset_index(drop=True)


def build_historical_security_master(
    research_universe: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build a security master using explicit historical metadata.

    Only securities present in the current research candidate universe are
    retained. Historical dates are applied by stable security_id/ISIN; ticker
    changes are handled separately through aliases rather than overwriting the
    current ticker with an inferred historical value.
    """
    master = build_security_master(research_universe)
    metadata = normalise_historical_metadata(metadata)
    return update_security_dates(master, metadata)


def point_in_time_research_universe(
    master: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
) -> pd.DataFrame:
    """Return the survivorship-safe research universe for one historical date."""
    return point_in_time_universe(master, as_of)


def normalise_security_aliases(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize explicit ticker aliases for securities whose codes changed.

    A code-change source alone does not establish which ISIN/security_id changed
    ticker. Therefore this function requires an explicit security_id mapping.
    """
    required = ["security_id", "ticker", "effective_from"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Security aliases are missing columns: {missing}")

    frame = raw.copy()
    frame["security_id"] = frame["security_id"].astype(str).str.strip().str.upper()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["effective_from"] = _date_series(frame["effective_from"])
    frame["effective_to"] = _date_series(frame["effective_to"]) if "effective_to" in frame else pd.NaT
    frame["source"] = frame["source"].astype(str).str.strip() if "source" in frame else "external ticker-history mapping"
    frame = frame[(frame["security_id"] != "") & (frame["ticker"] != "") & frame["effective_from"].notna()]
    return frame[SECURITY_ALIAS_COLUMNS].drop_duplicates().sort_values(
        ["security_id", "effective_from", "ticker"]
    ).reset_index(drop=True)


def ticker_as_of(
    aliases: pd.DataFrame,
    security_id: str,
    as_of: date | str | pd.Timestamp,
) -> str | None:
    """Resolve the explicit historical ticker for one security/date."""
    frame = normalise_security_aliases(aliases)
    as_of_ts = pd.Timestamp(as_of).normalize()
    matches = frame[
        (frame["security_id"] == str(security_id).strip().upper())
        & (frame["effective_from"] <= as_of_ts)
        & (frame["effective_to"].isna() | (frame["effective_to"] > as_of_ts))
    ]
    if matches.empty:
        return None
    return str(matches.sort_values("effective_from").iloc[-1]["ticker"])


def save_historical_security_master(master: pd.DataFrame, path: Path) -> Path:
    """Persist the historical security master as CSV."""
    missing = [column for column in SECURITY_COLUMNS if column not in master.columns]
    if missing:
        raise ValueError(f"Security master is missing columns: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    master[SECURITY_COLUMNS].to_csv(path, index=False)
    return path


def save_security_aliases(aliases: pd.DataFrame, path: Path) -> Path:
    """Persist explicit historical ticker aliases as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalise_security_aliases(aliases).to_csv(path, index=False)
    return path


__all__ = [
    "CODE_CHANGE_COLUMNS",
    "DELISTING_COLUMNS",
    "HISTORICAL_METADATA_COLUMNS",
    "SECURITY_ALIAS_COLUMNS",
    "build_historical_security_master",
    "normalise_historical_metadata",
    "normalise_security_aliases",
    "point_in_time_research_universe",
    "save_historical_security_master",
    "save_security_aliases",
    "ticker_as_of",
]
