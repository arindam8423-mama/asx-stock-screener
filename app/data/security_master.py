from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

SECURITY_COLUMNS = [
    "security_id",
    "ticker",
    "company_name",
    "isin",
    "instrument_type",
    "listing_date",
    "delisting_date",
    "status",
    "source",
]

ACTIVE_STATUSES = {"active", "listed"}


def _as_date_series(values: pd.Series) -> pd.Series:
    """Convert date-like values to normalized pandas timestamps, preserving nulls."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def build_security_master(universe: pd.DataFrame) -> pd.DataFrame:
    """Create the security-master shape from normalized ASX instrument metadata.

    The current ASX ISIN directory is a snapshot, so listing and delisting dates
    are intentionally left unknown. They must be populated from historical
    listing, delisting, and code-change sources before a point-in-time backtest.
    The ISIN is used as the initial stable security identifier because tickers
    can change over a security's lifetime.
    """
    required = ["ticker", "company_name", "isin", "instrument_type", "source"]
    missing = [column for column in required if column not in universe.columns]
    if missing:
        raise ValueError(f"Universe is missing columns: {missing}")

    frame = universe[required].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["company_name"] = frame["company_name"].astype(str).str.strip()
    frame["isin"] = frame["isin"].astype(str).str.strip().str.upper()
    frame["security_id"] = frame["isin"]
    frame["listing_date"] = pd.NaT
    frame["delisting_date"] = pd.NaT
    frame["status"] = "active"

    frame = frame[SECURITY_COLUMNS]
    frame = frame.drop_duplicates(subset=["security_id"], keep="first")
    return frame.sort_values("security_id").reset_index(drop=True)


def update_security_dates(
    master: pd.DataFrame,
    updates: pd.DataFrame,
) -> pd.DataFrame:
    """Apply externally sourced listing/delisting dates to a security master.

    ``updates`` must contain ``security_id`` and may contain ``listing_date``,
    ``delisting_date`` and ``status``. No dates are inferred from the current
    snapshot; missing values remain unknown.
    """
    if "security_id" not in updates.columns:
        raise ValueError("Security-date updates must contain security_id")

    result = master.copy()
    if "security_id" not in result.columns:
        raise ValueError("Security master must contain security_id")

    patch = updates.copy()
    for column in ("listing_date", "delisting_date"):
        if column in patch.columns:
            patch[column] = _as_date_series(patch[column])

    allowed = [
        column for column in ("listing_date", "delisting_date", "status")
        if column in patch.columns
    ]
    patch = patch[["security_id", *allowed]].drop_duplicates("security_id", keep="last")
    result = result.set_index("security_id")
    patch = patch.set_index("security_id")

    for column in allowed:
        result.loc[result.index.intersection(patch.index), column] = patch.loc[
            result.index.intersection(patch.index), column
        ]

    return result.reset_index()[SECURITY_COLUMNS].sort_values("security_id").reset_index(drop=True)


def point_in_time_universe(
    master: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
) -> pd.DataFrame:
    """Return securities known to be eligible on a historical date.

    A security is included only when its listing date is known and is on/before
    ``as_of``, and its delisting date is either unknown or after ``as_of``.
    This conservative rule prevents the current snapshot from silently becoming
    a survivorship-biased historical universe.
    """
    missing = [column for column in SECURITY_COLUMNS if column not in master.columns]
    if missing:
        raise ValueError(f"Security master is missing columns: {missing}")

    as_of_ts = pd.Timestamp(as_of).normalize()
    frame = master.copy()
    frame["listing_date"] = _as_date_series(frame["listing_date"])
    frame["delisting_date"] = _as_date_series(frame["delisting_date"])

    eligible = (
        frame["listing_date"].notna()
        & (frame["listing_date"] <= as_of_ts)
        & (frame["delisting_date"].isna() | (frame["delisting_date"] > as_of_ts))
    )
    return frame.loc[eligible, SECURITY_COLUMNS].reset_index(drop=True)


def save_security_master(master: pd.DataFrame, path: Path) -> Path:
    """Persist a normalized security master as CSV."""
    missing = [column for column in SECURITY_COLUMNS if column not in master.columns]
    if missing:
        raise ValueError(f"Security master is missing columns: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    master[SECURITY_COLUMNS].to_csv(path, index=False)
    return path
