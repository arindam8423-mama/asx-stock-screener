from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ASX_ISIN_DIRECTORY_URL = "https://www.asx.com.au/content/dam/asx/issuers/ISIN.xls"

UNIVERSE_COLUMNS = [
    "ticker",
    "company_name",
    "isin",
    "instrument_type",
    "source",
]


def _normalise_column_name(value: object) -> str:
    return "".join(str(value).strip().lower().replace("_", " ").split())


def _find_column(columns: list[object], candidates: tuple[str, ...]) -> object:
    normalised = {_normalise_column_name(column): column for column in columns}
    for candidate in candidates:
        key = _normalise_column_name(candidate)
        if key in normalised:
            return normalised[key]
    raise ValueError(f"Could not find any of {candidates} in ASX directory columns: {columns}")


def normalise_asx_isin_directory(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ASX ISIN directory into stable universe metadata.

    The ASX file is an instrument/issuer reference file rather than a pure
    ordinary-share universe. We therefore retain every identified instrument
    and leave instrument_type as unknown until a security-type source is
    added. This avoids silently introducing survivorship or product-selection
    assumptions into the research universe.
    """
    if raw.empty:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)

    columns = list(raw.columns)
    ticker_col = _find_column(columns, ("ASX code", "ASX Code", "code", "ticker"))
    isin_col = _find_column(columns, ("ISIN", "ISIN Code", "ISIN code"))
    name_col = _find_column(
        columns,
        ("Company Name", "Company name", "Issuer Name", "Issuer name", "Name"),
    )

    frame = raw[[ticker_col, name_col, isin_col]].copy()
    frame.columns = ["ticker", "company_name", "isin"]
    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".AX", "", regex=False)
    )
    frame["company_name"] = frame["company_name"].astype(str).str.strip()
    frame["isin"] = frame["isin"].astype(str).str.strip().str.upper()

    frame = frame.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    frame = frame.dropna(subset=["ticker", "isin"])
    frame["instrument_type"] = "unknown"
    frame["source"] = "ASX ISIN directory"
    frame = frame[UNIVERSE_COLUMNS]
    frame = frame.drop_duplicates(subset=["isin"], keep="first")
    return frame.sort_values(["ticker", "isin"]).reset_index(drop=True)


def download_asx_isin_directory(url: str = ASX_ISIN_DIRECTORY_URL) -> pd.DataFrame:
    """Download and normalise the current ASX ISIN directory."""
    request = Request(url, headers={"User-Agent": "asx-stock-screener/0.1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - URL is an ASX constant/default
        payload = response.read()
    raw = pd.read_excel(BytesIO(payload), engine="xlrd")
    return normalise_asx_isin_directory(raw)


def save_universe(universe: pd.DataFrame, path: Path) -> Path:
    """Persist normalised universe metadata as CSV."""
    missing = [column for column in UNIVERSE_COLUMNS if column not in universe.columns]
    if missing:
        raise ValueError(f"Universe is missing columns: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    universe[UNIVERSE_COLUMNS].to_csv(path, index=False)
    return path
