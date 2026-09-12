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

# The ASX ISIN directory contains many security types, not just ordinary
# shares. These name markers are deliberately conservative: they remove
# obvious funds/debt/derivative products while leaving company shares and
# listed equity vehicles for later research decisions.
NON_EQUITY_NAME_MARKERS = (
    " ETF",
    "FUND",
    "INDEX",
    "BOND",
    "WARRANT",
    "OPTION",
    "RIGHTS",
    "DEBENTURE",
    "NOTE",
    "PREFERENCE",
    "PREF ",
    "INSTALMENT",
    "CERTIFICATE",
    "MANAGED INVESTMENT",
    "SECURIT",
)

NON_EQUITY_ISSUER_MARKERS = (
    "CITIGROUP GLOBAL MARKETS",
    "MACQUARIE CAPITAL",
    "MORGAN STANLEY",
    "JPMORGAN",
    "UBS AG",
    "GOLDMAN SACHS",
)


def _normalise_column_name(value: object) -> str:
    return "".join(str(value).strip().lower().replace("_", " ").split())


def _find_column(columns: list[object], candidates: tuple[str, ...]) -> object:
    normalised = {_normalise_column_name(column): column for column in columns}
    for candidate in candidates:
        key = _normalise_column_name(candidate)
        if key in normalised:
            return normalised[key]
    raise ValueError(f"Could not find any of {candidates} in ASX directory columns: {columns}")


def _looks_like_non_equity(company_name: str) -> bool:
    name = f" {company_name.strip().upper()} "
    if any(marker in name for marker in NON_EQUITY_NAME_MARKERS):
        return True
    return any(marker in name for marker in NON_EQUITY_ISSUER_MARKERS)


def classify_instrument(ticker: str, company_name: str) -> str:
    """Classify the current universe using conservative reference-file heuristics.

    The public ASX ISIN directory does not provide a reliable security-type
    field, so this is intentionally a screening classification rather than a
    definitive ASX instrument taxonomy. Detailed security metadata belongs in
    the later corporate-actions/reference-data phase.
    """
    if _looks_like_non_equity(company_name):
        return "non_equity"
    if len(ticker) == 3:
        return "equity_candidate"
    return "other"


def normalise_asx_isin_directory(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ASX ISIN directory into stable instrument metadata."""
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
    frame["instrument_type"] = [
        classify_instrument(ticker, company_name)
        for ticker, company_name in zip(frame["ticker"], frame["company_name"])
    ]
    frame["source"] = "ASX ISIN directory"
    frame = frame[UNIVERSE_COLUMNS]
    frame = frame.drop_duplicates(subset=["isin"], keep="first")
    return frame.sort_values(["ticker", "isin"]).reset_index(drop=True)


def equity_universe(universe: pd.DataFrame) -> pd.DataFrame:
    """Return the conservative stock-screener candidate universe.

    This is deliberately separate from the raw ASX instrument universe so we
    retain all source instruments for auditability. It is not a historical,
    survivorship-safe universe; that requires security active dates and
    delisted/code-change data added in a later phase.
    """
    missing = [column for column in UNIVERSE_COLUMNS if column not in universe.columns]
    if missing:
        raise ValueError(f"Universe is missing columns: {missing}")
    return universe.loc[universe["instrument_type"] == "equity_candidate", UNIVERSE_COLUMNS].copy()


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
