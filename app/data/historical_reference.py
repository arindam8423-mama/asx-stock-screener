from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ASX_CODE_CHANGES_URL = (
    "https://www.asx.com.au/markets/market-resources/asx-codes-and-descriptors/asx-code-changes"
)

CODE_CHANGE_COLUMNS = [
    "effective_date",
    "old_ticker",
    "old_name",
    "new_ticker",
    "new_name",
    "source",
]

DELISTING_COLUMNS = [
    "ticker",
    "company_name",
    "delisting_date",
    "reason",
    "source",
]


def _normalise_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split()).strip()


def _normalise_code(value: object) -> str:
    return _normalise_text(value).upper().replace(".AX", "")


def _parse_effective_dates(values: pd.Series) -> pd.Series:
    """Parse ASX date labels, including the page's day-month labels without a year."""
    text = values.map(_normalise_text)
    parsed = pd.to_datetime(text, errors="coerce", format="mixed")

    missing = parsed.isna() & text.ne("")
    if missing.any():
        current_year = datetime.now().year
        with_year = text.loc[missing] + f"-{current_year}"
        parsed.loc[missing] = pd.to_datetime(
            with_year,
            errors="coerce",
            format="%d-%b-%Y",
        )

    return parsed.dt.normalize()


def _extract_code_change_table(table: pd.DataFrame) -> pd.DataFrame:
    """Convert one ASX code-change HTML table to the stable schema."""
    if table.empty:
        return pd.DataFrame(columns=CODE_CHANGE_COLUMNS)

    frame = table.copy()
    frame.columns = [_normalise_text(column).lower() for column in frame.columns]

    # The ASX page presents old/new details as grouped table headings. Pandas
    # can flatten those headings differently across HTML revisions, so accept
    # both labelled and positional five-column tables.
    if len(frame.columns) == 5:
        columns = list(frame.columns)
        date_col = next((c for c in columns if "as of" in c), columns[0])
        old_cols = [c for c in columns if "old" in c]
        new_cols = [c for c in columns if "new" in c]
        if len(old_cols) >= 2 and len(new_cols) >= 2:
            result = pd.DataFrame(
                {
                    "effective_date": frame[date_col],
                    "old_ticker": frame[old_cols[0]],
                    "old_name": frame[old_cols[1]],
                    "new_ticker": frame[new_cols[0]],
                    "new_name": frame[new_cols[1]],
                }
            )
        else:
            result = frame.iloc[:, :5].copy()
            result.columns = [
                "effective_date",
                "old_ticker",
                "old_name",
                "new_ticker",
                "new_name",
            ]
    else:
        return pd.DataFrame(columns=CODE_CHANGE_COLUMNS)

    result["effective_date"] = _parse_effective_dates(result["effective_date"])
    for column in ("old_ticker", "new_ticker"):
        result[column] = result[column].map(_normalise_code)
    for column in ("old_name", "new_name"):
        result[column] = result[column].map(_normalise_text)

    result = result.dropna(subset=["effective_date"])
    result = result[(result["old_ticker"] != "") & (result["new_ticker"] != "")]
    result["source"] = "ASX code changes"
    return result[CODE_CHANGE_COLUMNS]


def parse_asx_code_changes_html(html: str) -> pd.DataFrame:
    """Parse ASX code/name-change tables from saved HTML content."""
    tables = pd.read_html(StringIO(html))
    parsed = [_extract_code_change_table(table) for table in tables]
    parsed = [table for table in parsed if not table.empty]
    if not parsed:
        return pd.DataFrame(columns=CODE_CHANGE_COLUMNS)
    return (
        pd.concat(parsed, ignore_index=True)
        .drop_duplicates()
        .sort_values(["effective_date", "old_ticker", "new_ticker"])
        .reset_index(drop=True)
    )


def download_asx_code_changes(url: str = ASX_CODE_CHANGES_URL) -> pd.DataFrame:
    """Download and parse ASX's published code/name-change history."""
    request = Request(url, headers={"User-Agent": "asx-stock-screener/0.1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - URL is an ASX constant/default
        html = response.read().decode("utf-8", errors="replace")
    return parse_asx_code_changes_html(html)


def normalise_delistings(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a user-supplied historical delisting CSV/DataFrame.

    The source is intentionally external to the repository. This prevents the
    project from silently treating a partial public snapshot as complete history.
    """
    if raw.empty:
        return pd.DataFrame(columns=DELISTING_COLUMNS)

    columns = {str(column).strip().lower(): column for column in raw.columns}

    def find(*names: str) -> object:
        for name in names:
            if name in columns:
                return columns[name]
        raise ValueError(f"Could not find any of {names} in delisting columns: {list(raw.columns)}")

    ticker_col = find("ticker", "asx code", "code")
    name_col = find("company_name", "company name", "company")
    date_col = find("delisting_date", "date delisted", "date delisted ", "date")
    reason_col = next(
        (columns[name] for name in ("reason", "official reason", "delisting reason") if name in columns),
        None,
    )

    frame = pd.DataFrame(
        {
            "ticker": raw[ticker_col].map(_normalise_code),
            "company_name": raw[name_col].map(_normalise_text),
            "delisting_date": pd.to_datetime(raw[date_col], errors="coerce").dt.normalize(),
            "reason": raw[reason_col].map(_normalise_text) if reason_col is not None else "",
            "source": "external delisting dataset",
        }
    )
    frame = frame[(frame["ticker"] != "") & frame["delisting_date"].notna()]
    return frame[DELISTING_COLUMNS].drop_duplicates().sort_values(
        ["delisting_date", "ticker"]
    ).reset_index(drop=True)


def load_delistings(path: Path) -> pd.DataFrame:
    """Load and normalise an externally obtained delisting CSV."""
    return normalise_delistings(pd.read_csv(path))


def save_code_changes(changes: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    changes[CODE_CHANGE_COLUMNS].to_csv(path, index=False)
    return path


def save_delistings(delisted: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    delisted[DELISTING_COLUMNS].to_csv(path, index=False)
    return path
