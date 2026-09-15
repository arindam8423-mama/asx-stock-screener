import pandas as pd
import pytest

from app.data.historical_universe import (
    build_historical_security_master,
    normalise_historical_metadata,
    normalise_security_aliases,
    point_in_time_research_universe,
    ticker_as_of,
)


def research_universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "company_name": ["AAA LIMITED", "BBB LIMITED", "CCC LIMITED"],
            "isin": ["AU000000AAA1", "AU000000BBB2", "AU000000CCC3"],
            "instrument_type": ["equity_candidate"] * 3,
            "source": ["ASX ISIN directory"] * 3,
        }
    )


def test_listing_and_delisting_dates_control_point_in_time_eligibility() -> None:
    metadata = pd.DataFrame(
        {
            "security_id": ["AU000000AAA1", "AU000000BBB2", "AU000000CCC3"],
            "listing_date": ["2020-01-01", "2024-01-01", "2020-01-01"],
            "delisting_date": [None, "2024-06-30", "2023-12-31"],
            "status": ["active", "delisted", "delisted"],
            "source": ["test"] * 3,
        }
    )
    master = build_historical_security_master(research_universe(), metadata)

    assert set(point_in_time_research_universe(master, "2023-06-01")["ticker"]) == {"AAA", "CCC"}
    assert set(point_in_time_research_universe(master, "2024-03-01")["ticker"]) == {"AAA", "BBB"}
    assert set(point_in_time_research_universe(master, "2025-01-01")["ticker"]) == {"AAA"}


def test_unknown_listing_date_is_conservatively_excluded() -> None:
    metadata = pd.DataFrame(
        {
            "security_id": ["AU000000AAA1"],
            "listing_date": [None],
            "delisting_date": [None],
        }
    )
    master = build_historical_security_master(research_universe(), metadata)

    assert point_in_time_research_universe(master, "2025-01-01").empty


def test_historical_metadata_requires_stable_security_id() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        normalise_historical_metadata(pd.DataFrame({"ticker": ["AAA"], "listing_date": ["2020-01-01"]}))


def test_duplicate_metadata_uses_last_explicit_update() -> None:
    raw = pd.DataFrame(
        {
            "security_id": ["AU000000AAA1", "AU000000AAA1"],
            "listing_date": ["2020-01-01", "2020-02-01"],
            "delisting_date": [None, None],
        }
    )
    result = normalise_historical_metadata(raw)

    assert len(result) == 1
    assert result.iloc[0]["listing_date"] == pd.Timestamp("2020-02-01")


def test_ticker_change_is_explicitly_mapped_to_same_security() -> None:
    aliases = pd.DataFrame(
        {
            "security_id": ["AU000000AAA1", "AU000000AAA1"],
            "ticker": ["OLD", "NEW"],
            "effective_from": ["2019-01-01", "2022-05-01"],
            "effective_to": ["2022-05-01", None],
            "source": ["ASX code changes"] * 2,
        }
    )

    result = normalise_security_aliases(aliases)
    assert ticker_as_of(result, "AU000000AAA1", "2021-01-01") == "OLD"
    assert ticker_as_of(result, "AU000000AAA1", "2023-01-01") == "NEW"
