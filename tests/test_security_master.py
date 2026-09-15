import pandas as pd
import pytest

from app.data.security_master import (
    build_security_master,
    point_in_time_universe,
    save_security_master,
    update_security_dates,
)


def sample_universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "company_name": ["AAA LIMITED", "BBB LIMITED", "CCC LIMITED"],
            "isin": ["AU000000AAA1", "AU000000BBB2", "AU000000CCC3"],
            "instrument_type": ["equity_candidate"] * 3,
            "source": ["test"] * 3,
        }
    )


def test_build_security_master_uses_isin_as_stable_id() -> None:
    result = build_security_master(sample_universe())

    assert list(result.columns) == [
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
    assert result["security_id"].tolist() == [
        "AU000000AAA1",
        "AU000000BBB2",
        "AU000000CCC3",
    ]
    assert result["listing_date"].isna().all()
    assert result["delisting_date"].isna().all()
    assert set(result["status"]) == {"active"}


def test_update_security_dates_and_point_in_time_filter() -> None:
    master = build_security_master(sample_universe())
    updates = pd.DataFrame(
        {
            "security_id": ["AU000000AAA1", "AU000000BBB2", "AU000000CCC3"],
            "listing_date": ["2020-01-01", "2022-01-01", "2020-01-01"],
            "delisting_date": [None, "2023-06-30", "2021-12-31"],
            "status": ["active", "delisted", "delisted"],
        }
    )
    master = update_security_dates(master, updates)

    result = point_in_time_universe(master, "2022-06-01")
    assert result["ticker"].tolist() == ["AAA", "BBB"]

    result = point_in_time_universe(master, "2024-01-01")
    assert result["ticker"].tolist() == ["AAA"]


def test_point_in_time_universe_does_not_assume_unknown_listing_dates() -> None:
    master = build_security_master(sample_universe())
    result = point_in_time_universe(master, "2024-01-01")
    assert result.empty


def test_update_security_dates_requires_security_id() -> None:
    master = build_security_master(sample_universe())
    with pytest.raises(ValueError, match="security_id"):
        update_security_dates(master, pd.DataFrame({"listing_date": ["2020-01-01"]}))


def test_save_security_master(tmp_path) -> None:
    master = build_security_master(sample_universe())
    path = save_security_master(master, tmp_path / "security_master.csv")

    assert path.exists()
    saved = pd.read_csv(path)
    assert saved["security_id"].tolist() == [
        "AU000000AAA1",
        "AU000000BBB2",
        "AU000000CCC3",
    ]
