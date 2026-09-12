import pandas as pd
import pytest

from app.data.universe import equity_universe, normalise_asx_isin_directory


def test_normalise_asx_isin_directory() -> None:
    raw = pd.DataFrame(
        {
            "ASX Code": ["BHP", "CBA", "BHP"],
            "Company Name": [
                "BHP GROUP LIMITED",
                "COMMONWEALTH BANK OF AUSTRALIA",
                "BHP GROUP LIMITED",
            ],
            "ISIN": ["AU000000BHP4", "AU000000CBA7", "AU000000BHP4"],
        }
    )

    result = normalise_asx_isin_directory(raw)

    assert list(result.columns) == [
        "ticker",
        "company_name",
        "isin",
        "instrument_type",
        "source",
    ]
    assert len(result) == 2
    assert result.loc[result["ticker"] == "BHP", "company_name"].iloc[0] == "BHP GROUP LIMITED"
    assert set(result["instrument_type"]) == {"equity_candidate"}


def test_normalise_asx_isin_directory_classifies_obvious_non_equity() -> None:
    raw = pd.DataFrame(
        {
            "ASX Code": ["BHP", "1GOV", "360JCA", "14DAJ"],
            "Company Name": [
                "BHP GROUP LIMITED",
                "VANECK 1-5 YEAR AUSTRALIAN GOVERNMENT BOND ETF",
                "CITIGROUP GLOBAL MARKETS AUS PTY LTD",
                "1414 DEGREES LIMITED",
            ],
            "ISIN": [
                "AU000000BHP4",
                "AU0000295180",
                "AU000360JCA8",
                "AU0000435141",
            ],
        }
    )

    result = normalise_asx_isin_directory(raw)
    types = dict(zip(result["ticker"], result["instrument_type"]))

    assert types["BHP"] == "equity_candidate"
    assert types["1GOV"] == "non_equity"
    assert types["360JCA"] == "non_equity"
    assert types["14DAJ"] == "other"

    candidates = equity_universe(result)
    assert candidates["ticker"].tolist() == ["BHP"]


def test_equity_universe_requires_expected_columns() -> None:
    with pytest.raises(ValueError, match="Universe is missing"):
        equity_universe(pd.DataFrame({"ticker": ["BHP"]}))


def test_empty_asx_directory() -> None:
    result = normalise_asx_isin_directory(pd.DataFrame())

    assert result.empty
    assert list(result.columns) == [
        "ticker",
        "company_name",
        "isin",
        "instrument_type",
        "source",
    ]
