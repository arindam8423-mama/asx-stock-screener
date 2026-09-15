import pandas as pd

from app.data.research_universe import load_research_universe


def test_load_research_universe_normalises_and_deduplicates(tmp_path) -> None:
    path = tmp_path / "research_universe.csv"
    pd.DataFrame(
        {
            "ticker": [" cba ", "CBA", "BHP"],
            "company_name": [" Commonwealth Bank ", "Commonwealth Bank", "BHP Group"],
            "isin": ["AU000000CBA7", "AU000000CBA7", "AU000000BHP4"],
            "instrument_type": ["equity_candidate"] * 3,
            "source": ["ASX ISIN directory"] * 3,
        }
    ).to_csv(path, index=False)

    result = load_research_universe(path)

    assert result["ticker"].tolist() == ["BHP", "CBA"]
    assert result["isin"].tolist() == ["AU000000BHP4", "AU000000CBA7"]


def test_load_research_universe_requires_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"ticker": ["CBA"]}).to_csv(path, index=False)

    try:
        load_research_universe(path)
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Expected missing-column validation error")
