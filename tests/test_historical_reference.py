import pandas as pd

from app.data.historical_reference import normalise_delistings, parse_asx_code_changes_html


def test_parse_asx_code_changes_html() -> None:
    html = """
    <table>
      <thead>
        <tr><th>As of</th><th>Old ticker</th><th>Old name</th><th>New ticker</th><th>New name</th></tr>
      </thead>
      <tbody>
        <tr><td>2-Sep</td><td>C1X</td><td>Cosmos Exploration Limited</td><td>EAU</td><td>Eau Lithium Ltd</td></tr>
        <tr><td>1-Sep</td><td>WEB</td><td>Web Travel Group Limited</td><td>WEB</td><td>WebBeds Group Limited</td></tr>
      </tbody>
    </table>
    """

    result = parse_asx_code_changes_html(html)

    assert len(result) == 2
    assert result.loc[0, "old_ticker"] == "C1X"
    assert result.loc[0, "new_ticker"] == "EAU"
    assert result.loc[1, "new_name"] == "WebBeds Group Limited"
    assert pd.notna(result.loc[0, "effective_date"])


def test_parse_empty_code_changes() -> None:
    result = parse_asx_code_changes_html("<html><body>No tables</body></html>")
    assert result.empty


def test_normalise_delistings() -> None:
    raw = pd.DataFrame(
        {
            "ASX Code": ["ABC", "XYZ", "ABC"],
            "Company Name": ["Alpha Limited", "Xray Limited", "Alpha Limited"],
            "Date Delisted": ["2024-01-10", "2025-02-20", "2024-01-10"],
            "Reason": ["Takeover", "Failure", "Takeover"],
        }
    )

    result = normalise_delistings(raw)

    assert len(result) == 2
    assert result.loc[0, "ticker"] == "ABC"
    assert result.loc[0, "delisting_date"] == pd.Timestamp("2024-01-10")
    assert result.loc[1, "reason"] == "Failure"


def test_normalise_delistings_requires_date() -> None:
    raw = pd.DataFrame({"ASX Code": ["ABC"], "Company Name": ["Alpha Limited"]})

    try:
        normalise_delistings(raw)
    except ValueError as exc:
        assert "delisting" in str(exc).lower()
    else:
        raise AssertionError("Expected missing delisting date to raise ValueError")
