from pathlib import Path

import pandas as pd
import pytest

from app.data.market_data import load_universe, normalise_ohlcv, yahoo_symbol


def test_yahoo_symbol_adds_asx_suffix() -> None:
    assert yahoo_symbol("BHP") == "BHP.AX"
    assert yahoo_symbol("bhp") == "BHP.AX"
    assert yahoo_symbol("BHP.AX") == "BHP.AX"


def test_yahoo_symbol_rejects_empty_ticker() -> None:
    with pytest.raises(ValueError):
        yahoo_symbol("  ")


def test_load_universe_deduplicates_and_normalises(tmp_path: Path) -> None:
    path = tmp_path / "universe.csv"
    path.write_text("ticker\nbhp\nCBA.AX\nBHP\n\n")

    assert load_universe(path) == ["BHP", "CBA"]


def test_normalise_ohlcv() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.1],
            "High": [1.2, 1.3],
            "Low": [0.9, 1.0],
            "Close": [1.1, 1.2],
            "Adj Close": [1.1, 1.2],
            "Volume": [100, 200],
        },
        index=index,
    )

    result = normalise_ohlcv(raw, "XYZ")

    assert list(result.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert result["ticker"].tolist() == ["XYZ", "XYZ"]
    assert result["close"].tolist() == [1.1, 1.2]
