from __future__ import annotations

TEST_UNIVERSE: dict[str, list[str]] = {
    "large": ["CBA", "BHP", "CSL", "WBC", "NAB"],
    "mid": ["MQG", "WES", "WOW", "REA", "TCL"],
    "small": ["ALU", "IEL", "BAP", "NXT", "TNE"],
}


def test_universe() -> dict[str, list[str]]:
    """Return a copy of the fixed 15-stock research basket."""
    return {cap: tickers.copy() for cap, tickers in TEST_UNIVERSE.items()}


def all_test_tickers() -> list[str]:
    return [ticker for tickers in TEST_UNIVERSE.values() for ticker in tickers]
