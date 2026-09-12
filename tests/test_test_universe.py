from app.data.test_universe import TEST_UNIVERSE, all_test_tickers, test_universe


def test_fixed_research_universe_has_three_groups_of_five():
    assert set(TEST_UNIVERSE) == {"large", "mid", "small"}
    assert all(len(tickers) == 5 for tickers in TEST_UNIVERSE.values())
    assert len(all_test_tickers()) == 15
    assert len(set(all_test_tickers())) == 15


def test_universe_returns_copy():
    copied = test_universe()
    copied["large"].append("TEST")
    assert "TEST" not in TEST_UNIVERSE["large"]
