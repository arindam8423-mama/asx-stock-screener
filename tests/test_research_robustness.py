from app.research import _robustness_summary


def test_robustness_summary_groups_parameter_values():
    class Parameters:
        def __init__(self, value):
            self.lookback_days = value

    class Result:
        def __init__(self, return_pct, profit_factor):
            self.total_return_pct = return_pct
            self.profit_factor = profit_factor

    candidates = [
        (Parameters(10), Result(2.0, 1.2)),
        (Parameters(10), Result(-1.0, 0.8)),
        (Parameters(20), Result(3.0, 1.4)),
    ]

    summaries = _robustness_summary(candidates, "lookback_days")

    assert [summary.value for summary in summaries] == [10, 20]
    assert summaries[0].candidates == 2
    assert summaries[0].positive_return_count == 1
    assert summaries[0].profitable_factor_count == 1
    assert summaries[1].median_return_pct == 3.0
    assert summaries[1].median_profit_factor == 1.4
