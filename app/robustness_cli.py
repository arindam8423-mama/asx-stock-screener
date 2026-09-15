from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from app.research import analyze_mean_reversion_robustness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mean-reversion robustness research")
    parser.add_argument("--prices-dir", type=Path, default=None)
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--test-end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--min-trades", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    research = analyze_mean_reversion_robustness(
        args.prices_dir,
        args.test_start,
        args.test_end,
        args.min_trades,
    )

    eligible = research["eligible_candidates"]
    print("Mean-reversion robustness analysis")
    print("NOTE: test-period results are diagnostic only and are NOT used to select a strategy.")
    print(f"Grid combinations: {research['grid_size']}")
    print(f"Eligible configurations: {eligible} (minimum {research['min_trades']} trades)")
    print(f"Test: {research['test_start']} to {research['test_end']}")
    print(
        f"Positive-return configurations: {research['positive_return_count']} "
        f"({research['positive_return_count'] / eligible * 100:.1f}%)"
    )
    print(
        f"PF > 1.0 configurations: {research['profitable_factor_count']} "
        f"({research['profitable_factor_count'] / eligible * 100:.1f}%)"
    )
    print(
        f"Median test return across configurations: {research['median_return_pct']:.2f}% | "
        f"Median test PF: {research['median_profit_factor']:.2f}"
    )

    print("\nParameter-level robustness (median across configurations holding that value)")
    print("Parameter       | Value | Configs | Positive | PF > 1 | Median Ret | Median PF")
    print("-" * 82)
    for parameter, summaries in research["by_parameter"].items():
        for summary in summaries:
            print(
                f"{parameter:<15} | {summary.value:>5} | {summary.candidates:>7} | "
                f"{summary.positive_return_count:>8} | {summary.profitable_factor_count:>6} | "
                f"{summary.median_return_pct:>10.2f}% | {summary.median_profit_factor:>9.2f}"
            )

    print("\nTop test configurations — diagnostic only")
    print("Rank | Lookback | StdDev | RSI | Threshold | MaxHold | Trades | Return | PF | Max DD")
    print("-" * 91)
    for rank, candidate in enumerate(research["top_diagnostic"], start=1):
        p = candidate["parameters"]
        result = candidate["result"]
        print(
            f"{rank:>4} | {p.lookback_days:>8} | {p.stddevs:>6.1f} | {p.rsi_period:>3} | "
            f"{p.rsi_threshold:>9.1f} | {p.max_holding_days:>7} | {len(result.trades):>6} | "
            f"{result.total_return_pct:>6.2f}% | {result.profit_factor:>4.2f} | {result.max_drawdown_pct:>6.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
