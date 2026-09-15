from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from app.research_cross_sectional import optimize_cross_sectional_momentum, run_cross_sectional_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-sectional momentum research tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run the fixed 20-day cross-sectional momentum benchmark")
    benchmark.add_argument("--prices-dir", type=Path, default=None)

    optimize = subparsers.add_parser("optimize", help="Optimise cross-sectional momentum on train data and validate out of sample")
    optimize.add_argument("--prices-dir", type=Path, default=None)
    optimize.add_argument("--train-start", type=date.fromisoformat, default=date(2021, 1, 1))
    optimize.add_argument("--train-end", type=date.fromisoformat, default=date(2024, 12, 1))
    optimize.add_argument("--test-start", type=date.fromisoformat, default=date(2025, 1, 1))
    optimize.add_argument("--test-end", type=date.fromisoformat, default=date(2025, 12, 31))
    optimize.add_argument("--top-n-results", type=int, default=10)
    optimize.add_argument("--min-trades", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "benchmark":
        research = run_cross_sectional_benchmark(args.prices_dir)
        p = research["parameters"]
        result = research["overall"]
        print("Cross-sectional momentum benchmark")
        print(f"Parameters: {p.lookback_days}-day relative strength, top {p.top_n}, {p.max_holding_days}-day scheduled rebalance")
        print("Assumptions: $10,000 initial capital, 10% of current equity per position, $11 buy + $11 sell brokerage")
        print()
        print(f"Trades: {len(result.trades)}")
        print(f"Final capital: ${result.final_capital:,.2f}")
        print(f"Total return: {result.total_return_pct:.2f}%")
        print(f"Win rate: {result.win_rate_pct:.2f}%")
        print(f"Profit factor: {result.profit_factor:.2f}")
        print(f"Max drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"Average trade: {result.average_trade_pct:.2f}%")
        print(f"Average holding days: {result.average_holding_days:.2f}")
        print("\nBy cap group")
        for group, metrics in research["by_group"].items():
            print(f"{group:>5}: trades={metrics['trades']}, net_pnl=${metrics['net_pnl']:,.2f}, win_rate={metrics['win_rate_pct']:.2f}%")
        return 0

    research = optimize_cross_sectional_momentum(
        args.prices_dir,
        args.train_start,
        args.train_end,
        args.test_start,
        args.test_end,
        args.top_n_results,
        args.min_trades,
    )
    print("Cross-sectional momentum parameter optimisation")
    print("NOTE: finalists are selected using train data only; test results are out of sample.")
    print(f"Grid combinations: {research['grid_size']}")
    print(f"Eligible train candidates: {research['eligible_candidates']} (minimum {research['min_trades']} trades)")
    print(f"Train: {research['train_start']} to {research['train_end']} (purged before test)")
    print(f"Test:  {research['test_start']} to {research['test_end']}")
    print("\nRank | Lookback | Top N | MaxHold | Train Trades | Train Ret | Train PF | Test Trades | Test Ret | Test PF | Test DD")
    print("-" * 120)
    for rank, candidate in enumerate(research["validation"], start=1):
        p = candidate["parameters"]
        train = candidate["train"]
        test = candidate["test"]
        print(
            f"{rank:>4} | {p.lookback_days:>8} | {p.top_n:>5} | {p.max_holding_days:>7} | "
            f"{len(train.trades):>11} | {train.total_return_pct:>9.2f}% | {train.profit_factor:>8.2f} | "
            f"{len(test.trades):>11} | {test.total_return_pct:>8.2f}% | {test.profit_factor:>7.2f} | {test.max_drawdown_pct:>7.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
