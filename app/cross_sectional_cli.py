from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from app.cross_sectional_walk_forward import run_walk_forward
from app.research_cross_sectional import (
    optimize_cross_sectional_momentum,
    optimize_volatility_adjusted_momentum,
    run_cross_sectional_benchmark,
    run_trend_filtered_benchmark,
    run_volatility_adjusted_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-sectional momentum research tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("benchmark", "Run raw momentum benchmark"), ("trend-filtered-benchmark", "Run momentum with trend filter"), ("vol-adjusted-benchmark", "Run volatility-adjusted momentum benchmark")):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--prices-dir", type=Path, default=None)
    for name, help_text in (("optimize", "Optimise momentum + trend filters"), ("vol-adjusted-optimize", "Optimise volatility-adjusted momentum")):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--prices-dir", type=Path, default=None)
        command.add_argument("--train-start", type=date.fromisoformat, default=date(2021, 1, 1))
        command.add_argument("--train-end", type=date.fromisoformat, default=date(2024, 12, 1))
        command.add_argument("--test-start", type=date.fromisoformat, default=date(2025, 1, 1))
        command.add_argument("--test-end", type=date.fromisoformat, default=date(2025, 12, 31))
        command.add_argument("--top-n-results", type=int, default=10)
        command.add_argument("--min-trades", type=int, default=20)
    walk = subparsers.add_parser("walk-forward", help="Run expanding-window walk-forward validation")
    walk.add_argument("--prices-dir", type=Path, default=None)
    walk.add_argument("--min-trades", type=int, default=20)
    return parser


def _print_benchmark(research: dict[str, object], mode: str = "raw") -> None:
    p = research["parameters"]
    r = research["overall"]
    print("Trend-filtered cross-sectional momentum benchmark" if mode == "trend" else "Volatility-adjusted cross-sectional momentum benchmark" if mode == "vol" else "Cross-sectional momentum benchmark")
    if mode == "trend":
        print(f"Parameters: {p.lookback_days}-day return, top {p.top_n}, {p.max_holding_days}-day rebalance, filter={p.trend_filter}")
    elif mode == "vol":
        print(f"Parameters: {p.lookback_days}-day return / {p.volatility_days}-day volatility, top {p.top_n}, {p.max_holding_days}-day rebalance")
    else:
        print(f"Parameters: {p.lookback_days}-day relative strength, top {p.top_n}, {p.max_holding_days}-day rebalance")
    print("Assumptions: $10,000 initial capital, 10% current equity per position, $11 buy + $11 sell brokerage\n")
    print(f"Trades: {len(r.trades)}")
    print(f"Final capital: ${r.final_capital:,.2f}")
    print(f"Total return: {r.total_return_pct:.2f}%")
    print(f"Win rate: {r.win_rate_pct:.2f}%")
    print(f"Profit factor: {r.profit_factor:.2f}")
    print(f"Max drawdown: {r.max_drawdown_pct:.2f}%")
    print(f"Average trade: {r.average_trade_pct:.2f}%")
    print(f"Average holding days: {r.average_holding_days:.2f}")
    if mode == "trend":
        print("\nTrend filter pass rates")
        for name, rate in research["trend_filter_pass_rates"].items():
            print(f"{name}: {rate:.2f}%")
    print("\nBy cap group")
    for group, m in research["by_group"].items():
        print(f"{group:>5}: trades={m['trades']}, net_pnl=${m['net_pnl']:,.2f}, win_rate={m['win_rate_pct']:.2f}%")


def _print_optimization(research: dict[str, object], volatility_adjusted: bool = False) -> None:
    print("Volatility-adjusted cross-sectional momentum parameter optimisation" if volatility_adjusted else "Cross-sectional momentum + trend-filter parameter optimisation")
    print("NOTE: finalists are selected using train data only; test results are out of sample.")
    print(f"Grid combinations: {research['grid_size']}")
    print(f"Eligible train candidates: {research['eligible_candidates']} (minimum {research['min_trades']} trades)")
    print(f"Train: {research['train_start']} to {research['train_end']}")
    print(f"Test:  {research['test_start']} to {research['test_end']}")
    if not volatility_adjusted:
        print("\nTrend filter pass rates")
        for name, rate in research["trend_filter_pass_rates"].items():
            print(f"{name}: {rate:.2f}%")
    if volatility_adjusted:
        print("\nRank | Lookback | VolDays | Top N | Hold | Train Trades | Train Ret | Train PF | Test Trades | Test Ret | Test PF | Test DD")
    else:
        print("\nRank | Lookback | Top N | Hold | Trend Filter | Train Trades | Train Ret | Train PF | Test Trades | Test Ret | Test PF | Test DD")
    for rank, c in enumerate(research["validation"], 1):
        p, train, test = c["parameters"], c["train"], c["test"]
        extra = f"{p.volatility_days:>7} | " if volatility_adjusted else f"{p.trend_filter:>45} | "
        print(f"{rank:>4} | {p.lookback_days:>8} | {extra}{p.top_n:>5} | {p.max_holding_days:>4} | {len(train.trades):>11} | {train.total_return_pct:>9.2f}% | {train.profit_factor:>8.2f} | {len(test.trades):>11} | {test.total_return_pct:>8.2f}% | {test.profit_factor:>7.2f} | {test.max_drawdown_pct:>7.2f}%")


def _print_walk_forward(research: dict[str, object]) -> None:
    print("Expanding-window cross-sectional momentum walk-forward validation")
    print("Each window selects parameters on train data only, then evaluates the selected parameters on the next calendar year.")
    print(f"Grid combinations per window: {research['grid_size']}")
    print(f"Minimum train trades: {research['min_trades']}\n")
    print("Window | Selected parameters | Train Ret | Train PF | Test Trades | Test Ret | Test PF | Test DD")
    for index, row in enumerate(research["windows"], 1):
        if "parameters" not in row:
            print(f"{index} | no eligible candidate")
            continue
        p, train, test = row["parameters"], row["train"], row["test"]
        label = f"{p.lookback_days}d/top{p.top_n}/hold{p.max_holding_days}/{p.trend_filter}"
        print(f"{row['train_start']}→{row['train_end']} / {row['test_start']}→{row['test_end']} | {label} | {train.total_return_pct:>8.2f}% | {train.profit_factor:>8.2f} | {len(test.trades):>11} | {test.total_return_pct:>8.2f}% | {test.profit_factor:>7.2f} | {test.max_drawdown_pct:>7.2f}%")


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "benchmark":
        _print_benchmark(run_cross_sectional_benchmark(args.prices_dir))
    elif args.command == "trend-filtered-benchmark":
        _print_benchmark(run_trend_filtered_benchmark(args.prices_dir), "trend")
    elif args.command == "vol-adjusted-benchmark":
        _print_benchmark(run_volatility_adjusted_benchmark(args.prices_dir), "vol")
    elif args.command == "optimize":
        _print_optimization(optimize_cross_sectional_momentum(args.prices_dir, args.train_start, args.train_end, args.test_start, args.test_end, args.top_n_results, args.min_trades))
    elif args.command == "walk-forward":
        _print_walk_forward(run_walk_forward(args.prices_dir, args.min_trades))
    else:
        _print_optimization(optimize_volatility_adjusted_momentum(args.prices_dir, args.top_n_results, args.min_trades), True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
