from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from app.config import settings
from app.data.historical_reference import (
    download_asx_code_changes,
    load_delistings,
    save_code_changes,
    save_delistings,
)
from app.data.market_data import download_history, load_universe
from app.data.security_master import build_security_master, save_security_master
from app.data.storage import MarketDataStore
from app.data.universe import download_asx_isin_directory, equity_universe, save_universe
from app.research import (
    download_test_universe,
    optimize_mean_reversion,
    run_momentum_research,
    run_strategy_comparison,
    run_train_test_benchmarks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASX stock screener tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-prices", help="Download historical ASX OHLCV data")
    download.add_argument("--universe", type=Path, required=True)
    download.add_argument("--start", required=True, help="Inclusive start date, e.g. 2021-01-01")
    download.add_argument("--end", required=True, help="Exclusive end date, e.g. 2026-01-01")
    download.add_argument("--continue-on-error", action="store_true")

    test_download = subparsers.add_parser(
        "download-test-data", help="Download historical prices for the fixed 15-stock research basket"
    )
    test_download.add_argument("--start", required=True)
    test_download.add_argument("--end", required=True)

    momentum = subparsers.add_parser(
        "backtest-momentum", help="Run the first breakout-momentum strategy on the test basket"
    )
    momentum.add_argument("--prices-dir", type=Path, default=None)

    comparison = subparsers.add_parser(
        "compare-strategies", help="Run all research strategy families side by side on the test basket"
    )
    comparison.add_argument("--prices-dir", type=Path, default=None)

    validation = subparsers.add_parser(
        "validate-strategies", help="Evaluate all fixed strategy benchmarks on the train/test split"
    )
    validation.add_argument("--prices-dir", type=Path, default=None)
    validation.add_argument("--train-start", type=date.fromisoformat, default=date(2021, 1, 1))
    validation.add_argument("--train-end", type=date.fromisoformat, default=date(2024, 12, 1))
    validation.add_argument("--test-start", type=date.fromisoformat, default=date(2025, 1, 1))
    validation.add_argument("--test-end", type=date.fromisoformat, default=date(2025, 12, 31))

    optimize = subparsers.add_parser(
        "optimize-mean-reversion", help="Optimise mean reversion on train data and validate finalists out of sample"
    )
    optimize.add_argument("--prices-dir", type=Path, default=None)
    optimize.add_argument("--train-start", type=date.fromisoformat, default=date(2021, 1, 1))
    optimize.add_argument("--train-end", type=date.fromisoformat, default=date(2024, 12, 1))
    optimize.add_argument("--test-start", type=date.fromisoformat, default=date(2025, 1, 1))
    optimize.add_argument("--test-end", type=date.fromisoformat, default=date(2025, 12, 31))
    optimize.add_argument("--top-n", type=int, default=10)
    optimize.add_argument("--min-trades", type=int, default=20)

    universe = subparsers.add_parser(
        "download-universe", help="Download the current ASX equity candidate universe"
    )
    universe.add_argument("--output", type=Path, default=None)
    universe.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional path for the complete unfiltered ASX instrument directory",
    )

    master = subparsers.add_parser(
        "build-security-master",
        help="Build the security-master foundation from a normalized universe CSV",
    )
    master.add_argument("--universe", type=Path, default=None)
    master.add_argument("--output", type=Path, default=None)

    code_changes = subparsers.add_parser(
        "download-code-changes", help="Download ASX historical code/name changes"
    )
    code_changes.add_argument("--output", type=Path, default=None)

    delistings = subparsers.add_parser(
        "import-delistings", help="Import an externally obtained historical delisting CSV"
    )
    delistings.add_argument("--input", type=Path, required=True)
    delistings.add_argument("--output", type=Path, default=None)

    ingest = subparsers.add_parser("ingest", help="Create/refresh the local DuckDB prices view")
    ingest.add_argument("--prices-dir", type=Path, default=None)

    return parser


def download_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    tickers = load_universe(args.universe)
    print(f"Downloading {len(tickers)} ASX tickers from {args.start} to {args.end}")

    failures = 0
    for index, ticker in enumerate(tickers, start=1):
        try:
            path = download_history(ticker, args.start, args.end, settings.prices_dir)
            print(f"[{index}/{len(tickers)}] {ticker}: {path}")
        except Exception as exc:  # noqa: BLE001 - one bad ticker should not stop a batch
            failures += 1
            print(f"[{index}/{len(tickers)}] {ticker}: ERROR: {exc}")
            if not args.continue_on_error:
                raise

    print(f"Completed with {failures} failure(s)")
    return 1 if failures else 0


def download_test_data_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    failures = download_test_universe(args.start, args.end)
    print(f"Test-universe download completed with {failures} failure(s)")
    return 1 if failures else 0


def _print_result(name: str, result) -> None:
    print(f"{name}")
    print(f"  Trades: {len(result.trades)}")
    print(f"  Final capital: ${result.final_capital:,.2f}")
    print(f"  Total return: {result.total_return_pct:.2f}%")
    print(f"  Win rate: {result.win_rate_pct:.2f}%")
    print(f"  Profit factor: {result.profit_factor:.2f}")
    print(f"  Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"  Average trade: {result.average_trade_pct:.2f}%")
    print(f"  Median trade: {result.median_trade_pct:.2f}%")
    print(f"  Average holding days: {result.average_holding_days:.2f}")


def backtest_momentum_command(args: argparse.Namespace) -> int:
    research = run_momentum_research(args.prices_dir)
    result = research["overall"]
    print("Breakout momentum research")
    _print_result("", result)
    print("\nBy cap group")
    for group, metrics in research["by_group"].items():
        print(
            f"{group:>5}: trades={metrics['trades']}, "
            f"net_pnl=${metrics['net_pnl']:,.2f}, "
            f"win_rate={metrics['win_rate_pct']:.2f}%"
        )
    return 0


def compare_strategies_command(args: argparse.Namespace) -> int:
    comparison = run_strategy_comparison(args.prices_dir)
    print("Strategy comparison — 15-stock research basket")
    print("Assumptions: $10,000 initial capital, 10% position size, $11 buy + $11 sell brokerage, 20-day max hold")
    print()
    headers = ["Strategy", "Trades", "Final $", "Return", "Win %", "PF", "Max DD", "Avg trade", "Avg days"]
    print(" | ".join(f"{header:<18}" for header in headers))
    print("-" * 125)
    for name, research in comparison.items():
        result = research["overall"]
        print(
            f"{name:<18} | {len(result.trades):>6} | ${result.final_capital:>9,.2f} | "
            f"{result.total_return_pct:>7.2f}% | {result.win_rate_pct:>6.2f}% | "
            f"{result.profit_factor:>4.2f} | {result.max_drawdown_pct:>7.2f}% | "
            f"{result.average_trade_pct:>8.2f}% | {result.average_holding_days:>8.2f}"
        )

    print("\nNet P&L by cap group")
    print("Strategy            Large         Mid       Small")
    print("-" * 55)
    for name, research in comparison.items():
        groups = research["by_group"]
        print(
            f"{name:<18} ${groups['large']['net_pnl']:>9,.2f} "
            f"${groups['mid']['net_pnl']:>9,.2f} ${groups['small']['net_pnl']:>9,.2f}"
        )
    return 0


def _print_validation_row(name: str, train, test) -> None:
    print(
        f"{name:<20} | {len(train.trades):>6} | {train.total_return_pct:>7.2f}% | "
        f"{train.profit_factor:>5.2f} | {train.max_drawdown_pct:>7.2f}% || "
        f"{len(test.trades):>6} | {test.total_return_pct:>7.2f}% | "
        f"{test.profit_factor:>5.2f} | {test.max_drawdown_pct:>7.2f}%"
    )


def validate_strategies_command(args: argparse.Namespace) -> int:
    results = run_train_test_benchmarks(
        args.prices_dir,
        args.train_start,
        args.train_end,
        args.test_start,
        args.test_end,
    )
    print("Train/test validation — fixed strategy benchmarks")
    print(f"Train: {args.train_start} to {args.train_end} (purged before test)")
    print(f"Test:  {args.test_start} to {args.test_end}")
    print("\nStrategy             | Train trades | Train ret | Train PF | Train DD || Test trades | Test ret | Test PF | Test DD")
    print("-" * 122)
    for name, result in results.items():
        _print_validation_row(name, result["train"], result["test"])
    return 0


def optimize_mean_reversion_command(args: argparse.Namespace) -> int:
    research = optimize_mean_reversion(
        args.prices_dir,
        args.train_start,
        args.train_end,
        args.test_start,
        args.test_end,
        args.top_n,
        args.min_trades,
    )
    print("Mean-reversion parameter optimisation")
    print(f"Grid combinations: {research['grid_size']}")
    print(f"Eligible train candidates: {research['eligible_candidates']} (minimum {research['min_trades']} trades)")
    print(f"Train: {research['train_start']} to {research['train_end']} (purged before test)")
    print(f"Test:  {research['test_start']} to {research['test_end']}")
    print("\nRanked finalists — train vs untouched test")
    print(
        "Rank | Lookback | StdDev | RSI | Threshold | MaxHold | "
        "Train Trades | Train Ret | Train PF | Test Trades | Test Ret | Test PF | Test DD"
    )
    print("-" * 145)
    for rank, candidate in enumerate(research["validation"], start=1):
        p = candidate["parameters"]
        train = candidate["train"]
        test = candidate["test"]
        print(
            f"{rank:>4} | {p.lookback_days:>8} | {p.stddevs:>6.1f} | {p.rsi_period:>3} | "
            f"{p.rsi_threshold:>9.1f} | {p.max_holding_days:>7} | {len(train.trades):>11} | "
            f"{train.total_return_pct:>9.2f}% | {train.profit_factor:>8.2f} | "
            f"{len(test.trades):>11} | {test.total_return_pct:>8.2f}% | "
            f"{test.profit_factor:>7.2f} | {test.max_drawdown_pct:>7.2f}%"
        )
    return 0


def download_universe_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    raw_output = args.raw_output
    output = args.output or (settings.data_dir / "universe.csv")

    universe = download_asx_isin_directory()
    candidates = equity_universe(universe)

    if raw_output is not None:
        save_universe(universe, raw_output)
        print(f"Saved {len(universe):,} raw ASX instruments to {raw_output}")

    save_universe(candidates, output)
    print(f"Saved {len(candidates):,} equity candidates to {output}")
    print(f"Excluded {len(universe) - len(candidates):,} non-equity/other instruments")
    return 0


def build_security_master_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    universe_path = args.universe or (settings.data_dir / "universe.csv")
    output = args.output or (settings.data_dir / "security_master.csv")
    universe = pd.read_csv(universe_path)
    master = build_security_master(universe)
    save_security_master(master, output)
    print(f"Saved {len(master):,} securities to {output}")
    print("Listing/delisting dates remain unknown until historical reference data is added")
    return 0


def download_code_changes_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    output = args.output or (settings.data_dir / "code_changes.csv")
    changes = download_asx_code_changes()
    save_code_changes(changes, output)
    print(f"Saved {len(changes):,} ASX code/name changes to {output}")
    return 0


def import_delistings_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    output = args.output or (settings.data_dir / "delistings.csv")
    delisted = load_delistings(args.input)
    save_delistings(delisted, output)
    print(f"Imported {len(delisted):,} historical delistings to {output}")
    return 0


def ingest_command(args: argparse.Namespace) -> int:
    prices_dir = args.prices_dir or settings.prices_dir
    store = MarketDataStore(settings.duckdb_path)
    count = store.ingest_parquet_directory(prices_dir)
    print(f"Loaded {count:,} price rows into the local DuckDB view")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "download-prices":
        return download_command(args)
    if args.command == "download-test-data":
        return download_test_data_command(args)
    if args.command == "backtest-momentum":
        return backtest_momentum_command(args)
    if args.command == "compare-strategies":
        return compare_strategies_command(args)
    if args.command == "validate-strategies":
        return validate_strategies_command(args)
    if args.command == "optimize-mean-reversion":
        return optimize_mean_reversion_command(args)
    if args.command == "download-universe":
        return download_universe_command(args)
    if args.command == "build-security-master":
        return build_security_master_command(args)
    if args.command == "download-code-changes":
        return download_code_changes_command(args)
    if args.command == "import-delistings":
        return import_delistings_command(args)
    if args.command == "ingest":
        return ingest_command(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
