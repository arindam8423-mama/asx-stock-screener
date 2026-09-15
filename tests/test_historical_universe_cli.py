from pathlib import Path

import pandas as pd

from app.historical_universe_cli import audit_historical_universe


def test_audit_reports_missing_metadata_without_inference(tmp_path: Path, capsys) -> None:
    research_path = tmp_path / "research.csv"
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "company_name": ["AAA LIMITED"],
            "isin": ["AU000000AAA1"],
            "instrument_type": ["equity_candidate"],
            "source": ["test"],
        }
    ).to_csv(research_path, index=False)

    audit_historical_universe(research_path, tmp_path / "missing.csv")
    output = capsys.readouterr().out

    assert "Current candidates:       1" in output
    assert "NOT FOUND" in output
    assert "No point-in-time universe will be inferred" in output


def test_audit_reports_point_in_time_counts(tmp_path: Path, capsys) -> None:
    research_path = tmp_path / "research.csv"
    metadata_path = tmp_path / "metadata.csv"

    pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "company_name": ["AAA LIMITED", "BBB LIMITED"],
            "isin": ["AU000000AAA1", "AU000000BBB2"],
            "instrument_type": ["equity_candidate"] * 2,
            "source": ["test"] * 2,
        }
    ).to_csv(research_path, index=False)
    pd.DataFrame(
        {
            "security_id": ["AU000000AAA1", "AU000000BBB2"],
            "listing_date": ["2020-01-01", "2024-01-01"],
            "delisting_date": [None, "2024-06-30"],
            "status": ["active", "delisted"],
            "source": ["test", "test"],
        }
    ).to_csv(metadata_path, index=False)

    audit_historical_universe(research_path, metadata_path)
    output = capsys.readouterr().out

    assert "Listing dates known:      2" in output
    assert "2021-01-01: 1" in output
    assert "2025-01-01: 1" in output
