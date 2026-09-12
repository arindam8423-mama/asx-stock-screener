from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings shared by data and backtest components."""

    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/asx.duckdb")

    buy_brokerage_aud: float = 11.0
    sell_brokerage_aud: float = 11.0
    max_holding_days: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ASX_",
        extra="ignore",
    )

    @property
    def prices_dir(self) -> Path:
        return self.data_dir / "prices"

    def ensure_data_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
