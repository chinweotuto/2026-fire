"""System configuration settings loaded from environment or .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATABASE_URL: str = "sqlite:///./sports_prediction.db"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    SECRET_KEY: str = "dev-secret-key-change-in-production-1234567890"

    DATA_RAW_DIR: Path = Path("./data/raw")
    DATA_PROCESSED_DIR: Path = Path("./data/processed")
    ARTIFACTS_DIR: Path = Path("./artifacts")
    MODELS_DIR: Path = Path("./artifacts/models")

    # Data sufficiency thresholds (minimum training samples required)
    MIN_SAMPLES_FOOTBALL_1X2: int = 100
    MIN_SAMPLES_FOOTBALL_GOALS: int = 100
    MIN_SAMPLES_BASKETBALL: int = 100
    MIN_SAMPLES_TENNIS: int = 100

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
