from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    registry_root: str
    primary_model_name: str
    primary_model_version: str

    secondary_model_name: str
    secondary_model_version: str
    secondary_model_ratio: float

    feedback_dir: Path = Path("feedback")

    drift_output_path: Path = Path("outputs/drift")
    drift_threshold: float = 2.0
    drift_feedback_window: int = 50
    drift_min_samples: int = 50

    device: str = "cpu"
    cors_origins: list[str] = ["*"]
    max_upload_bytes: int = 50 * 1024 * 1024


settings = Settings()  # type: ignore
