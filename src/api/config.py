from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    registry_root: str
    model_name: str
    model_version: str
    device: str = "cpu"
    cors_origins: list[str] = ["*"]
    max_upload_bytes: int = 50 * 1024 * 1024  # 50MB
    feedback_dir: Path = Path("feedback")


settings = Settings()  # type: ignore
