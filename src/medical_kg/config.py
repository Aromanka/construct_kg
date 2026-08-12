from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")


class LLMSettings(BaseModel):
    provider: str = "openai"
    model: str
    api_key: SecretStr
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout: float = Field(default=120.0, gt=0)


class ProcessingSettings(BaseModel):
    max_concurrency: int = Field(default=8, ge=1)
    requests_per_minute: int = Field(default=100, ge=1)
    tokens_per_minute: int = Field(default=1_000_000, ge=1)
    max_retries: int = Field(default=4, ge=0)
    request_timeout: float = Field(default=120.0, gt=0)
    retry_backoff: float = Field(default=2.0, gt=0)


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = "medical_kg"
    user: str
    password: SecretStr
    echo: bool = False

    @property
    def url(self) -> str:
        from urllib.parse import quote_plus

        password = quote_plus(self.password.get_secret_value())
        user = quote_plus(self.user)
        return f"postgresql+asyncpg://{user}:{password}@{self.host}:{self.port}/{self.database}"


class ExtractionSettings(BaseModel):
    passes: list[str] = Field(default_factory=lambda: ["general"])
    stage_version: str = "extract:v1"
    code_version: str = "unknown"

    @model_validator(mode="after")
    def unique_passes(self) -> "ExtractionSettings":
        if not self.passes or len(self.passes) != len(set(self.passes)):
            raise ValueError("extraction.passes must be a non-empty list of unique names")
        return self


class RelationSettings(BaseModel):
    vocabulary_file: Path = Path("config/relations.yaml")


class PromptSettings(BaseModel):
    directory: Path = Path("prompts")


class LoggingSettings(BaseModel):
    level: str = "INFO"
    json: bool = True


class AppSettings(BaseModel):
    llm: LLMSettings
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    database: DatabaseSettings
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    relations: RelationSettings = Field(default_factory=RelationSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    project_root: Path = Field(exclude=True)

    @model_validator(mode="after")
    def resolve_paths(self) -> "AppSettings":
        if not self.relations.vocabulary_file.is_absolute():
            self.relations.vocabulary_file = self.project_root / self.relations.vocabulary_file
        if not self.prompts.directory.is_absolute():
            self.prompts.directory = self.project_root / self.prompts.directory
        return self


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        resolved = os.getenv(name, default)
        if resolved is None:
            raise ValueError(f"Required environment variable {name!r} is not set")
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def load_settings(path: str | Path = "config.yaml") -> AppSettings:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}. Copy config.example.yaml to config.yaml."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw = _expand_environment(raw)
    raw["project_root"] = config_path.parent
    return AppSettings.model_validate(raw)
