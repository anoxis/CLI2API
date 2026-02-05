"""Application settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CLI2API_",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CLI Path (auto-detect if not set)
    claude_cli_path: Optional[str] = Field(
        default=None,
        description="Path to claude CLI executable",
    )

    # Timeout
    default_timeout: int = Field(
        default=300,
        description="Default CLI execution timeout in seconds",
    )

    # Default model
    default_model: str = Field(
        default="sonnet",
        description="Default model to use if not specified",
    )

    # Custom models (comma-separated, e.g. "sonnet,opus,haiku")
    claude_models: Optional[str] = Field(
        default=None,
        description="Comma-separated list of Claude models to expose",
    )

    # Logging
    log_level: str = "INFO"
    log_json: bool = Field(
        default=False,
        description="Use JSON format for structured logging",
    )

    def get_claude_models(self) -> list[str]:
        """Get list of Claude models.

        Claude CLI doesn't have a models cache, so we use defaults
        or user-configured models.
        """
        if self.claude_models:
            return [m.strip() for m in self.claude_models.split(",")]
        # Default Claude models (aliases supported by CLI)
        return ["sonnet", "opus", "haiku"]

    @field_validator("claude_cli_path", mode="before")
    @classmethod
    def detect_claude_cli(cls, v: Optional[str]) -> Optional[str]:
        """Auto-detect claude CLI path if not provided."""
        from cli2api.utils.cli_detector import (
            cache_path,
            detect_claude_cli,
            verify_claude_executable,
        )
        from cli2api.utils.logging import get_logger

        logger = get_logger(__name__)

        if v:
            # User explicitly set path - verify it
            path = Path(v)
            if verify_claude_executable(path):
                logger.info(f"Using explicitly set Claude CLI path: {v}")
                # Cache explicitly set path
                cache_path(path)
                return v
            logger.warning(f"Explicitly set Claude CLI path is invalid: {v}")

        # Auto-detect (includes cache check)
        detected_path = detect_claude_cli()

        if detected_path:
            logger.info(f"Auto-detected Claude CLI: {detected_path}")
            # Cache detected path for next startup
            cache_path(detected_path)
            return str(detected_path)

        logger.error("Claude CLI not found")
        return None
