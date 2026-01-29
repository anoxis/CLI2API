"""FastAPI dependencies for dependency injection."""

from functools import lru_cache
from pathlib import Path

from cli2api.config.settings import Settings
from cli2api.providers.claude import ClaudeCodeProvider


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Singleton Settings instance.
    """
    return Settings()


@lru_cache
def get_provider() -> ClaudeCodeProvider:
    """Get cached Claude provider.

    Returns:
        Singleton ClaudeCodeProvider instance.

    Raises:
        RuntimeError: If Claude CLI is not configured.
    """
    settings = get_settings()
    if not settings.claude_cli_path:
        raise RuntimeError("Claude CLI not found. Set CLI2API_CLAUDE_CLI_PATH.")

    return ClaudeCodeProvider(
        executable_path=Path(settings.claude_cli_path),
        default_timeout=settings.default_timeout,
        models=settings.get_claude_models(),
    )
