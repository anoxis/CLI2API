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
        error_msg = """
Claude CLI not found. Please:

1. Install Claude Code: https://docs.anthropic.com/en/docs/claude-code
2. Or set environment variable: export CLI2API_CLAUDE_CLI_PATH=/path/to/claude
3. Or ensure Claude is in your PATH

Auto-detection checked:
- Cached path (~/.cache/cli2api/claude_cli_path)
- System PATH (shutil.which)
- Running processes (ps aux / wmic)
- VSCode extensions (platform-specific)
- NPM global packages
- Common paths (platform-specific)

For debugging, run: CLI2API_LOG_LEVEL=DEBUG cli2api

Note: Detected path is cached in ~/.cache/cli2api/claude_cli_path
To force re-detection, delete the cache file.
"""
        raise RuntimeError(error_msg.strip())

    return ClaudeCodeProvider(
        executable_path=Path(settings.claude_cli_path),
        default_timeout=settings.default_timeout,
        models=settings.get_claude_models(),
    )
