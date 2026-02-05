"""Tests for configuration and settings."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cli2api.config.settings import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Test default settings values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(
                _env_file=None,
                claude_cli_path=None,
            )

            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            assert settings.debug is False
            assert settings.default_timeout == 300
            assert settings.default_model == "sonnet"
            assert settings.log_level == "INFO"

    def test_env_override(self):
        """Test environment variable override."""
        env_vars = {
            "CLI2API_HOST": "127.0.0.1",
            "CLI2API_PORT": "9000",
            "CLI2API_DEBUG": "true",
            "CLI2API_DEFAULT_TIMEOUT": "600",
            "CLI2API_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(
                _env_file=None,
                claude_cli_path=None,
            )

            assert settings.host == "127.0.0.1"
            assert settings.port == 9000
            assert settings.debug is True
            assert settings.default_timeout == 600
            assert settings.log_level == "DEBUG"

    def test_cli_path_from_env(self):
        """Test CLI path from environment."""
        env_vars = {
            "CLI2API_CLAUDE_CLI_PATH": "/custom/path/to/claude",
        }

        # Mock verify_claude_executable to accept the custom path
        with patch("cli2api.utils.cli_detector.verify_claude_executable", return_value=True):
            with patch.dict(os.environ, env_vars, clear=True):
                settings = Settings(_env_file=None)

                assert settings.claude_cli_path == "/custom/path/to/claude"

    def test_cli_path_auto_detect(self):
        """Test CLI path auto-detection."""
        # Mock detect_claude_cli to return a detected path
        with patch("cli2api.utils.cli_detector.detect_claude_cli") as mock_detect:
            mock_detect.return_value = Path("/detected/path/to/claude")

            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=None)

                assert settings.claude_cli_path == "/detected/path/to/claude"

    def test_cli_path_not_found(self):
        """Test when CLI is not found."""
        # Mock detect_claude_cli to return None (not found)
        with patch("cli2api.utils.cli_detector.detect_claude_cli", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=None)

                assert settings.claude_cli_path is None

    def test_explicit_path_overrides_auto_detect(self):
        """Test that explicit path overrides auto-detection."""
        env_vars = {
            "CLI2API_CLAUDE_CLI_PATH": "/explicit/claude",
        }

        # Mock verify to accept explicit path
        with patch("cli2api.utils.cli_detector.verify_claude_executable", return_value=True):
            # Mock detect_claude_cli (should not be called when explicit path is valid)
            with patch("cli2api.utils.cli_detector.detect_claude_cli", return_value=Path("/auto/detected")):
                with patch.dict(os.environ, env_vars, clear=True):
                    settings = Settings(_env_file=None)

                    assert settings.claude_cli_path == "/explicit/claude"

    def test_get_claude_models_default(self):
        """Test default Claude models."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None, claude_cli_path=None)

            models = settings.get_claude_models()
            assert "sonnet" in models
            assert "opus" in models
            assert "haiku" in models

    def test_get_claude_models_custom(self):
        """Test custom Claude models from env."""
        env_vars = {
            "CLI2API_CLAUDE_MODELS": "custom1,custom2,custom3",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None, claude_cli_path=None)

            models = settings.get_claude_models()
            assert models == ["custom1", "custom2", "custom3"]


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_port_as_string(self):
        """Test that string port is converted to int."""
        env_vars = {"CLI2API_PORT": "8080"}

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(
                _env_file=None,
                claude_cli_path=None,
            )

            assert settings.port == 8080
            assert isinstance(settings.port, int)

    def test_debug_true_values(self):
        """Test various truthy values for debug."""
        for value in ["true", "True", "TRUE", "1", "yes"]:
            env_vars = {"CLI2API_DEBUG": value}

            with patch.dict(os.environ, env_vars, clear=True):
                settings = Settings(
                    _env_file=None,
                    claude_cli_path=None,
                )

                assert settings.debug is True

    def test_debug_false_values(self):
        """Test various falsy values for debug."""
        for value in ["false", "False", "FALSE", "0", "no"]:
            env_vars = {"CLI2API_DEBUG": value}

            with patch.dict(os.environ, env_vars, clear=True):
                settings = Settings(
                    _env_file=None,
                    claude_cli_path=None,
                )

                assert settings.debug is False
