"""Tests for Claude CLI auto-detection."""

import platform
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from cli2api.utils.cli_detector import (
    CACHE_FILE,
    cache_path,
    detect_claude_cli,
    detect_from_common_paths,
    detect_from_npm,
    detect_from_path,
    detect_from_processes,
    detect_from_vscode,
    get_cached_path,
    get_vscode_extensions_dir,
    verify_claude_executable,
)


class TestVerifyClaudeExecutable:
    """Tests for verify_claude_executable function."""

    def test_returns_false_for_nonexistent_path(self):
        """Test that verify returns False for non-existent path."""
        path = Path("/nonexistent/path/to/claude")
        assert verify_claude_executable(path) is False

    def test_returns_false_for_directory(self, tmp_path):
        """Test that verify returns False for directory."""
        dir_path = tmp_path / "claude"
        dir_path.mkdir()
        assert verify_claude_executable(dir_path) is False

    def test_returns_false_for_non_executable(self, tmp_path):
        """Test that verify returns False for non-executable file."""
        file_path = tmp_path / "claude"
        file_path.write_text("#!/bin/bash\necho 'test'")
        file_path.chmod(0o644)  # Not executable
        assert verify_claude_executable(file_path) is False

    @patch("subprocess.run")
    def test_returns_true_for_valid_executable(self, mock_run, tmp_path):
        """Test that verify returns True for valid Claude executable."""
        file_path = tmp_path / "claude"
        file_path.write_text("#!/bin/bash\necho 'claude version 1.0'")
        file_path.chmod(0o755)  # Make executable

        # Mock successful --version check
        mock_run.return_value = Mock(returncode=0)

        assert verify_claude_executable(file_path) is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_returns_false_when_version_check_fails(self, mock_run, tmp_path):
        """Test that verify returns False when --version fails."""
        file_path = tmp_path / "claude"
        file_path.write_text("#!/bin/bash\nexit 1")
        file_path.chmod(0o755)

        # Mock failed --version check
        mock_run.return_value = Mock(returncode=1)

        assert verify_claude_executable(file_path) is False

    @patch("subprocess.run")
    def test_handles_timeout_gracefully(self, mock_run, tmp_path):
        """Test that verify handles timeout gracefully."""
        file_path = tmp_path / "claude"
        file_path.write_text("#!/bin/bash\nsleep 10")
        file_path.chmod(0o755)

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("claude", 2)

        assert verify_claude_executable(file_path) is False


class TestCaching:
    """Tests for path caching functionality."""

    def test_get_cached_path_returns_none_when_no_cache(self, tmp_path, monkeypatch):
        """Test that get_cached_path returns None when cache doesn't exist."""
        # Use temporary cache location
        cache_file = tmp_path / "cache" / "claude_cli_path"
        monkeypatch.setattr("cli2api.utils.cli_detector.CACHE_FILE", cache_file)

        assert get_cached_path() is None

    @patch("cli2api.utils.cli_detector.verify_claude_executable")
    def test_get_cached_path_returns_valid_path(self, mock_verify, tmp_path, monkeypatch):
        """Test that get_cached_path returns valid cached path."""
        cache_file = tmp_path / "cache" / "claude_cli_path"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Write a path to cache
        test_path = Path("/usr/local/bin/claude")
        cache_file.write_text(str(test_path))

        # Mock verification as successful
        mock_verify.return_value = True

        monkeypatch.setattr("cli2api.utils.cli_detector.CACHE_FILE", cache_file)

        result = get_cached_path()
        assert result == test_path
        mock_verify.assert_called_once_with(test_path)

    @patch("cli2api.utils.cli_detector.verify_claude_executable")
    def test_get_cached_path_invalidates_bad_path(self, mock_verify, tmp_path, monkeypatch):
        """Test that get_cached_path removes invalid cache."""
        cache_file = tmp_path / "cache" / "claude_cli_path"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("/invalid/path/claude")

        # Mock verification as failed
        mock_verify.return_value = False

        monkeypatch.setattr("cli2api.utils.cli_detector.CACHE_FILE", cache_file)

        result = get_cached_path()
        assert result is None
        assert not cache_file.exists()

    def test_cache_path_creates_directory(self, tmp_path, monkeypatch):
        """Test that cache_path creates cache directory."""
        cache_file = tmp_path / "new_cache_dir" / "claude_cli_path"
        monkeypatch.setattr("cli2api.utils.cli_detector.CACHE_FILE", cache_file)

        test_path = Path("/usr/local/bin/claude")
        cache_path(test_path)

        assert cache_file.exists()
        assert cache_file.read_text() == str(test_path)

    def test_cache_path_handles_errors_gracefully(self, tmp_path, monkeypatch):
        """Test that cache_path handles write errors gracefully."""
        # Create a read-only directory
        cache_dir = tmp_path / "readonly"
        cache_dir.mkdir()
        cache_file = cache_dir / "claude_cli_path"

        # Make directory read-only
        cache_dir.chmod(0o444)

        monkeypatch.setattr("cli2api.utils.cli_detector.CACHE_FILE", cache_file)

        test_path = Path("/usr/local/bin/claude")
        # Should not raise exception
        cache_path(test_path)

        # Cleanup: restore permissions
        cache_dir.chmod(0o755)


class TestGetVSCodeExtensionsDir:
    """Tests for get_vscode_extensions_dir function."""

    @patch("platform.system")
    def test_returns_windows_path(self, mock_system):
        """Test that Windows path is returned on Windows."""
        mock_system.return_value = "Windows"
        result = get_vscode_extensions_dir()
        assert "AppData" in str(result)
        assert "Code" in str(result)
        assert "extensions" in str(result)

    @patch("platform.system")
    def test_returns_unix_path(self, mock_system):
        """Test that Unix path is returned on macOS/Linux."""
        for system in ["Darwin", "Linux"]:
            mock_system.return_value = system
            result = get_vscode_extensions_dir()
            assert ".vscode" in str(result)
            assert "extensions" in str(result)


class TestDetectFromPath:
    """Tests for detect_from_path function."""

    @patch("shutil.which")
    @patch("cli2api.utils.cli_detector.verify_claude_executable")
    def test_finds_claude_in_path(self, mock_verify, mock_which):
        """Test that detect_from_path finds Claude in PATH."""
        mock_which.return_value = "/usr/local/bin/claude"
        mock_verify.return_value = True

        result = detect_from_path()
        assert result == Path("/usr/local/bin/claude")

    @patch("shutil.which")
    def test_returns_none_when_not_in_path(self, mock_which):
        """Test that detect_from_path returns None when not in PATH."""
        mock_which.return_value = None

        result = detect_from_path()
        assert result is None


class TestDetectFromProcesses:
    """Tests for detect_from_processes function."""

    @patch("platform.system")
    @patch("subprocess.run")
    @patch("cli2api.utils.cli_detector.verify_claude_executable")
    def test_detects_from_ps_aux_on_unix(self, mock_verify, mock_run, mock_system):
        """Test detection from ps aux on Unix systems."""
        mock_system.return_value = "Darwin"
        mock_verify.return_value = True

        # Mock ps aux output with Claude process
        ps_output = """
USER       PID  %CPU %MEM      VSZ    RSS   TT  STAT STARTED      TIME COMMAND
user     12345   1.0  2.0  1234567 123456   ??  S    1:00PM   0:01.23 /path/to/vscode/native-binary/claude --args
"""
        mock_run.return_value = Mock(stdout=ps_output, returncode=0)

        # Need to mock Path.exists() for the extracted path
        with patch("pathlib.Path.exists", return_value=True):
            result = detect_from_processes()
            assert result == Path("/path/to/vscode/native-binary/claude")

    @patch("platform.system")
    @patch("subprocess.run")
    def test_returns_none_when_no_processes_found(self, mock_run, mock_system):
        """Test that returns None when no Claude processes found."""
        mock_system.return_value = "Linux"
        mock_run.return_value = Mock(stdout="no claude processes here", returncode=0)

        result = detect_from_processes()
        assert result is None

    @patch("platform.system")
    @patch("subprocess.run")
    def test_handles_subprocess_timeout(self, mock_run, mock_system):
        """Test that handles subprocess timeout gracefully."""
        mock_system.return_value = "Darwin"
        mock_run.side_effect = subprocess.TimeoutExpired("ps", 2)

        result = detect_from_processes()
        assert result is None


class TestDetectFromVSCode:
    """Tests for detect_from_vscode function."""

    @patch("cli2api.utils.cli_detector.get_vscode_extensions_dir")
    @patch("cli2api.utils.cli_detector.verify_claude_executable")
    @patch("platform.system")
    def test_finds_latest_version(self, mock_system, mock_verify, mock_vscode_dir, tmp_path):
        """Test that detect_from_vscode finds latest version."""
        mock_system.return_value = "Darwin"
        mock_vscode_dir.return_value = tmp_path

        # Create multiple version directories
        ext1 = tmp_path / "anthropic.claude-code-2.1.27-darwin-arm64"
        ext2 = tmp_path / "anthropic.claude-code-2.1.31-darwin-arm64"
        ext3 = tmp_path / "anthropic.claude-code-2.1.29-darwin-arm64"

        for ext_dir in [ext1, ext2, ext3]:
            binary_path = ext_dir / "resources" / "native-binary"
            binary_path.mkdir(parents=True)
            (binary_path / "claude").touch()

        # Verify should succeed for the latest version
        def verify_side_effect(path):
            return "2.1.31" in str(path)

        mock_verify.side_effect = verify_side_effect

        result = detect_from_vscode()
        assert result is not None
        assert "2.1.31" in str(result)

    @patch("cli2api.utils.cli_detector.get_vscode_extensions_dir")
    def test_returns_none_when_vscode_dir_not_found(self, mock_vscode_dir, tmp_path):
        """Test that returns None when VSCode directory doesn't exist."""
        mock_vscode_dir.return_value = tmp_path / "nonexistent"

        result = detect_from_vscode()
        assert result is None

    @patch("cli2api.utils.cli_detector.get_vscode_extensions_dir")
    @patch("platform.system")
    def test_handles_windows_exe(self, mock_system, mock_vscode_dir, tmp_path):
        """Test that handles .exe extension on Windows."""
        mock_system.return_value = "Windows"
        mock_vscode_dir.return_value = tmp_path

        ext_dir = tmp_path / "anthropic.claude-code-2.1.31-win32-x64"
        binary_path = ext_dir / "resources" / "native-binary"
        binary_path.mkdir(parents=True)

        # Windows should look for claude.exe
        exe_file = binary_path / "claude.exe"
        exe_file.touch()

        with patch("cli2api.utils.cli_detector.verify_claude_executable", return_value=True):
            result = detect_from_vscode()
            assert result == exe_file


class TestDetectFromNPM:
    """Tests for detect_from_npm function."""

    @patch("subprocess.run")
    @patch("cli2api.utils.cli_detector.verify_claude_executable")
    def test_finds_claude_in_npm_global(self, mock_verify, mock_run, tmp_path):
        """Test that detect_from_npm finds Claude in npm global packages."""
        npm_root = tmp_path / "npm_global"
        npm_root.mkdir()

        # Create package structure
        package_dir = npm_root / "@anthropic-ai" / "claude-code"
        bin_dir = package_dir / "bin"
        bin_dir.mkdir(parents=True)
        claude_bin = bin_dir / "claude"
        claude_bin.touch()

        # Mock npm root command
        mock_run.return_value = Mock(stdout=str(npm_root), returncode=0)
        mock_verify.return_value = True

        result = detect_from_npm()
        assert result == claude_bin

    @patch("subprocess.run")
    def test_returns_none_when_npm_fails(self, mock_run):
        """Test that returns None when npm command fails."""
        mock_run.side_effect = FileNotFoundError()

        result = detect_from_npm()
        assert result is None


class TestDetectFromCommonPaths:
    """Tests for detect_from_common_paths function."""

    @patch("platform.system")
    @patch("cli2api.utils.cli_detector.verify_claude_executable")
    def test_checks_unix_paths(self, mock_verify, mock_system):
        """Test that checks Unix common paths."""
        mock_system.return_value = "Darwin"

        def verify_side_effect(path):
            return str(path) == "/opt/homebrew/bin/claude"

        mock_verify.side_effect = verify_side_effect

        result = detect_from_common_paths()
        assert result == Path("/opt/homebrew/bin/claude")

    @patch("platform.system")
    @patch("cli2api.utils.cli_detector.verify_claude_executable")
    def test_checks_windows_paths(self, mock_verify, mock_system):
        """Test that checks Windows common paths."""
        mock_system.return_value = "Windows"
        mock_verify.return_value = False

        result = detect_from_common_paths()
        assert result is None


class TestDetectClaudeCLI:
    """Tests for main detect_claude_cli orchestrator."""

    @patch("cli2api.utils.cli_detector.get_cached_path")
    def test_uses_cache_first(self, mock_cache):
        """Test that detect_claude_cli checks cache first."""
        mock_cache.return_value = Path("/cached/claude")

        result = detect_claude_cli()
        assert result == Path("/cached/claude")

    @patch("cli2api.utils.cli_detector.get_cached_path")
    @patch("cli2api.utils.cli_detector.detect_from_path")
    @patch("cli2api.utils.cli_detector.detect_from_processes")
    @patch("cli2api.utils.cli_detector.detect_from_vscode")
    def test_tries_methods_in_order(self, mock_vscode, mock_processes, mock_path, mock_cache):
        """Test that detect_claude_cli tries methods in correct order."""
        # Cache returns None
        mock_cache.return_value = None
        # PATH returns None
        mock_path.return_value = None
        # Processes returns None
        mock_processes.return_value = None
        # VSCode succeeds
        mock_vscode.return_value = Path("/vscode/claude")

        result = detect_claude_cli()
        assert result == Path("/vscode/claude")

        # Verify order of calls
        mock_cache.assert_called_once()
        mock_path.assert_called_once()
        mock_processes.assert_called_once()
        mock_vscode.assert_called_once()

    @patch("cli2api.utils.cli_detector.get_cached_path")
    @patch("cli2api.utils.cli_detector.detect_from_path")
    @patch("cli2api.utils.cli_detector.detect_from_processes")
    @patch("cli2api.utils.cli_detector.detect_from_vscode")
    @patch("cli2api.utils.cli_detector.detect_from_npm")
    @patch("cli2api.utils.cli_detector.detect_from_common_paths")
    def test_returns_none_when_all_fail(
        self, mock_common, mock_npm, mock_vscode, mock_processes, mock_path, mock_cache
    ):
        """Test that returns None when all detection methods fail."""
        # All methods return None
        mock_cache.return_value = None
        mock_path.return_value = None
        mock_processes.return_value = None
        mock_vscode.return_value = None
        mock_npm.return_value = None
        mock_common.return_value = None

        result = detect_claude_cli()
        assert result is None
