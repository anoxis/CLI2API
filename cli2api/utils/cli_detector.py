"""Claude CLI auto-detection utilities with caching and cross-platform support."""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from cli2api.utils.logging import get_logger

logger = get_logger(__name__)

# Cache file location
CACHE_FILE = Path.home() / ".cache" / "cli2api" / "claude_cli_path"


def verify_claude_executable(path: Path) -> bool:
    """Verify that the path is a valid Claude CLI executable.

    Args:
        path: Path to potential Claude CLI executable.

    Returns:
        True if the path is a valid, executable Claude CLI.
    """
    if not path.exists():
        logger.debug(f"Path does not exist: {path}")
        return False

    if not path.is_file():
        logger.debug(f"Path is not a file: {path}")
        return False

    if not os.access(path, os.X_OK):
        logger.debug(f"Path is not executable: {path}")
        return False

    # Quick version check to confirm it's Claude
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0:
            logger.debug(f"Valid Claude CLI found: {path}")
            return True
        else:
            logger.debug(f"Claude CLI returned non-zero exit code: {path}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
        logger.debug(f"Failed to verify Claude CLI at {path}: {e}")
        return False


def get_cached_path() -> Optional[Path]:
    """Get cached Claude CLI path if still valid.

    Returns:
        Cached path if valid, None otherwise.
    """
    if not CACHE_FILE.exists():
        logger.debug("No cache file found")
        return None

    try:
        cached_path = Path(CACHE_FILE.read_text().strip())
        if verify_claude_executable(cached_path):
            logger.debug(f"Using cached Claude CLI path: {cached_path}")
            return cached_path
        else:
            logger.debug("Cached path is invalid, removing cache")
            CACHE_FILE.unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Failed to read cache: {e}")

    return None


def cache_path(path: Path) -> None:
    """Cache the detected Claude CLI path.

    Args:
        path: Path to cache.
    """
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(str(path))
        logger.debug(f"Cached Claude CLI path: {path}")
    except Exception as e:
        logger.debug(f"Failed to cache path: {e}")


def get_vscode_extensions_dir() -> Path:
    """Get VSCode extensions directory for current OS.

    Returns:
        Path to VSCode extensions directory.
    """
    system = platform.system()
    if system == "Windows":
        return Path.home() / "AppData" / "Roaming" / "Code" / "extensions"
    else:  # macOS and Linux
        return Path.home() / ".vscode" / "extensions"


def detect_from_path() -> Optional[Path]:
    """Detect Claude CLI from system PATH.

    Returns:
        Path to Claude CLI if found in PATH, None otherwise.
    """
    logger.debug("Checking system PATH for Claude CLI")
    path_str = shutil.which("claude")
    if path_str:
        path = Path(path_str)
        if verify_claude_executable(path):
            logger.info(f"Found Claude CLI in system PATH: {path}")
            return path
    logger.debug("Claude CLI not found in system PATH")
    return None


def detect_from_processes() -> Optional[Path]:
    """Detect Claude CLI from running processes (cross-platform).

    Returns:
        Path to Claude CLI if found in running processes, None otherwise.
    """
    system = platform.system()
    logger.debug(f"Checking running processes for Claude CLI on {system}")

    try:
        if system == "Windows":
            # Windows: use wmic to get process executable paths
            result = subprocess.run(
                ["wmic", "process", "where", "name like '%claude%'", "get", "ExecutablePath"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            logger.debug(f"Windows process detection output (first 200 chars): {result.stdout[:200]}")

            for line in result.stdout.splitlines():
                line = line.strip()
                if line and "claude" in line.lower() and Path(line).exists():
                    path = Path(line)
                    if verify_claude_executable(path):
                        logger.info(f"Found Claude in running processes: {path}")
                        return path
        else:
            # macOS/Linux: use ps aux
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            logger.debug(f"Unix process detection found {len(result.stdout.splitlines())} processes")

            for line in result.stdout.splitlines():
                if "claude" in line.lower() and "native-binary" in line:
                    parts = line.split()
                    for part in parts:
                        if "claude" in part and Path(part).exists():
                            path = Path(part)
                            if verify_claude_executable(path):
                                logger.info(f"Found Claude in running processes: {path}")
                                return path
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"Process detection failed: {e}")

    logger.debug("No Claude CLI found in running processes")
    return None


def detect_from_vscode() -> Optional[Path]:
    """Detect Claude CLI from VSCode extensions (cross-platform, version-agnostic).

    Returns:
        Path to Claude CLI if found in VSCode extensions, None otherwise.
    """
    vscode_dir = get_vscode_extensions_dir()
    logger.debug(f"Checking VSCode extensions directory: {vscode_dir}")

    if not vscode_dir.exists():
        logger.debug(f"VSCode extensions directory not found: {vscode_dir}")
        return None

    # Pattern: anthropic.claude-code-*-{platform}
    pattern = "anthropic.claude-code-*"
    matches = sorted(vscode_dir.glob(pattern), reverse=True)  # Latest version first
    logger.debug(f"Found {len(matches)} VSCode extension directories matching pattern")

    for ext_dir in matches:
        # Windows uses .exe
        binary_name = "claude.exe" if platform.system() == "Windows" else "claude"
        claude_path = ext_dir / "resources" / "native-binary" / binary_name

        if verify_claude_executable(claude_path):
            logger.info(f"Found Claude in VSCode extension: {ext_dir.name}")
            return claude_path
        else:
            logger.debug(f"Invalid Claude binary in {ext_dir.name}")

    logger.debug("No valid Claude CLI found in VSCode extensions")
    return None


def detect_from_npm() -> Optional[Path]:
    """Detect Claude CLI from NPM global packages.

    Returns:
        Path to Claude CLI if found in NPM globals, None otherwise.
    """
    logger.debug("Checking NPM global packages for Claude CLI")

    try:
        # Get npm global directory
        result = subprocess.run(
            ["npm", "root", "-g"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode == 0:
            npm_root = Path(result.stdout.strip())
            logger.debug(f"NPM global root: {npm_root}")

            # Common npm package names for Claude
            package_names = [
                "@anthropic-ai/claude-code",
                "claude-code",
                "@anthropic/claude",
            ]

            for package_name in package_names:
                package_dir = npm_root / package_name
                if package_dir.exists():
                    # Try common binary locations
                    binary_paths = [
                        package_dir / "bin" / "claude",
                        package_dir / "bin" / "claude.exe",
                        package_dir / "claude",
                        package_dir / "claude.exe",
                    ]

                    for binary_path in binary_paths:
                        if verify_claude_executable(binary_path):
                            logger.info(f"Found Claude in NPM package: {package_name}")
                            return binary_path

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"NPM detection failed: {e}")

    logger.debug("No Claude CLI found in NPM global packages")
    return None


def detect_from_common_paths() -> Optional[Path]:
    """Detect Claude CLI from common installation paths (cross-platform).

    Returns:
        Path to Claude CLI if found in common paths, None otherwise.
    """
    logger.debug("Checking common installation paths for Claude CLI")

    system = platform.system()

    if system == "Windows":
        common_paths = [
            Path("C:/Program Files/Claude/claude.exe"),
            Path("C:/Program Files (x86)/Claude/claude.exe"),
            Path.home() / "AppData" / "Local" / "Programs" / "Claude" / "claude.exe",
        ]
    else:  # macOS and Linux
        common_paths = [
            Path("/opt/homebrew/bin/claude"),
            Path("/usr/local/bin/claude"),
            Path.home() / ".local/bin/claude",
            Path.home() / "bin/claude",
        ]

    logger.debug(f"Checking {len(common_paths)} common paths")

    for path in common_paths:
        if verify_claude_executable(path):
            logger.info(f"Found Claude CLI in common path: {path}")
            return path
        else:
            logger.debug(f"Not found or invalid: {path}")

    logger.debug("No Claude CLI found in common paths")
    return None


def detect_claude_cli() -> Optional[Path]:
    """Main orchestrator for detecting Claude CLI.

    Tries detection methods in priority order:
    1. Cache
    2. System PATH
    3. Running processes
    4. VSCode extensions
    5. NPM global
    6. Common paths

    Returns:
        Path to Claude CLI if found, None otherwise.
    """
    logger.debug("Starting Claude CLI auto-detection")

    # Try cache first (fastest)
    path = get_cached_path()
    if path:
        return path

    # Try each detection method in order
    detection_methods = [
        ("System PATH", detect_from_path),
        ("Running processes", detect_from_processes),
        ("VSCode extensions", detect_from_vscode),
        ("NPM global", detect_from_npm),
        ("Common paths", detect_from_common_paths),
    ]

    for method_name, method_func in detection_methods:
        logger.debug(f"Trying detection method: {method_name}")
        path = method_func()
        if path:
            logger.info(f"Claude CLI detected via {method_name}: {path}")
            return path

    logger.error("Claude CLI not found by any detection method")
    return None
