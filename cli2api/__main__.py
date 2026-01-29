"""Entry point for running CLI2API as a module."""

import uvicorn

from cli2api.api.dependencies import get_settings


def main():
    """Run the CLI2API server."""
    settings = get_settings()

    uvicorn.run(
        "cli2api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
