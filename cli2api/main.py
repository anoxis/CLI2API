"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cli2api.api.dependencies import get_provider, get_settings
from cli2api.api.router import api_router
from cli2api.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Single source of truth for version from pyproject.toml
__version__ = version("cli2api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    provider = get_provider()

    # Setup structured logging
    setup_logging(
        level=settings.log_level,
        json_format=settings.log_json,
        logger_name=None,  # Configure root logger
    )

    logger.info(
        "Starting CLI2API",
        extra={
            "version": __version__,
            "host": settings.host,
            "port": settings.port,
            "available_models": provider.supported_models,
        },
    )

    yield

    logger.info("CLI2API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="CLI2API",
        description="OpenAI-compatible API over Claude Code CLI",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware for browser clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(api_router)

    # Validation error handler with logging
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        body = await request.body()
        logger.error(f"Validation error for {request.method} {request.url}")
        logger.error(f"Request body: {body.decode()[:1000]}")
        logger.error(f"Validation errors: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "Invalid request format",
                    "type": "invalid_request_error",
                    "details": exc.errors(),
                }
            },
        )

    # Health check endpoint
    @app.get("/health")
    async def health():
        """Health check endpoint with CLI availability check."""
        settings = get_settings()

        async def check_cli(path: str | None) -> str:
            """Check if a CLI tool is available."""
            if not path:
                return "not_configured"
            try:
                proc = await asyncio.create_subprocess_exec(
                    path,
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=2.0)
                return "available" if proc.returncode == 0 else "error"
            except Exception:
                return "unavailable"

        claude_status = await check_cli(settings.claude_cli_path)

        return {
            "status": "healthy" if claude_status == "available" else "degraded",
            "cli": {"claude": claude_status},
        }

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        provider = get_provider()
        return {
            "name": "CLI2API",
            "version": __version__,
            "description": "OpenAI-compatible API over Claude Code CLI",
            "available_models": [f"claude: {m}" for m in provider.supported_models],
            "endpoints": {
                "chat_completions": "/v1/chat/completions",
                "models": "/v1/models",
                "health": "/health",
            },
        }

    return app


# Application instance
app = create_app()
