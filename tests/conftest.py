"""Pytest configuration and fixtures."""

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from cli2api.main import app
from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.internal import ProviderChunk, ProviderResult
from cli2api.schemas.openai import ChatMessage


@pytest.fixture
def client():
    """Create a sync test client for the API."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client for streaming tests."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class MockClaudeProvider(ClaudeCodeProvider):
    """Mock Claude provider for testing without real CLI."""

    def __init__(
        self,
        response_content: str = "Mock response",
        stream_chunks: list[str] | None = None,
        should_fail: bool = False,
        fail_message: str = "Mock error",
    ):
        self.response_content = response_content
        self.stream_chunks = stream_chunks or ["Hello", " ", "World", "!"]
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.executable_path = Path("/mock/claude")
        self.default_timeout = 300
        self.supported_models = ["sonnet", "opus", "haiku"]

    async def execute(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> ProviderResult:
        if self.should_fail:
            raise RuntimeError(self.fail_message)
        return ProviderResult(
            content=self.response_content,
            session_id="mock-session-123",
            usage={"input_tokens": 10, "output_tokens": 20},
        )

    async def execute_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> AsyncIterator[ProviderChunk]:
        if self.should_fail:
            raise RuntimeError(self.fail_message)
        for chunk in self.stream_chunks:
            yield ProviderChunk(content=chunk)
        yield ProviderChunk(content="", is_final=True)


@pytest.fixture
def mock_provider():
    """Create a mock provider instance."""
    return MockClaudeProvider()


@pytest.fixture
def mock_provider_factory():
    """Factory for creating mock providers with custom settings."""

    def _create(
        response_content: str = "Mock response",
        stream_chunks: list[str] | None = None,
        should_fail: bool = False,
        fail_message: str = "Mock error",
    ):
        return MockClaudeProvider(
            response_content=response_content,
            stream_chunks=stream_chunks,
            should_fail=should_fail,
            fail_message=fail_message,
        )

    return _create


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
    ]


@pytest.fixture
def mock_subprocess_success():
    """Mock successful subprocess execution."""

    async def mock_communicate(input=None):
        return (
            json.dumps(
                {
                    "result": "Test response from CLI",
                    "session_id": "test-session-123",
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                }
            ).encode(),
            b"",
        )

    mock_proc = AsyncMock()
    mock_proc.communicate = mock_communicate
    mock_proc.returncode = 0

    return mock_proc


@pytest.fixture
def mock_subprocess_stream():
    """Mock streaming subprocess execution."""

    class MockStdout:
        def __init__(self):
            self.lines = [
                b'{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n',
                b'{"type": "content_block_delta", "delta": {"type": "text_delta", "text": " World"}}\n',
                b'{"type": "result", "usage": {"input_tokens": 5, "output_tokens": 10}}\n',
            ]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.lines):
                raise StopAsyncIteration
            line = self.lines[self.index]
            self.index += 1
            return line

    class MockStdin:
        def write(self, data):
            pass

        async def drain(self):
            pass

        def close(self):
            pass

    mock_proc = AsyncMock()
    mock_proc.stdout = MockStdout()
    mock_proc.stdin = MockStdin()
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    return mock_proc
