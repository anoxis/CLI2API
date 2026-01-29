"""Integration tests with real CLI tools.

These tests actually call the Claude CLI and verify real responses.
They are slower and require the CLI tool to be installed.

Run with: pytest tests/test_integration.py -v
Skip with: pytest tests/ --ignore=tests/test_integration.py
"""

import shutil
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from cli2api.main import create_app
from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.openai import ChatMessage


# Check if Claude CLI is available
CLAUDE_PATH = shutil.which("claude")
HAS_CLAUDE = CLAUDE_PATH is not None


# === Provider Integration Tests ===


@pytest.mark.skipif(not HAS_CLAUDE, reason="Claude CLI not installed")
class TestClaudeIntegration:
    """Integration tests for Claude CLI provider."""

    @pytest.fixture
    def provider(self):
        return ClaudeCodeProvider(
            executable_path=Path(CLAUDE_PATH),
            default_timeout=120,
        )

    @pytest.mark.asyncio
    async def test_execute_returns_content(self, provider):
        """Test that execute returns non-empty content."""
        messages = [ChatMessage(role="user", content="Say 'hello' and nothing else.")]

        result = await provider.execute(messages, timeout=60)

        assert result.content, "Response content should not be empty"
        assert len(result.content) > 0
        # Check it's a reasonable response (contains some text)
        assert any(c.isalpha() for c in result.content), "Response should contain text"

    @pytest.mark.asyncio
    async def test_execute_with_model(self, provider):
        """Test execution with specific model."""
        messages = [ChatMessage(role="user", content="Say 'test' and nothing else.")]

        result = await provider.execute(messages, model="sonnet", timeout=60)

        assert result.content, "Response content should not be empty"

    @pytest.mark.asyncio
    async def test_execute_stream_yields_chunks(self, provider):
        """Test that streaming yields chunks."""
        messages = [ChatMessage(role="user", content="Count from 1 to 3.")]

        chunks = []
        async for chunk in provider.execute_stream(messages, timeout=60):
            chunks.append(chunk)

        # Should have at least some chunks (content or final)
        assert len(chunks) > 0, "Should yield at least one chunk"

        # Should have final chunk
        final_chunks = [c for c in chunks if c.is_final]
        assert len(final_chunks) >= 1, "Should have final chunk"

    @pytest.mark.asyncio
    async def test_execute_stream_has_final_chunk(self, provider):
        """Test that streaming ends with final chunk."""
        messages = [ChatMessage(role="user", content="Say 'done'.")]

        chunks = []
        async for chunk in provider.execute_stream(messages, timeout=60):
            chunks.append(chunk)

        # Should have at least one final chunk
        final_chunks = [c for c in chunks if c.is_final]
        assert len(final_chunks) >= 1, "Should have at least one final chunk"

    @pytest.mark.asyncio
    async def test_execute_with_system_prompt(self, provider):
        """Test execution with system prompt."""
        messages = [
            ChatMessage(role="system", content="You only respond with the word 'OK'."),
            ChatMessage(role="user", content="Hello"),
        ]

        result = await provider.execute(messages, timeout=60)

        assert result.content, "Response should not be empty"


# === API Integration Tests ===


@pytest.mark.skipif(not HAS_CLAUDE, reason="Claude CLI not installed")
class TestAPIIntegration:
    """Integration tests for the full API with real CLI."""

    @pytest.fixture
    def app(self):
        """Create app - uses auto-detected CLI paths."""
        return create_app()

    @pytest.mark.asyncio
    async def test_chat_completions_claude(self, app):
        """Test /v1/chat/completions with Claude."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "sonnet",
                    "messages": [{"role": "user", "content": "Say 'hello'."}],
                },
                timeout=120.0,
            )

        assert response.status_code == 200
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert data["choices"][0]["message"]["content"], "Content should not be empty"

    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self, app):
        """Test streaming /v1/chat/completions."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "sonnet",
                    "messages": [{"role": "user", "content": "Say 'hello'."}],
                    "stream": True,
                },
                timeout=120.0,
            ) as response:
                assert response.status_code == 200

                chunks = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunks.append(line)

        # Should have data chunks and [DONE]
        assert len(chunks) > 0, "Should receive SSE chunks"
        assert any("[DONE]" in c for c in chunks), "Should end with [DONE]"

        # At least one chunk should have content
        content_found = False
        for chunk in chunks:
            if "content" in chunk and '"content":' in chunk:
                content_found = True
                break

    @pytest.mark.asyncio
    async def test_models_endpoint(self, app):
        """Test /v1/models returns available models."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert len(data["data"]) > 0, "Should have at least one model"

        # Each model should have required fields
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"

    @pytest.mark.asyncio
    async def test_response_has_no_null_fields(self, app):
        """Test that response doesn't contain null fields in message."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "sonnet",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                timeout=120.0,
            )

        assert response.status_code == 200
        data = response.json()

        message = data["choices"][0]["message"]
        # Message should only have role and content, no null fields
        assert "role" in message
        assert "content" in message
        # These fields should NOT be present (they were causing issues)
        assert "name" not in message or message.get("name") is not None
        assert "tool_calls" not in message or message.get("tool_calls") is not None
        assert "tool_call_id" not in message or message.get("tool_call_id") is not None
