"""Tests for Claude provider."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.openai import ChatMessage


class TestClaudeCodeProvider:
    """Tests for ClaudeCodeProvider."""

    @pytest.fixture
    def provider(self):
        """Create a Claude provider instance."""
        return ClaudeCodeProvider(
            executable_path=Path("/usr/bin/claude"),
            default_timeout=300,
        )

    def test_name_and_models(self, provider):
        assert provider.name == "claude"
        assert "sonnet" in provider.supported_models
        assert "opus" in provider.supported_models
        assert "haiku" in provider.supported_models

    def test_format_messages_simple(self, provider):
        messages = [
            ChatMessage(role="user", content="Hello!"),
        ]
        prompt, system = provider.format_messages_as_prompt(messages)

        assert "Hello!" in prompt
        assert system is None

    def test_format_messages_with_system(self, provider):
        messages = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hello!"),
        ]
        prompt, system = provider.format_messages_as_prompt(messages)

        assert "Hello!" in prompt
        assert system == "Be helpful"

    def test_format_messages_with_history(self, provider):
        messages = [
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        prompt, _ = provider.format_messages_as_prompt(messages)

        assert "Hello!" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt

    def test_build_command_basic(self, provider):
        messages = [ChatMessage(role="user", content="Hello!")]
        cmd = provider.build_command(messages)

        assert cmd[0] == "/usr/bin/claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_build_command_streaming(self, provider):
        messages = [ChatMessage(role="user", content="Hello!")]
        cmd = provider.build_command(messages, stream=True)

        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"

    def test_build_command_with_model(self, provider):
        messages = [ChatMessage(role="user", content="Hello!")]
        cmd = provider.build_command(messages, model="sonnet")

        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "sonnet"

    def test_build_command_with_system_prompt(self, provider):
        messages = [
            ChatMessage(role="system", content="Be brief"),
            ChatMessage(role="user", content="Hello!"),
        ]
        cmd = provider.build_command(messages)

        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "Be brief"

    @pytest.mark.asyncio
    async def test_execute_success(self, provider, mock_subprocess_success):
        messages = [ChatMessage(role="user", content="Hello!")]

        with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess_success):
            result = await provider.execute(messages)

        assert result.content == "Test response from CLI"
        assert result.session_id == "test-session-123"
        assert result.usage["input_tokens"] == 5

    @pytest.mark.asyncio
    async def test_execute_timeout(self, provider):
        messages = [ChatMessage(role="user", content="Hello!")]

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(TimeoutError, match="timed out"):
                await provider.execute(messages, timeout=1)

    @pytest.mark.asyncio
    async def test_execute_cli_error(self, provider):
        messages = [ChatMessage(role="user", content="Hello!")]

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"CLI error occurred"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="CLI failed"):
                await provider.execute(messages)

    @pytest.mark.asyncio
    async def test_execute_stream(self, provider, mock_subprocess_stream):
        messages = [ChatMessage(role="user", content="Hello!")]

        with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess_stream):
            chunks = []
            async for chunk in provider.execute_stream(messages):
                chunks.append(chunk)

        # Should have content chunks and final chunk
        content_chunks = [c for c in chunks if c.content]
        final_chunks = [c for c in chunks if c.is_final]

        assert len(content_chunks) >= 1
        assert len(final_chunks) >= 1
        assert "Hello" in "".join(c.content for c in content_chunks)


class TestMockProvider:
    """Tests using the mock provider from conftest."""

    @pytest.mark.asyncio
    async def test_mock_provider_execute(self, mock_provider, sample_messages):
        result = await mock_provider.execute(sample_messages)

        assert result.content == "Mock response"
        assert result.session_id == "mock-session-123"

    @pytest.mark.asyncio
    async def test_mock_provider_execute_stream(self, mock_provider, sample_messages):
        chunks = []
        async for chunk in mock_provider.execute_stream(sample_messages):
            chunks.append(chunk)

        content = "".join(c.content for c in chunks if c.content)
        assert content == "Hello World!"

    @pytest.mark.asyncio
    async def test_mock_provider_failure(self, mock_provider_factory, sample_messages):
        provider = mock_provider_factory(should_fail=True, fail_message="Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            await provider.execute(sample_messages)

    @pytest.mark.asyncio
    async def test_mock_provider_custom_response(
        self, mock_provider_factory, sample_messages
    ):
        provider = mock_provider_factory(response_content="Custom response!")

        result = await provider.execute(sample_messages)
        assert result.content == "Custom response!"

    @pytest.mark.asyncio
    async def test_mock_provider_custom_stream(
        self, mock_provider_factory, sample_messages
    ):
        provider = mock_provider_factory(stream_chunks=["A", "B", "C"])

        chunks = []
        async for chunk in provider.execute_stream(sample_messages):
            chunks.append(chunk)

        content = "".join(c.content for c in chunks if c.content)
        assert content == "ABC"
