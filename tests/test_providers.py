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


class TestDeserializeStringValues:
    """Tests for _deserialize_string_values in ClaudeCodeProvider."""

    @pytest.fixture
    def provider(self):
        return ClaudeCodeProvider(
            executable_path=Path("/usr/bin/claude"),
            default_timeout=300,
        )

    def test_json_array_string_is_deserialized(self, provider):
        """JSON array encoded as string should be parsed to native array."""
        obj = {"follow_up": '[{"text":"Yes","mode":null}]'}
        result = provider._deserialize_string_values(obj)
        assert isinstance(result["follow_up"], list)
        assert result["follow_up"][0]["text"] == "Yes"

    def test_json_object_string_is_deserialized(self, provider):
        """JSON object encoded as string should be parsed to native dict."""
        obj = {"data": '{"key": "value"}'}
        result = provider._deserialize_string_values(obj)
        assert isinstance(result["data"], dict)
        assert result["data"]["key"] == "value"

    def test_plain_string_is_unchanged(self, provider):
        """Non-JSON strings should pass through unchanged."""
        obj = {"question": "What do you want?", "command": "ls -la"}
        result = provider._deserialize_string_values(obj)
        assert result["question"] == "What do you want?"
        assert result["command"] == "ls -la"

    def test_non_string_values_are_unchanged(self, provider):
        """Non-string values (int, bool, list, dict) should pass through."""
        obj = {"count": 5, "active": True, "items": [1, 2], "meta": {"k": "v"}}
        result = provider._deserialize_string_values(obj)
        assert result == obj

    def test_invalid_json_starting_with_bracket_is_unchanged(self, provider):
        """Strings starting with [ or { but not valid JSON should pass through."""
        obj = {"note": "[not valid json", "other": "{also not valid"}
        result = provider._deserialize_string_values(obj)
        assert result["note"] == "[not valid json"
        assert result["other"] == "{also not valid"

    def test_empty_string_is_unchanged(self, provider):
        """Empty strings should pass through."""
        obj = {"empty": ""}
        result = provider._deserialize_string_values(obj)
        assert result["empty"] == ""

    def test_null_string_becomes_none(self, provider):
        """String "null" should be converted to Python None."""
        obj = {"cwd": "null", "command": "ls -la"}
        result = provider._deserialize_string_values(obj)
        assert result["cwd"] is None
        assert result["command"] == "ls -la"

    def test_real_kilo_code_follow_up(self, provider):
        """Reproduce exact Kilo Code ask_followup_question scenario."""
        obj = {
            "question": "Согласны с планом?",
            "follow_up": '[{"text":"Y — Да","mode":null},{"text":"N — Нет","mode":null}]',
        }
        result = provider._deserialize_string_values(obj)
        assert isinstance(result["follow_up"], list)
        assert len(result["follow_up"]) == 2
        assert result["follow_up"][0]["text"] == "Y — Да"
        assert result["question"] == "Согласны с планом?"


class TestExtractNativeToolCall:
    """Tests for _extract_native_tool_call — the full path that caused 'o.map is not a function'.

    Claude Opus returns native tool_use blocks. When tool arguments contain
    JSON-encoded strings (e.g. follow_up as "[{...}]"), Kilo Code receives
    a string instead of an array and crashes on .map().

    These tests verify that _extract_native_tool_call deserializes such
    strings so the final 'arguments' JSON contains native arrays/objects.
    """

    @pytest.fixture
    def provider(self):
        return ClaudeCodeProvider(
            executable_path=Path("/usr/bin/claude"),
            default_timeout=300,
        )

    def test_ask_followup_question_follow_up_is_array(self, provider):
        """Exact reproduction of the o.map crash: follow_up must be array, not string."""
        block = {
            "type": "tool_use",
            "id": "toolu_01ABC",
            "name": "ask_followup_question",
            "input": {
                "question": "Согласны с включением шагов 2.4 и 2.5?",
                "follow_up": '[{"text":"Y — Да, включить","mode":null},{"text":"N — Нужно обсудить","mode":null}]',
            },
        }

        result = provider._extract_native_tool_call(block)

        assert result is not None
        assert result["function"]["name"] == "ask_followup_question"
        assert result["id"] == "toolu_01ABC"

        # Parse the arguments JSON that Kilo Code will receive
        args = json.loads(result["function"]["arguments"])
        assert args["question"] == "Согласны с включением шагов 2.4 и 2.5?"

        # THIS is the critical assertion — follow_up must be a list, not a string
        assert isinstance(args["follow_up"], list), (
            f"follow_up should be list but got {type(args['follow_up']).__name__}: "
            f"{args['follow_up']!r}"
        )
        assert len(args["follow_up"]) == 2
        assert args["follow_up"][0]["text"] == "Y — Да, включить"

    def test_simple_tool_call_unchanged(self, provider):
        """Tool calls with plain string arguments should not be affected."""
        block = {
            "type": "tool_use",
            "id": "toolu_02DEF",
            "name": "execute_command",
            "input": {"command": "ls -la"},
        }

        result = provider._extract_native_tool_call(block)
        args = json.loads(result["function"]["arguments"])

        assert args["command"] == "ls -la"
        assert isinstance(args["command"], str)

    def test_tool_call_with_nested_json_object_string(self, provider):
        """JSON object encoded as string in arguments should be deserialized."""
        block = {
            "type": "tool_use",
            "id": "toolu_03GHI",
            "name": "some_tool",
            "input": {
                "config": '{"key": "value", "nested": true}',
            },
        }

        result = provider._extract_native_tool_call(block)
        args = json.loads(result["function"]["arguments"])

        assert isinstance(args["config"], dict)
        assert args["config"]["key"] == "value"

    def test_missing_tool_name_returns_none(self, provider):
        """Block without tool name should return None."""
        block = {"type": "tool_use", "input": {"x": 1}}

        assert provider._extract_native_tool_call(block) is None

    def test_empty_input_returns_empty_args(self, provider):
        """Block with empty input should produce empty arguments."""
        block = {
            "type": "tool_use",
            "name": "list_files",
            "input": {},
        }

        result = provider._extract_native_tool_call(block)
        args = json.loads(result["function"]["arguments"])
        assert args == {}

    def test_cwd_null_string_becomes_json_null(self, provider):
        """String "null" for cwd should be converted to JSON null in arguments."""
        block = {
            "type": "tool_use",
            "id": "toolu_04CWD",
            "name": "execute_command",
            "input": {"command": "ls -la", "cwd": "null"},
        }

        result = provider._extract_native_tool_call(block)
        args = json.loads(result["function"]["arguments"])
        assert args["cwd"] is None
        assert args["command"] == "ls -la"

    def test_generates_id_when_missing(self, provider):
        """Should generate a call_ prefixed ID when block has no id."""
        block = {
            "type": "tool_use",
            "name": "test_tool",
            "input": {},
        }

        result = provider._extract_native_tool_call(block)
        assert result["id"].startswith("call_")
