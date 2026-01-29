"""Tests for SSE streaming utilities."""

import json

import pytest

from cli2api.streaming.sse import sse_encode, sse_error


class TestSSEEncode:
    """Tests for sse_encode function."""

    def test_encode_dict(self):
        data = {"message": "Hello", "count": 42}
        result = sse_encode(data)

        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        # Extract JSON and verify
        json_str = result[6:-2]  # Remove "data: " and "\n\n"
        parsed = json.loads(json_str)
        assert parsed == data

    def test_encode_string(self):
        result = sse_encode("Hello World")
        assert result == "data: Hello World\n\n"

    def test_encode_unicode(self):
        data = {"message": "Привет мир"}
        result = sse_encode(data)

        # Should not escape unicode (ensure_ascii=False)
        assert "Привет мир" in result

        json_str = result[6:-2]
        parsed = json.loads(json_str)
        assert parsed["message"] == "Привет мир"

    def test_encode_nested_dict(self):
        data = {
            "choices": [
                {"delta": {"content": "Hello"}, "index": 0}
            ],
            "id": "test-123",
        }
        result = sse_encode(data)

        json_str = result[6:-2]
        parsed = json.loads(json_str)
        assert parsed["choices"][0]["delta"]["content"] == "Hello"

    def test_encode_empty_dict(self):
        result = sse_encode({})
        assert result == "data: {}\n\n"

    def test_encode_number(self):
        result = sse_encode(42)
        assert result == "data: 42\n\n"


class TestSSEError:
    """Tests for sse_error function."""

    def test_basic_error(self):
        result = sse_error("Something went wrong")

        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        json_str = result[6:-2]
        parsed = json.loads(json_str)

        assert "error" in parsed
        assert parsed["error"]["message"] == "Something went wrong"
        assert parsed["error"]["type"] == "server_error"
        assert parsed["error"]["code"] == "server_error"

    def test_error_with_custom_code(self):
        result = sse_error("Not found", code="not_found")

        json_str = result[6:-2]
        parsed = json.loads(json_str)

        assert parsed["error"]["code"] == "not_found"

    def test_error_with_unicode(self):
        result = sse_error("Ошибка сервера")

        assert "Ошибка сервера" in result


class TestSSEStreamFormat:
    """Tests for SSE stream format compliance."""

    def test_openai_compatible_chunk(self):
        """Test that chunks are formatted like OpenAI's streaming API."""
        chunk_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "claude-code",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        result = sse_encode(chunk_data)

        # Verify format
        lines = result.split("\n")
        assert lines[0].startswith("data: ")
        assert lines[1] == ""
        assert lines[2] == ""

        # Verify JSON is valid and contains expected fields
        json_str = lines[0][6:]
        parsed = json.loads(json_str)
        assert parsed["object"] == "chat.completion.chunk"
        assert parsed["choices"][0]["delta"]["content"] == "Hello"

    def test_done_event_format(self):
        """Test the [DONE] event format."""
        done_event = "data: [DONE]\n\n"

        lines = done_event.split("\n")
        assert lines[0] == "data: [DONE]"
        assert lines[1] == ""
