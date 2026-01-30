"""Tests for StreamingToolParser."""

import json
import pytest

from cli2api.streaming.tool_parser import StreamingToolParser, ParseResult


class TestStreamingToolParser:
    """Tests for marker-based tool call parsing."""

    def test_text_only_no_markers(self):
        """Text without markers should be passed through."""
        parser = StreamingToolParser()

        result = parser.feed("Hello, I will help you with that.")
        assert result.text == "Hello, I will help you with that."
        assert result.tool_calls == []

    def test_single_tool_call_complete(self):
        """Complete tool call in single chunk."""
        parser = StreamingToolParser()

        text = '<tool_call>{"name": "read_file", "arguments": {"path": "main.py"}}</tool_call>'
        result = parser.feed(text)

        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "read_file"

        args = json.loads(result.tool_calls[0]["function"]["arguments"])
        assert args["path"] == "main.py"

    def test_text_before_tool_call(self):
        """Text before tool call should be streamed."""
        parser = StreamingToolParser()

        text = 'Let me read the file.<tool_call>{"name": "read_file", "arguments": {}}</tool_call>'
        result = parser.feed(text)

        assert result.text == "Let me read the file."
        assert len(result.tool_calls) == 1

    def test_text_after_tool_call(self):
        """Text after tool call should be streamed."""
        parser = StreamingToolParser()

        text = '<tool_call>{"name": "read_file", "arguments": {}}</tool_call>Done!'
        result = parser.feed(text)

        assert result.text == "Done!"
        assert len(result.tool_calls) == 1

    def test_text_around_tool_call(self):
        """Text before and after tool call."""
        parser = StreamingToolParser()

        text = 'Before<tool_call>{"name": "test", "arguments": {}}</tool_call>After'
        result = parser.feed(text)

        assert result.text == "BeforeAfter"
        assert len(result.tool_calls) == 1

    def test_multiple_tool_calls(self):
        """Multiple tool calls in sequence."""
        parser = StreamingToolParser()

        text = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
            '<tool_call>{"name": "read_file", "arguments": {"path": "b.py"}}</tool_call>'
        )
        result = parser.feed(text)

        assert result.text == ""
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["function"]["name"] == "read_file"
        assert result.tool_calls[1]["function"]["name"] == "read_file"

    def test_fragmented_marker_start(self):
        """Marker split across chunks - start marker."""
        parser = StreamingToolParser()

        # First chunk ends with partial marker
        result1 = parser.feed("Hello <tool")
        assert result1.text == "Hello "
        assert result1.tool_calls == []

        # Second chunk completes marker
        result2 = parser.feed('_call>{"name": "test", "arguments": {}}</tool_call>')
        assert result2.text == ""
        assert len(result2.tool_calls) == 1

    def test_fragmented_marker_end(self):
        """Marker split across chunks - end marker."""
        parser = StreamingToolParser()

        # First chunk has content but partial end marker
        result1 = parser.feed('<tool_call>{"name": "test", "arguments": {}}</tool')
        assert result1.text == ""
        assert result1.tool_calls == []  # Not complete yet

        # Second chunk completes marker
        result2 = parser.feed("_call>")
        assert result2.text == ""
        assert len(result2.tool_calls) == 1

    def test_fragmented_json_content(self):
        """JSON content split across chunks."""
        parser = StreamingToolParser()

        result1 = parser.feed('<tool_call>{"name": "read_')
        assert result1.text == ""
        assert result1.tool_calls == []

        result2 = parser.feed('file", "arguments": {"path": "test.py"}}</tool_call>')
        assert result2.text == ""
        assert len(result2.tool_calls) == 1

        args = json.loads(result2.tool_calls[0]["function"]["arguments"])
        assert args["path"] == "test.py"

    def test_angle_bracket_not_marker(self):
        """Angle bracket that's not a marker should be passed through."""
        parser = StreamingToolParser()

        result = parser.feed("Use <div> for HTML")
        assert result.text == "Use <div> for HTML"
        assert result.tool_calls == []

    def test_partial_marker_not_completed(self):
        """Partial marker at end that turns out not to be a marker."""
        parser = StreamingToolParser()

        # Looks like start of marker
        result1 = parser.feed("Check <to")
        assert result1.text == "Check "  # Buffering potential marker

        # But it's not
        result2 = parser.feed("day's weather")
        assert "today's weather" in result2.text or "to" in result1.text

    def test_finalize_incomplete_marker(self):
        """Finalize should handle incomplete tool call."""
        parser = StreamingToolParser()

        parser.feed('<tool_call>{"name": "test"')  # Incomplete

        result = parser.finalize()
        # Should try to parse incomplete JSON or emit as text
        assert parser.state.value == "text"  # Reset to text state

    def test_finalize_with_partial_marker(self):
        """Finalize should emit buffered partial marker."""
        parser = StreamingToolParser()

        parser.feed("Hello <tool")  # Partial marker at end

        result = parser.finalize()
        # Partial marker should be emitted as text
        assert "<tool" in result.text or parser.partial_marker == ""

    def test_nested_format_support(self):
        """Support {"tool_call": {...}} format inside markers."""
        parser = StreamingToolParser()

        text = '<tool_call>{"tool_call": {"name": "test", "arguments": {"x": 1}}}</tool_call>'
        result = parser.feed(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "test"

    def test_get_all_tool_calls(self):
        """get_all_tool_calls returns accumulated calls."""
        parser = StreamingToolParser()

        parser.feed('<tool_call>{"name": "a", "arguments": {}}</tool_call>')
        parser.feed('<tool_call>{"name": "b", "arguments": {}}</tool_call>')

        all_calls = parser.get_all_tool_calls()
        assert len(all_calls) == 2

    def test_has_tool_calls(self):
        """has_tool_calls returns correct state."""
        parser = StreamingToolParser()

        assert parser.has_tool_calls() is False

        parser.feed('<tool_call>{"name": "test", "arguments": {}}</tool_call>')

        assert parser.has_tool_calls() is True

    def test_reset(self):
        """Reset clears all state."""
        parser = StreamingToolParser()

        parser.feed('<tool_call>{"name": "test", "arguments": {}}</tool_call>')
        assert parser.has_tool_calls() is True

        parser.reset()

        assert parser.has_tool_calls() is False
        assert parser.buffer == ""
        assert parser.partial_marker == ""

    def test_complex_arguments(self):
        """Tool call with complex nested arguments."""
        parser = StreamingToolParser()

        args = {
            "path": "/path/to/file.py",
            "content": "def hello():\n    print('world')",
            "options": {"create": True, "backup": False},
        }
        text = f'<tool_call>{{"name": "write_file", "arguments": {json.dumps(args)}}}</tool_call>'

        result = parser.feed(text)

        assert len(result.tool_calls) == 1
        parsed_args = json.loads(result.tool_calls[0]["function"]["arguments"])
        assert parsed_args["path"] == args["path"]
        assert parsed_args["content"] == args["content"]
        assert parsed_args["options"] == args["options"]

    def test_tool_call_id_generation(self):
        """Each tool call gets unique ID."""
        parser = StreamingToolParser()

        parser.feed('<tool_call>{"name": "a", "arguments": {}}</tool_call>')
        parser.feed('<tool_call>{"name": "b", "arguments": {}}</tool_call>')

        calls = parser.get_all_tool_calls()
        ids = [c["id"] for c in calls]

        assert len(set(ids)) == 2  # All unique
        assert all(id.startswith("call_") for id in ids)

    def test_whitespace_in_markers(self):
        """Whitespace inside markers should be handled."""
        parser = StreamingToolParser()

        text = '''<tool_call>
        {"name": "test", "arguments": {}}
        </tool_call>'''

        result = parser.feed(text)

        assert len(result.tool_calls) == 1

    def test_real_world_streaming_simulation(self):
        """Simulate real streaming with small chunks."""
        parser = StreamingToolParser()

        chunks = [
            "I'll read ",
            "the file for you.",
            "\n<tool",
            "_call>",
            '{"name"',
            ': "read_file",',
            ' "arguments": ',
            '{"path": "main.py"',
            "}}",
            "</tool_",
            "call>",
            "\nDone!",
        ]

        all_text = ""
        all_tool_calls = []

        for chunk in chunks:
            result = parser.feed(chunk)
            all_text += result.text
            all_tool_calls.extend(result.tool_calls)

        final = parser.finalize()
        all_text += final.text

        assert "I'll read the file for you." in all_text
        assert "Done!" in all_text
        assert "<tool_call>" not in all_text
        assert "</tool_call>" not in all_text
        assert len(all_tool_calls) == 1
        assert all_tool_calls[0]["function"]["name"] == "read_file"


class TestParseResultDataclass:
    """Tests for ParseResult dataclass."""

    def test_default_values(self):
        """ParseResult has correct defaults."""
        result = ParseResult()
        assert result.text == ""
        assert result.tool_calls == []

    def test_with_values(self):
        """ParseResult accepts values."""
        result = ParseResult(text="hello", tool_calls=[{"id": "test"}])
        assert result.text == "hello"
        assert result.tool_calls == [{"id": "test"}]
