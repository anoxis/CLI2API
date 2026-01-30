"""Streaming parser for tool calls with markers."""

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from cli2api.constants import (
    ID_HEX_LENGTH,
    TOOL_CALL_ID_PREFIX,
    TOOL_CALL_START_MARKER,
    TOOL_CALL_END_MARKER,
)


class ParserState(Enum):
    """Parser state machine states."""
    TEXT = "text"                    # Normal text streaming
    MAYBE_MARKER = "maybe_marker"    # Saw '<', checking if marker
    BUFFERING = "buffering"          # Inside <tool_call>...</tool_call>
    MAYBE_END = "maybe_end"          # Saw '</' inside buffer, checking if end


@dataclass
class ParseResult:
    """Result of parsing a chunk."""
    text: str = ""                              # Text to stream immediately
    tool_calls: list[dict] = field(default_factory=list)  # Completed tool calls


class StreamingToolParser:
    """
    Parses tool_calls from stream without blocking text output.

    Uses markers to identify tool calls:
        <tool_call>
        {"name": "tool_name", "arguments": {...}}
        </tool_call>

    Text before/after markers is streamed immediately.
    Content inside markers is buffered and parsed as JSON.

    Example:
        parser = StreamingToolParser()

        # Chunk 1: "Let me read "
        result = parser.feed("Let me read ")
        # result.text = "Let me read ", result.tool_calls = []

        # Chunk 2: "the file.<tool_call>{"
        result = parser.feed("the file.<tool_call>{")
        # result.text = "the file.", result.tool_calls = []

        # Chunk 3: '"name": "read_file"}</tool_call>'
        result = parser.feed('"name": "read_file"}</tool_call>')
        # result.text = "", result.tool_calls = [{"id": "...", "function": {...}}]
    """

    TOOL_START = TOOL_CALL_START_MARKER
    TOOL_END = TOOL_CALL_END_MARKER

    def __init__(self):
        self.state = ParserState.TEXT
        self.buffer = ""
        self.partial_marker = ""
        self.tool_calls: list[dict] = []

    def reset(self):
        """Reset parser state."""
        self.state = ParserState.TEXT
        self.buffer = ""
        self.partial_marker = ""
        self.tool_calls = []

    @staticmethod
    def generate_tool_call_id() -> str:
        """Generate unique tool call ID."""
        return f"{TOOL_CALL_ID_PREFIX}{uuid.uuid4().hex[:ID_HEX_LENGTH]}"

    def _parse_tool_json(self, json_str: str) -> Optional[dict]:
        """Parse tool call JSON and convert to OpenAI format."""
        json_str = json_str.strip()
        if not json_str:
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        # Support both formats:
        # {"name": "...", "arguments": {...}}
        # {"tool_call": {"name": "...", "arguments": {...}}}
        if "tool_call" in data:
            data = data["tool_call"]

        if "name" not in data:
            return None

        return {
            "id": self.generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": data["name"],
                "arguments": json.dumps(data.get("arguments", {})),
            },
        }

    def _check_marker_start(self, text: str, pos: int) -> tuple[bool, int]:
        """
        Check if text starting at pos begins with TOOL_START marker.

        Returns:
            (is_complete_match, chars_consumed)
            - is_complete_match: True if full marker found
            - chars_consumed: number of chars that are part of marker
        """
        remaining = text[pos:]
        marker = self.TOOL_START

        if remaining.startswith(marker):
            return True, len(marker)

        # Check for partial marker at end of text
        for i in range(1, len(marker)):
            if remaining == marker[:i]:
                return False, i

        return False, 0

    def _check_marker_end(self, text: str, pos: int) -> tuple[bool, int]:
        """Check if text starting at pos begins with TOOL_END marker."""
        remaining = text[pos:]
        marker = self.TOOL_END

        if remaining.startswith(marker):
            return True, len(marker)

        # Check for partial marker
        for i in range(1, len(marker)):
            if remaining == marker[:i]:
                return False, i

        return False, 0

    def feed(self, chunk: str) -> ParseResult:
        """
        Process a chunk of streamed text.

        Args:
            chunk: New text chunk from stream

        Returns:
            ParseResult with text to stream and any completed tool calls
        """
        result = ParseResult()

        if not chunk:
            return result

        # Prepend any partial marker from previous chunk
        text = self.partial_marker + chunk
        self.partial_marker = ""

        i = 0
        text_start = 0

        while i < len(text):
            if self.state == ParserState.TEXT:
                # Look for start of tool_call marker
                if text[i] == '<':
                    # Check if this is start of marker before emitting text
                    is_complete, chars = self._check_marker_start(text, i)

                    if is_complete:
                        # Full marker found - emit text before marker, switch to buffering
                        if i > text_start:
                            result.text += text[text_start:i]
                        self.state = ParserState.BUFFERING
                        i += chars
                        text_start = i
                    elif chars > 0:
                        # Partial marker at end - emit text before, save marker for next chunk
                        if i > text_start:
                            result.text += text[text_start:i]
                        self.partial_marker = text[i:]
                        return result
                    else:
                        # Not a marker, just '<' character - continue scanning
                        i += 1
                else:
                    i += 1

            elif self.state == ParserState.BUFFERING:
                # Look for end of tool_call marker
                if text[i] == '<':
                    is_complete, chars = self._check_marker_end(text, i)

                    if is_complete:
                        # End marker found - parse the buffered content
                        tool_call = self._parse_tool_json(self.buffer)
                        if tool_call:
                            self.tool_calls.append(tool_call)
                            result.tool_calls.append(tool_call)

                        # Reset and continue
                        self.buffer = ""
                        self.state = ParserState.TEXT
                        i += chars
                        text_start = i
                    elif chars > 0:
                        # Partial end marker - save for next chunk
                        self.partial_marker = text[i:]
                        return result
                    else:
                        # Not end marker, add to buffer
                        self.buffer += text[i]
                        i += 1
                else:
                    self.buffer += text[i]
                    i += 1

        # Emit remaining text if in TEXT state
        if self.state == ParserState.TEXT and text_start < len(text):
            result.text += text[text_start:]

        return result

    def finalize(self) -> ParseResult:
        """
        Finalize parsing - handle any remaining buffered content.

        Call this when the stream ends to handle edge cases.

        Returns:
            ParseResult with any remaining text or tool calls
        """
        result = ParseResult()

        # If we have a partial marker, it wasn't a marker - emit as text
        if self.partial_marker:
            result.text += self.partial_marker
            self.partial_marker = ""

        # If we're still buffering, the tool call wasn't closed properly
        # Try to parse it anyway (might be valid JSON without closing tag)
        if self.state == ParserState.BUFFERING and self.buffer:
            tool_call = self._parse_tool_json(self.buffer)
            if tool_call:
                self.tool_calls.append(tool_call)
                result.tool_calls.append(tool_call)
            else:
                # Invalid JSON - emit as text (shouldn't happen normally)
                result.text += self.TOOL_START + self.buffer
            self.buffer = ""

        self.state = ParserState.TEXT
        return result

    def get_all_tool_calls(self) -> list[dict]:
        """Get all tool calls parsed so far."""
        return self.tool_calls.copy()

    def has_tool_calls(self) -> bool:
        """Check if any tool calls have been parsed."""
        return len(self.tool_calls) > 0
