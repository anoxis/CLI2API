"""Streaming parser for tool calls with markers."""

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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

    TOOL_START = "<tool_call>"
    TOOL_END = "</tool_call>"

    def __init__(self):
        self.state = ParserState.TEXT
        self.buffer = ""           # Buffer for tool call content
        self.partial_marker = ""   # Buffer for partial marker detection
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
        return f"call_{uuid.uuid4().hex[:24]}"

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


class LegacyToolParser:
    """
    Fallback parser for responses without markers.

    Detects JSON tool calls using bracket counting.
    Used for backwards compatibility with responses that don't use markers.
    """

    TOOL_CALL_INDICATORS = ['"tool_call"', '"tool_calls"']

    def __init__(self):
        self.buffer = ""
        self.in_json = False
        self.json_start = -1
        self.brace_depth = 0
        self.in_string = False
        self.escape_next = False
        self.tool_calls: list[dict] = []

    def feed(self, chunk: str) -> ParseResult:
        """Process chunk looking for JSON tool calls."""
        result = ParseResult()
        text = self.buffer + chunk
        self.buffer = ""

        i = 0
        text_start = 0

        while i < len(text):
            char = text[i]

            if not self.in_json:
                # Look for start of JSON object
                if char == '{':
                    # Check if this might be a tool call
                    lookahead = text[i:i+100]  # Look ahead for indicators
                    if any(ind in lookahead for ind in self.TOOL_CALL_INDICATORS):
                        # Emit text before JSON
                        if i > text_start:
                            result.text += text[text_start:i]

                        self.in_json = True
                        self.json_start = i
                        self.brace_depth = 1
                        self.in_string = False
                        i += 1
                        continue

                i += 1

            else:
                # Inside JSON - track braces
                if self.escape_next:
                    self.escape_next = False
                    i += 1
                    continue

                if char == '\\' and self.in_string:
                    self.escape_next = True
                    i += 1
                    continue

                if char == '"':
                    self.in_string = not self.in_string

                if not self.in_string:
                    if char == '{':
                        self.brace_depth += 1
                    elif char == '}':
                        self.brace_depth -= 1

                        if self.brace_depth == 0:
                            # Complete JSON object
                            json_str = text[self.json_start:i+1]
                            tool_call = self._parse_json(json_str)
                            if tool_call:
                                result.tool_calls.extend(tool_call)
                                self.tool_calls.extend(tool_call)

                            self.in_json = False
                            text_start = i + 1

                i += 1

        # Handle remaining text
        if self.in_json:
            # Incomplete JSON - buffer it
            self.buffer = text[self.json_start:]
        elif text_start < len(text):
            result.text += text[text_start:]

        return result

    def _parse_json(self, json_str: str) -> list[dict]:
        """Parse JSON and extract tool calls."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        tool_calls = []

        # Single tool_call
        if "tool_call" in data:
            tc_data = data["tool_call"]
            if "name" in tc_data:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tc_data["name"],
                        "arguments": json.dumps(tc_data.get("arguments", {})),
                    },
                })

        # Multiple tool_calls
        if "tool_calls" in data:
            for tc_data in data["tool_calls"]:
                if "name" in tc_data:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": tc_data["name"],
                            "arguments": json.dumps(tc_data.get("arguments", {})),
                        },
                    })

        return tool_calls

    def finalize(self) -> ParseResult:
        """Finalize and parse any remaining buffer."""
        result = ParseResult()
        if self.buffer:
            result.text += self.buffer
            self.buffer = ""
        return result

    def get_all_tool_calls(self) -> list[dict]:
        return self.tool_calls.copy()

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
