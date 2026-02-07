"""Tests for tool call argument validation."""

import json

import pytest

from cli2api.tools.validator import (
    build_required_params_index,
    filter_valid_tool_calls,
    validate_tool_call,
)


# === Fixtures ===

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "encoding": {"type": "string", "description": "Encoding"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attempt_completion",
            "description": "Complete the task",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Result text"},
                },
                "required": ["result"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List directory contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": [],
            },
        },
    },
]


def _make_tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": "call_test123",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _make_tool_call_raw(name: str, arguments_str: str) -> dict:
    return {
        "id": "call_test123",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_str,
        },
    }


# === Tests for build_required_params_index ===


class TestBuildRequiredParamsIndex:
    def test_builds_index(self):
        index = build_required_params_index(SAMPLE_TOOLS)
        assert index["execute_command"] == ["command"]
        assert index["read_file"] == ["path"]
        assert index["attempt_completion"] == ["result"]
        assert index["list_files"] == []

    def test_empty_tools(self):
        assert build_required_params_index([]) == {}

    def test_skips_non_function_tools(self):
        tools = [{"type": "code_interpreter"}]
        assert build_required_params_index(tools) == {}


# === Tests for validate_tool_call ===


class TestValidateToolCall:
    def setup_method(self):
        self.index = build_required_params_index(SAMPLE_TOOLS)

    def test_valid_tool_call_passes(self):
        tc = _make_tool_call("execute_command", {"command": "ls -la"})
        assert validate_tool_call(tc, self.index) is True

    def test_valid_with_extra_params(self):
        tc = _make_tool_call("read_file", {"path": "test.py", "encoding": "utf-8"})
        assert validate_tool_call(tc, self.index) is True

    def test_missing_required_param_fails(self):
        tc = _make_tool_call("execute_command", {})
        assert validate_tool_call(tc, self.index) is False

    def test_empty_arguments_with_required_params_fails(self):
        tc = _make_tool_call_raw("execute_command", "{}")
        assert validate_tool_call(tc, self.index) is False

    def test_no_required_params_always_passes(self):
        tc = _make_tool_call("list_files", {})
        assert validate_tool_call(tc, self.index) is True

    def test_unknown_tool_passes(self):
        tc = _make_tool_call("unknown_tool", {})
        assert validate_tool_call(tc, self.index) is True

    def test_unparseable_arguments_fails(self):
        tc = _make_tool_call_raw("execute_command", "not-json")
        assert validate_tool_call(tc, self.index) is False

    def test_non_dict_arguments_fails(self):
        tc = _make_tool_call_raw("execute_command", '"just a string"')
        assert validate_tool_call(tc, self.index) is False


# === Tests for filter_valid_tool_calls ===


class TestFilterValidToolCalls:
    def test_all_valid_passes_through(self):
        tool_calls = [
            _make_tool_call("execute_command", {"command": "ls"}),
            _make_tool_call("read_file", {"path": "test.py"}),
        ]
        result = filter_valid_tool_calls(tool_calls, SAMPLE_TOOLS)
        assert len(result) == 2

    def test_filters_invalid_keeps_valid(self):
        tool_calls = [
            _make_tool_call("execute_command", {"command": "ls"}),  # valid
            _make_tool_call("read_file", {}),  # invalid — missing path
        ]
        result = filter_valid_tool_calls(tool_calls, SAMPLE_TOOLS)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "execute_command"

    def test_all_invalid_returns_none(self):
        tool_calls = [
            _make_tool_call("execute_command", {}),
            _make_tool_call("read_file", {}),
        ]
        result = filter_valid_tool_calls(tool_calls, SAMPLE_TOOLS)
        assert result is None

    def test_none_tool_calls_returns_none(self):
        assert filter_valid_tool_calls(None, SAMPLE_TOOLS) is None

    def test_empty_tool_calls_returns_empty(self):
        assert filter_valid_tool_calls([], SAMPLE_TOOLS) == []

    def test_none_tools_returns_tool_calls_unchanged(self):
        tool_calls = [_make_tool_call("execute_command", {})]
        result = filter_valid_tool_calls(tool_calls, None)
        assert result == tool_calls

    def test_no_required_params_tool_always_passes(self):
        tool_calls = [_make_tool_call("list_files", {})]
        result = filter_valid_tool_calls(tool_calls, SAMPLE_TOOLS)
        assert len(result) == 1
