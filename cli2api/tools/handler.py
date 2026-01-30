"""Tool calling handler for prompt-based tool use."""

import json
import re
import uuid
from typing import Any, Optional


class ToolHandler:
    """Handler for prompt-based tool calling.

    Since Claude Code CLI doesn't natively support OpenAI-style
    tool calling, we use a prompt-based approach:
    1. Include tool definitions in the system prompt
    2. Instruct the model to respond with JSON when calling a tool
    3. Parse the response to extract tool calls
    """

    # Pattern to match <tool_call>...</tool_call> markers
    MARKER_TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>\s*(.*?)\s*</tool_call>',
        re.DOTALL,
    )

    # Pattern to match JSON tool calls in code blocks (single or multiple)
    TOOL_CALL_PATTERN = re.compile(
        r'```(?:json)?\s*\n?\s*(\{[^`]*?"tool_calls?"[^`]*?\})\s*\n?\s*```',
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern to match Kilo Code style tool calls:
    # [Tool Call: name({...})]
    # [Tool Call: name({...}) id=call_xxx]
    KILO_TOOL_CALL_PATTERN = re.compile(
        r'\[Tool Call:\s*(\w+)\s*\((\{.*?\})\)(?:\s*id=\S+)?\s*\]',
        re.DOTALL,
    )

    @staticmethod
    def format_tools_prompt(tools: list[dict]) -> str:
        """Format tools as a system prompt addition.

        Args:
            tools: List of tool definitions in OpenAI format.

        Returns:
            Prompt text describing available tools.
        """
        if not tools:
            return ""

        # Sort tools by name for consistent cache hits
        tools = sorted(
            tools,
            key=lambda t: t.get("function", {}).get("name", "")
        )

        lines = [
            "You have access to the following tools:\n",
        ]

        for tool in tools:
            if tool.get("type") != "function":
                continue

            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "No description")
            params = func.get("parameters", {})

            lines.append(f"## {name}")
            lines.append(f"{description}\n")

            # Format parameters
            properties = params.get("properties", {})
            required = params.get("required", [])

            if properties:
                lines.append("Parameters:")
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    is_required = param_name in required
                    req_marker = " (required)" if is_required else " (optional)"
                    lines.append(f"- {param_name} ({param_type}{req_marker}): {param_desc}")
                lines.append("")

        lines.append("\n---")
        lines.append("RESPONSE FORMAT:")
        lines.append("")
        lines.append("When calling tools, wrap EACH tool call in <tool_call> tags.")
        lines.append("You may include explanatory text before or after the tags.")
        lines.append("")
        lines.append("For a SINGLE tool call:")
        lines.append("<tool_call>")
        lines.append('{"name": "TOOL_NAME", "arguments": {...}}')
        lines.append("</tool_call>")
        lines.append("")
        lines.append("For MULTIPLE tool calls, use SEPARATE tags for each:")
        lines.append("<tool_call>")
        lines.append('{"name": "read_file", "arguments": {"path": "file1.py"}}')
        lines.append("</tool_call>")
        lines.append("<tool_call>")
        lines.append('{"name": "read_file", "arguments": {"path": "file2.py"}}')
        lines.append("</tool_call>")
        lines.append("")
        lines.append("CRITICAL RULES:")
        lines.append("- ALWAYS wrap tool calls in <tool_call>...</tool_call> tags")
        lines.append("- Each tool call must be in its own tag pair")
        lines.append("- When reading multiple files, include multiple <tool_call> tags")
        lines.append("- You CAN include text explanation before/after tags")
        lines.append("- Do NOT execute tools yourself - just output the tags")
        lines.append("- If user wants to read a file -> use read_file tool")
        lines.append("- If user wants to list files -> use list_files tool")
        lines.append("- ALWAYS include ALL required parameters for each tool")
        lines.append("")
        lines.append("ATTEMPT_COMPLETION USAGE:")
        lines.append("When using attempt_completion, ALWAYS include the 'result' parameter:")
        lines.append("<tool_call>")
        lines.append('{"name": "attempt_completion", "arguments": {"result": "Your response text here"}}')
        lines.append("</tool_call>")
        lines.append("- The 'result' parameter is REQUIRED and must contain your final response")
        lines.append("- Use attempt_completion ONLY after you have gathered all needed information")

        return "\n".join(lines)

    @staticmethod
    def generate_tool_call_id() -> str:
        """Generate a unique ID for a tool call.

        Returns:
            Tool call ID in format "call_<hex>".
        """
        return f"call_{uuid.uuid4().hex[:24]}"

    @classmethod
    def _extract_json_objects(cls, content: str) -> list[tuple[int, int, str]]:
        """Extract JSON objects from content using bracket counting.

        Returns list of (start_pos, end_pos, json_string) tuples.
        """
        results = []
        i = 0
        while i < len(content):
            if content[i] == '{':
                # Found potential JSON start
                depth = 0
                start = i
                in_string = False
                escape_next = False

                while i < len(content):
                    char = content[i]

                    if escape_next:
                        escape_next = False
                        i += 1
                        continue

                    if char == '\\' and in_string:
                        escape_next = True
                        i += 1
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string

                    if not in_string:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                # Found complete JSON object
                                json_str = content[start:i + 1]
                                results.append((start, i + 1, json_str))
                                break
                    i += 1
            else:
                i += 1
        return results

    @classmethod
    def parse_tool_calls(
        cls, content: str
    ) -> tuple[Optional[str], Optional[list[dict]]]:
        """Parse response content for tool calls.

        Args:
            content: Model response content.

        Returns:
            Tuple of (remaining_content, tool_calls).
            - remaining_content: Text without tool call JSON, or None if only tool call
            - tool_calls: List of parsed tool calls in OpenAI format, or None
        """
        if not content:
            return content, None

        tool_calls = []
        positions_to_remove = []

        # First try marker-based format: <tool_call>...</tool_call>
        for match in cls.MARKER_TOOL_CALL_PATTERN.finditer(content):
            json_str = match.group(1).strip()
            try:
                data = json.loads(json_str)

                # Support both {"name": ...} and {"tool_call": {"name": ...}}
                if "tool_call" in data:
                    data = data["tool_call"]

                if "name" in data:
                    tool_call = {
                        "id": cls.generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(data.get("arguments", {})),
                        },
                    }
                    tool_calls.append(tool_call)
                    positions_to_remove.append((match.start(), match.end()))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        # If we found marker-based tool calls, return early
        if tool_calls:
            remaining = content
            for start, end in sorted(positions_to_remove, reverse=True):
                remaining = remaining[:start] + remaining[end:]
            remaining = remaining.strip()
            return remaining if remaining else None, tool_calls

        # Fallback: try direct JSON parse (raw JSON without code block)
        stripped = content.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                data = json.loads(stripped)

                # Handle single tool_call
                tool_call_data = data.get("tool_call", {})
                if tool_call_data and "name" in tool_call_data:
                    tool_call = {
                        "id": cls.generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": tool_call_data["name"],
                            "arguments": json.dumps(
                                tool_call_data.get("arguments", {})
                            ),
                        },
                    }
                    tool_calls.append(tool_call)
                    return None, tool_calls  # Only tool call, no remaining content

                # Handle multiple tool_calls (parallel)
                tool_calls_data = data.get("tool_calls", [])
                if tool_calls_data and isinstance(tool_calls_data, list):
                    for tc_data in tool_calls_data:
                        if tc_data and "name" in tc_data:
                            tool_call = {
                                "id": cls.generate_tool_call_id(),
                                "type": "function",
                                "function": {
                                    "name": tc_data["name"],
                                    "arguments": json.dumps(
                                        tc_data.get("arguments", {})
                                    ),
                                },
                            }
                            tool_calls.append(tool_call)
                    if tool_calls:
                        return None, tool_calls  # Only tool calls, no remaining content
            except (json.JSONDecodeError, TypeError, KeyError):
                pass  # Not valid JSON, continue with other methods

        # Try code blocks
        for match in cls.TOOL_CALL_PATTERN.finditer(content):
            json_str = match.group(1)
            try:
                data = json.loads(json_str)

                # Handle single tool_call
                tool_call_data = data.get("tool_call", {})
                if tool_call_data and "name" in tool_call_data:
                    tool_call = {
                        "id": cls.generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": tool_call_data["name"],
                            "arguments": json.dumps(
                                tool_call_data.get("arguments", {})
                            ),
                        },
                    }
                    tool_calls.append(tool_call)
                    positions_to_remove.append((match.start(), match.end()))

                # Handle multiple tool_calls (parallel)
                tool_calls_data = data.get("tool_calls", [])
                if tool_calls_data and isinstance(tool_calls_data, list):
                    for tc_data in tool_calls_data:
                        if tc_data and "name" in tc_data:
                            tool_call = {
                                "id": cls.generate_tool_call_id(),
                                "type": "function",
                                "function": {
                                    "name": tc_data["name"],
                                    "arguments": json.dumps(
                                        tc_data.get("arguments", {})
                                    ),
                                },
                            }
                            tool_calls.append(tool_call)
                    if tool_calls_data:
                        positions_to_remove.append((match.start(), match.end()))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        # Try Kilo Code style: [Tool Call: name({...})]
        if not tool_calls:
            for match in cls.KILO_TOOL_CALL_PATTERN.finditer(content):
                tool_name = match.group(1)
                args_str = match.group(2)
                try:
                    arguments = json.loads(args_str)
                    tool_call = {
                        "id": cls.generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                    tool_calls.append(tool_call)
                    positions_to_remove.append((match.start(), match.end()))
                except (json.JSONDecodeError, TypeError):
                    continue

        # If no code block matches, try inline JSON extraction
        if not tool_calls:
            for start, end, json_str in cls._extract_json_objects(content):
                # Check if this looks like a tool call
                if '"tool_call"' not in json_str and '"tool_calls"' not in json_str:
                    continue

                try:
                    data = json.loads(json_str)

                    # Handle single tool_call
                    tool_call_data = data.get("tool_call", {})
                    if tool_call_data and "name" in tool_call_data:
                        tool_call = {
                            "id": cls.generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_call_data["name"],
                                "arguments": json.dumps(
                                    tool_call_data.get("arguments", {})
                                ),
                            },
                        }
                        tool_calls.append(tool_call)
                        positions_to_remove.append((start, end))

                    # Handle multiple tool_calls (parallel)
                    tool_calls_data = data.get("tool_calls", [])
                    if tool_calls_data and isinstance(tool_calls_data, list):
                        for tc_data in tool_calls_data:
                            if tc_data and "name" in tc_data:
                                tool_call = {
                                    "id": cls.generate_tool_call_id(),
                                    "type": "function",
                                    "function": {
                                        "name": tc_data["name"],
                                        "arguments": json.dumps(
                                            tc_data.get("arguments", {})
                                        ),
                                    },
                                }
                                tool_calls.append(tool_call)
                        if tool_calls_data:
                            positions_to_remove.append((start, end))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

        if not tool_calls:
            return content, None

        # Remove tool call JSON from content (in reverse order to preserve positions)
        remaining = content
        for start, end in sorted(positions_to_remove, reverse=True):
            remaining = remaining[:start] + remaining[end:]

        remaining = remaining.strip()

        # If only whitespace remains, return None for content
        if not remaining:
            remaining = None

        return remaining, tool_calls

    @classmethod
    def _sanitize_system_prompt(cls, content: str) -> str:
        """Remove conflicting tool instructions from system prompt.

        Kilo Code and other tools may include instructions like
        "Use the provider-native tool-calling mechanism" which conflicts
        with our prompt-based approach.
        """
        # Phrases that conflict with prompt-based tool calling
        conflicting_phrases = [
            "Use the provider-native tool-calling mechanism",
            "provider-native tool-calling",
            "Do not include XML markup or examples",
        ]

        for phrase in conflicting_phrases:
            content = content.replace(phrase, "")

        return content

    @classmethod
    def inject_tools_into_messages(
        cls,
        messages: list[Any],
        tools: Optional[list[dict]],
    ) -> list[Any]:
        """Inject tool definitions into messages.

        Strategy: Add tools to BOTH system prompt AND the last user message.
        This ensures the instructions are fresh in the model's context.

        Args:
            messages: Original chat messages.
            tools: Tool definitions to inject.

        Returns:
            Modified messages with tools injected.
        """
        if not tools:
            return messages

        tool_prompt = cls.format_tools_prompt(tools)
        if not tool_prompt:
            return messages

        # Make a copy to avoid modifying original
        messages = list(messages)
        from cli2api.schemas.openai import ChatMessage

        # 1. Sanitize system prompt if exists
        for i, msg in enumerate(messages):
            is_system = False
            if hasattr(msg, "role") and msg.role == "system":
                is_system = True
            elif isinstance(msg, dict) and msg.get("role") == "system":
                is_system = True

            if is_system:
                if hasattr(msg, "content"):
                    existing_content = msg.content or ""
                    sanitized = cls._sanitize_system_prompt(existing_content)
                    messages[i] = ChatMessage(role="system", content=sanitized)
                elif isinstance(msg, dict):
                    msg["content"] = cls._sanitize_system_prompt(msg.get("content", ""))

        # 2. Find last user message and append tool instructions there
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if hasattr(msg, "role") and msg.role == "user":
                last_user_idx = i
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            msg = messages[last_user_idx]
            if hasattr(msg, "content"):
                existing = msg.content or ""
                new_content = f"{existing}\n\n[RESPOND IN JSON FORMAT]\n{tool_prompt}"
                messages[last_user_idx] = ChatMessage(role="user", content=new_content)
            elif isinstance(msg, dict):
                existing = msg.get("content", "")
                msg["content"] = f"{existing}\n\n[RESPOND IN JSON FORMAT]\n{tool_prompt}"
        else:
            # No user message found, insert as system message
            system_msg = ChatMessage(role="system", content=tool_prompt)
            messages.insert(0, system_msg)

        return messages
