"""Claude Code CLI provider."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from cli2api.constants import ID_HEX_LENGTH, TOOL_CALL_ID_PREFIX
from cli2api.schemas.internal import ProviderChunk, ProviderResult
from cli2api.schemas.openai import ChatMessage
from cli2api.streaming.tool_parser import StreamingToolParser
from cli2api.tools.handler import ToolHandler
from cli2api.tools.validator import filter_valid_tool_calls
from cli2api.utils.logging import get_logger

logger = get_logger(__name__)


# ==================== Message Formatting Utilities ====================


def _extract_tool_call_info(tc: Any) -> tuple[str, str, str]:
    """Extract tool call information from dict or object."""
    if isinstance(tc, dict):
        name = tc.get("function", {}).get("name", "unknown")
        args = tc.get("function", {}).get("arguments", "{}")
        tc_id = tc.get("id", "")
    else:
        name = tc.function.name if hasattr(tc, "function") else "unknown"
        args = tc.function.arguments if hasattr(tc, "function") else "{}"
        tc_id = tc.id if hasattr(tc, "id") else ""
    return name, args, tc_id


def _format_tool_calls_text(tool_calls: list[Any]) -> str:
    """Format tool calls as text for prompt injection."""
    lines = []
    for tc in tool_calls:
        name, args, tc_id = _extract_tool_call_info(tc)
        lines.append(f"[Tool Call: {name}({args}) id={tc_id}]")
    return "\n".join(lines)


def _format_assistant_message(content: Optional[str], tool_calls: Optional[list[Any]] = None) -> str:
    """Format assistant message with optional tool calls."""
    result = content or ""
    if tool_calls:
        tool_calls_text = _format_tool_calls_text(tool_calls)
        if tool_calls_text:
            result = f"{result}\n{tool_calls_text}" if result else tool_calls_text
    return f"[Previous Assistant Response]\n{result}"


def _format_tool_result(tool_call_id: Optional[str], content: Optional[str]) -> str:
    """Format tool result message."""
    tc_id = tool_call_id or "unknown"
    result = content or ""
    return f"[Tool Result for {tc_id}]\n{result}"


# ==================== Claude Code Provider ====================


class ClaudeCodeProvider:
    """Provider for Claude Code CLI.

    Uses `claude -p` for non-interactive execution with JSON output.
    Supports both streaming (--output-format stream-json) and
    non-streaming (--output-format json) modes.

    Accepts any model name and passes it to CLI via --model flag.
    """

    name = "claude"

    def __init__(
        self,
        executable_path: Path,
        default_timeout: int = 300,
        models: list[str] | None = None,
    ):
        """Initialize the provider.

        Args:
            executable_path: Path to the Claude CLI executable.
            default_timeout: Default timeout for CLI execution in seconds.
            models: List of supported model names.
        """
        self.executable_path = executable_path
        self.default_timeout = default_timeout
        self.supported_models = models or [
            "sonnet",
            "opus",
            "haiku",
        ]

    def format_messages_as_prompt(
        self, messages: list[ChatMessage]
    ) -> tuple[str, Optional[str]]:
        """Convert messages to prompt and optional system prompt.

        Returns:
            Tuple of (user_prompt, system_prompt).
        """
        parts = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                parts.append(msg.content)
            elif msg.role == "assistant":
                parts.append(_format_assistant_message(msg.content, msg.tool_calls))
            elif msg.role == "tool":
                parts.append(_format_tool_result(msg.tool_call_id, msg.content))

        prompt = "\n\n".join(parts)
        return prompt, system_prompt

    # Windows CreateProcess limit is 32767 chars for the entire command line.
    # We use a conservative threshold to leave room for quoting/escaping.
    MAX_CMD_LINE_LENGTH = 32000

    def build_command(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> tuple[list[str], Optional[str]]:
        """Build claude CLI command.

        Returns:
            Tuple of (cmd_args, stdin_prompt_or_none).
            For short prompts, prompt is included as a CLI argument (stdin_prompt=None).
            For long prompts, it's passed via stdin to avoid Windows limits.
        """
        prompt, system_prompt = self.format_messages_as_prompt(messages)

        # Build cmd without prompt first, then check total length
        cmd = [str(self.executable_path), "-p"]

        # Output format
        if stream:
            cmd.extend([
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
            ])
        else:
            cmd.extend(["--output-format", "json"])

        # Model selection (except "claude" which means use CLI default)
        if model and model != "claude":
            cmd.extend(["--model", model])

        # System prompt
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # Check total command line length with prompt included
        # Each arg gets a space separator + potential quoting
        cmd_length = sum(len(a) for a in cmd) + len(cmd)  # args + spaces
        total_with_prompt = cmd_length + len(prompt) + 3  # +3 for quotes and space

        if total_with_prompt > self.MAX_CMD_LINE_LENGTH:
            # Too long for Windows — pass prompt via stdin
            return cmd, prompt
        else:
            # Safe to pass as CLI argument
            cmd.insert(2, prompt)  # Insert after "-p"
            return cmd, None

    # ==================== Subprocess Utilities ====================

    async def _run_subprocess(
        self,
        cmd: list[str],
        timeout: Optional[int] = None,
        stdin_data: Optional[bytes] = None,
    ) -> tuple[bytes, bytes]:
        """Run subprocess with timeout handling.

        Args:
            cmd: Command and arguments.
            timeout: Timeout in seconds.
            stdin_data: Data to send to process stdin (e.g. the prompt).
        """
        timeout = timeout or self.default_timeout

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"CLI execution timed out after {timeout}s")

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Claude CLI failed: {error_msg}")

        return stdout, stderr

    async def _create_stream_process(
        self,
        cmd: list[str],
        stdin_data: Optional[bytes] = None,
    ) -> asyncio.subprocess.Process:
        """Create subprocess for streaming.

        Args:
            cmd: Command and arguments.
            stdin_data: Data to send to process stdin before streaming output.
        """
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if stdin_data and proc.stdin:
            proc.stdin.write(stdin_data)
            await proc.stdin.drain()
            proc.stdin.close()
        logger.info(f"Claude process started: PID={proc.pid}")
        return proc

    async def _cleanup_process(self, proc: asyncio.subprocess.Process) -> None:
        """Clean up subprocess if still running."""
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

    def _parse_json_line(self, line: str) -> Optional[dict]:
        """Parse a JSON line, returning None if invalid.

        Logs parsing errors at debug level for diagnostics.
        """
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            # Log at debug level - invalid JSON is expected for plain text lines
            logger.debug(f"JSON parse skipped (not JSON): {line[:100]}... Error: {e.msg}")
            return None

    def _handle_tool_calls_in_content(
        self,
        content: str,
        tools: Optional[list[dict]],
    ) -> tuple[str, Optional[list[dict]]]:
        """Parse content for tool calls if tools are provided."""
        if tools and content:
            remaining, tool_calls = ToolHandler.parse_tool_calls(content)
            tool_calls = filter_valid_tool_calls(tool_calls, tools)
            return remaining, tool_calls
        return content, None

    def _create_final_chunk(
        self,
        tool_calls: Optional[list[dict]] = None,
        usage: Optional[dict] = None,
    ) -> ProviderChunk:
        """Create a final chunk for stream completion."""
        return ProviderChunk(
            content="",
            is_final=True,
            tool_calls=tool_calls,
            usage=usage,
        )

    @staticmethod
    def _deserialize_string_values(obj: dict) -> dict:
        """Deserialize JSON-encoded string values in tool arguments.

        Claude sometimes returns structured data (arrays, objects) as
        JSON-encoded strings. Kilo Code expects native types, so we
        parse them back. E.g. follow_up: '[{"text":"Yes"}]' -> [{"text":"Yes"}]

        Also converts string "null" to None — Claude sometimes generates
        "cwd": "null" instead of "cwd": null for optional parameters.
        """
        result = {}
        for key, value in obj.items():
            if isinstance(value, str) and value == "null":
                result[key] = None
            elif isinstance(value, str) and value and value[0] in ('[', '{'):
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    result[key] = value
            else:
                result[key] = value
        return result

    def _extract_native_tool_call(self, block: dict) -> Optional[dict]:
        """Extract tool call from Claude native tool_use block."""
        tool_name = block.get("name")
        tool_input = block.get("input", {})
        tool_id = block.get("id", f"{TOOL_CALL_ID_PREFIX}{uuid.uuid4().hex[:ID_HEX_LENGTH]}")

        if not tool_name:
            return None

        if isinstance(tool_input, dict):
            tool_input = self._deserialize_string_values(tool_input)

        return {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
            },
        }

    # ==================== Execute Methods ====================

    async def execute(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> ProviderResult:
        """Execute Claude CLI (non-streaming)."""
        if tools:
            messages = ToolHandler.inject_tools_into_messages(messages, tools)

        cmd, stdin_prompt = self.build_command(messages, model=model, stream=False)

        try:
            stdin_data = stdin_prompt.encode() if stdin_prompt else None
            stdout, _ = await self._run_subprocess(cmd, timeout, stdin_data=stdin_data)
        except RuntimeError:
            raise

        output = self._parse_json_line(stdout.decode())
        if output is None:
            # Fallback: treat as plain text
            content = stdout.decode().strip()
            content, tool_calls = self._handle_tool_calls_in_content(content, tools)
            return ProviderResult(content=content or "", tool_calls=tool_calls)

        content = output.get("result", "")
        content, tool_calls = self._handle_tool_calls_in_content(content, tools)

        return ProviderResult(
            content=content or "",
            session_id=output.get("session_id"),
            usage=output.get("usage"),
            tool_calls=tool_calls,
        )

    async def execute_stream(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[ProviderChunk]:
        """Stream output from Claude CLI.

        When tools are provided, uses StreamingToolParser to:
        - Stream text immediately (no buffering)
        - Parse tool_calls on-the-fly using <tool_call> markers
        - Strip markers from output
        """
        if tools:
            logger.info(f"Injecting {len(tools)} tools into messages")
            messages = ToolHandler.inject_tools_into_messages(messages, tools)

        cmd, stdin_prompt = self.build_command(messages, model=model, stream=True)
        stdin_data = stdin_prompt.encode() if stdin_prompt else None
        logger.info(f"Claude stream: model={model}, prompt_len={len(stdin_prompt or cmd[2])}, via_stdin={stdin_prompt is not None}")

        proc = await self._create_stream_process(cmd, stdin_data=stdin_data)

        tool_parser = StreamingToolParser() if tools else None
        accumulated_content = ""
        native_tool_calls: list[dict] = []  # For Opus native tool calling

        try:
            async for line in proc.stdout:
                line = line.decode().strip()
                if not line:
                    continue

                event = self._parse_json_line(line)
                if event is None:
                    # Non-JSON line, treat as plain text
                    if tool_parser:
                        result = tool_parser.feed(line)
                        if result.text:
                            yield ProviderChunk(content=result.text)
                    elif line:
                        yield ProviderChunk(content=line)
                    accumulated_content += line
                    continue

                event_type = event.get("type", "")
                logger.debug(f"[STREAM] event_type={event_type}, keys={list(event.keys())}")

                # Handle stream_event wrapper (used with --include-partial-messages)
                if event_type == "stream_event":
                    inner_event = event.get("event", {})
                    inner_type = inner_event.get("type", "")
                    logger.debug(f"[STREAM] stream_event inner_type={inner_type}")

                    if inner_type == "content_block_delta":
                        delta = inner_event.get("delta", {})
                        delta_type = delta.get("type", "")
                        logger.info(f"[STREAM] delta_type={delta_type}, delta_keys={list(delta.keys())}")

                        if delta_type == "text_delta":
                            text = delta.get("text", "")
                            logger.info(f"[STREAM] text_delta: {text[:100]}..." if len(text) > 100 else f"[STREAM] text_delta: {text}")
                            accumulated_content += text

                            if tool_parser:
                                # Parse on-the-fly, stream only non-marker text
                                result = tool_parser.feed(text)
                                if result.text:
                                    yield ProviderChunk(content=result.text)
                            elif text:
                                yield ProviderChunk(content=text)

                        elif delta_type == "thinking_delta":
                            thinking = delta.get("thinking", "")
                            logger.info(f"[STREAM] THINKING: {thinking[:100]}..." if len(thinking) > 100 else f"[STREAM] THINKING: {thinking}")
                            if thinking:
                                yield ProviderChunk(content="", reasoning=thinking)

                        elif delta_type == "input_json_delta":
                            # Native tool calling (Opus) - accumulate JSON
                            partial_json = delta.get("partial_json", "")
                            logger.info(f"[STREAM] input_json_delta: {partial_json[:50]}...")
                            accumulated_content += partial_json

                    elif inner_type == "message_delta":
                        logger.info(f"[STREAM] message_delta: {inner_event.get('delta', {})}")
                        delta = inner_event.get("delta", {})
                        stop_reason = delta.get("stop_reason")
                        if stop_reason:
                            tool_calls = None
                            # Native tool calling (stop_reason=tool_use)
                            if stop_reason == "tool_use":
                                tool_calls = native_tool_calls if native_tool_calls else None
                                tool_calls = filter_valid_tool_calls(tool_calls, tools)
                                logger.info(f"[STREAM] Native tool_calls: {len(tool_calls) if tool_calls else 0}")
                            elif tool_parser:
                                final_result = tool_parser.finalize()
                                if final_result.text:
                                    yield ProviderChunk(content=final_result.text)
                                tool_calls = tool_parser.get_all_tool_calls()
                                tool_calls = filter_valid_tool_calls(tool_calls, tools)
                            else:
                                _, tool_calls = self._handle_tool_calls_in_content(
                                    accumulated_content, tools
                                )
                            yield self._create_final_chunk(tool_calls, inner_event.get("usage"))
                            return

                    continue

                # Handle direct event types (without stream_event wrapper)
                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type", "")

                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        accumulated_content += text

                        if tool_parser:
                            result = tool_parser.feed(text)
                            if result.text:
                                yield ProviderChunk(content=result.text)
                        elif text:
                            yield ProviderChunk(content=text)

                    elif delta_type == "thinking_delta":
                        thinking = delta.get("thinking", "")
                        if thinking:
                            yield ProviderChunk(content="", reasoning=thinking)

                elif event_type == "message_delta":
                    if tool_parser:
                        final_result = tool_parser.finalize()
                        if final_result.text:
                            yield ProviderChunk(content=final_result.text)
                        tool_calls = tool_parser.get_all_tool_calls()
                        tool_calls = filter_valid_tool_calls(tool_calls, tools)
                    else:
                        _, tool_calls = self._handle_tool_calls_in_content(
                            accumulated_content, tools
                        )
                    yield self._create_final_chunk(tool_calls, event.get("usage"))
                    return

                elif event_type == "assistant":
                    logger.info(f"[STREAM] ASSISTANT event, message keys: {list(event.get('message', {}).keys())}")
                    content_blocks = event.get("message", {}).get("content", [])
                    for i, block in enumerate(content_blocks):
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            logger.info(f"[STREAM] ASSISTANT block[{i}] type={block_type}")
                            # Extract native tool calls
                            if block_type == "tool_use":
                                tool_call = self._extract_native_tool_call(block)
                                if tool_call:
                                    native_tool_calls.append(tool_call)
                                    logger.info(f"[STREAM] Extracted native tool_call: {tool_call.get('function', {}).get('name')}")
                    async for chunk in self._process_assistant_event(event, accumulated_content):
                        # Only yield if content wasn't already streamed via text_delta
                        if chunk.content:
                            if chunk.content in accumulated_content:
                                # Already sent via text_delta, skip
                                continue
                            accumulated_content += chunk.content

                            if tool_parser:
                                result = tool_parser.feed(chunk.content)
                                if result.text:
                                    yield ProviderChunk(content=result.text)
                            elif chunk.content:
                                yield chunk
                        else:
                            yield chunk

                elif event_type == "result":
                    if tool_parser:
                        final_result = tool_parser.finalize()
                        if final_result.text:
                            yield ProviderChunk(content=final_result.text)
                        tool_calls = tool_parser.get_all_tool_calls()
                        tool_calls = filter_valid_tool_calls(tool_calls, tools)
                    else:
                        _, tool_calls = self._handle_tool_calls_in_content(
                            accumulated_content, tools
                        )
                    yield self._create_final_chunk(tool_calls, event.get("usage"))
                    return

            # Stream ended - send final chunk
            if tool_parser:
                final_result = tool_parser.finalize()
                if final_result.text:
                    yield ProviderChunk(content=final_result.text)
                tool_calls = tool_parser.get_all_tool_calls()
                tool_calls = filter_valid_tool_calls(tool_calls, tools)
            else:
                _, tool_calls = self._handle_tool_calls_in_content(accumulated_content, tools)

            # Log diagnostics when stream produced no content
            if not accumulated_content:
                stderr_data = b""
                if proc.stderr:
                    try:
                        stderr_data = await asyncio.wait_for(proc.stderr.read(), timeout=2)
                    except (asyncio.TimeoutError, Exception):
                        pass
                logger.warning(
                    f"[STREAM] Empty response from Claude CLI. "
                    f"returncode={proc.returncode}, "
                    f"stderr={stderr_data.decode(errors='replace')[:500]}"
                )

            yield self._create_final_chunk(tool_calls)

        finally:
            await self._cleanup_process(proc)

    async def _process_assistant_event(
        self, event: dict, accumulated_content: str
    ) -> AsyncIterator[ProviderChunk]:
        """Process assistant event content blocks."""
        message = event.get("message", {})
        content_blocks = message.get("content", [])

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            if block_type == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    yield ProviderChunk(content="", reasoning=thinking_text)

            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    yield ProviderChunk(content=text)

            elif block_type == "tool_use":
                # Native tool calls are extracted separately, no need to show in content
                pass
