"""Claude Code CLI provider."""

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from cli2api.schemas.internal import ProviderChunk, ProviderResult
from cli2api.schemas.openai import ChatMessage
from cli2api.streaming.tool_parser import StreamingToolParser
from cli2api.tools.handler import ToolHandler
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


def _format_step_indicator(tool_name: str, tool_input: dict) -> Optional[str]:
    """Format step indicator for tool use."""
    step_text = None

    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        if cmd:
            step_text = f"`⚡ {cmd[:60]}{'...' if len(cmd) > 60 else ''}`\n"
    elif tool_name in ("Read", "Glob"):
        path = tool_input.get("file_path", tool_input.get("pattern", ""))
        step_text = f"`📄 {tool_name}: {path}`\n"
    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        if pattern:
            step_text = f"`🔍 {tool_name}: {pattern[:40]}...`\n"
        else:
            step_text = f"`🔍 {tool_name}`\n"
    elif tool_name in ("Edit", "Write"):
        path = tool_input.get("file_path", "")
        step_text = f"`✏️ {tool_name}: {path}`\n"
    elif tool_name == "Task":
        desc = tool_input.get("description", "")
        step_text = f"`🔧 {desc}`\n" if desc else f"`🔧 Task`\n"
    else:
        step_text = f"`🔧 {tool_name}`\n"

    return step_text


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

    def build_command(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> list[str]:
        """Build claude CLI command."""
        prompt, system_prompt = self.format_messages_as_prompt(messages)

        cmd = [
            str(self.executable_path),
            "-p",  # Print mode (non-interactive)
            prompt,
        ]

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

        return cmd

    # ==================== Subprocess Utilities ====================

    async def _run_subprocess(
        self,
        cmd: list[str],
        timeout: Optional[int] = None,
    ) -> tuple[bytes, bytes]:
        """Run subprocess with timeout handling."""
        timeout = timeout or self.default_timeout

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
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
    ) -> asyncio.subprocess.Process:
        """Create subprocess for streaming."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info(f"Claude process started: PID={proc.pid}")
        return proc

    async def _cleanup_process(self, proc: asyncio.subprocess.Process) -> None:
        """Clean up subprocess if still running."""
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

    def _parse_json_line(self, line: str) -> Optional[dict]:
        """Parse a JSON line, returning None if invalid."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _handle_tool_calls_in_content(
        self,
        content: str,
        tools: Optional[list[dict]],
    ) -> tuple[str, Optional[list[dict]]]:
        """Parse content for tool calls if tools are provided."""
        if tools and content:
            return ToolHandler.parse_tool_calls(content)
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

        cmd = self.build_command(messages, model=model, stream=False)

        try:
            stdout, _ = await self._run_subprocess(cmd, timeout)
        except RuntimeError:
            raise

        # Parse JSON output
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

        cmd = self.build_command(messages, model=model, stream=True)
        logger.info(f"Claude stream: model={model}, prompt_len={len(cmd[2])}")

        proc = await self._create_stream_process(cmd)

        # Use streaming parser when tools are provided
        tool_parser = StreamingToolParser() if tools else None
        accumulated_content = ""  # For fallback parsing

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

                # Handle stream_event wrapper (used with --include-partial-messages)
                if event_type == "stream_event":
                    inner_event = event.get("event", {})
                    inner_type = inner_event.get("type", "")

                    if inner_type == "content_block_delta":
                        delta = inner_event.get("delta", {})
                        delta_type = delta.get("type", "")

                        if delta_type == "text_delta":
                            text = delta.get("text", "")
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
                            if thinking:
                                yield ProviderChunk(content="", reasoning=thinking)

                    elif inner_type == "message_delta":
                        delta = inner_event.get("delta", {})
                        if delta.get("stop_reason"):
                            # Finalize tool parser
                            if tool_parser:
                                final_result = tool_parser.finalize()
                                if final_result.text:
                                    yield ProviderChunk(content=final_result.text)
                                tool_calls = tool_parser.get_all_tool_calls()
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
                    else:
                        _, tool_calls = self._handle_tool_calls_in_content(
                            accumulated_content, tools
                        )
                    yield self._create_final_chunk(tool_calls, event.get("usage"))
                    return

                elif event_type == "assistant":
                    # Process assistant event but don't duplicate content
                    # that was already received via text_delta events
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
            else:
                _, tool_calls = self._handle_tool_calls_in_content(accumulated_content, tools)

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
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                step_text = _format_step_indicator(tool_name, tool_input)
                if step_text:
                    yield ProviderChunk(content=step_text)
