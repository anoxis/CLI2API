"""Chat completions endpoint - OpenAI compatible."""

import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from cli2api.api.dependencies import get_provider
from cli2api.api.utils import parse_model_name
from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaContent,
    ResponseMessage,
    StreamChoice,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)
from cli2api.streaming.sse import sse_encode, sse_error
from cli2api.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    provider: ClaudeCodeProvider = Depends(get_provider),
):
    """Create a chat completion.

    OpenAI-compatible endpoint supporting both streaming and non-streaming modes.

    Args:
        request: Chat completion request.
        provider: Claude provider (injected).
        settings: Application settings (injected).

    Returns:
        ChatCompletionResponse for non-streaming, StreamingResponse for streaming.
    """
    actual_model = parse_model_name(request.model)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if request.stream:
        return StreamingResponse(
            stream_completion(
                provider=provider,
                messages=request.messages,
                model=actual_model,
                completion_id=completion_id,
                tools=request.tools,
                reasoning_effort=request.reasoning_effort,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        try:
            result = await provider.execute(
                messages=request.messages,
                model=actual_model,
                tools=request.tools,
            )
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Build usage info
        usage = UsageInfo()
        if result.usage:
            usage = UsageInfo(
                prompt_tokens=result.usage.get("input_tokens", 0),
                completion_tokens=result.usage.get("output_tokens", 0),
                total_tokens=(
                    result.usage.get("input_tokens", 0)
                    + result.usage.get("output_tokens", 0)
                ),
            )

        # Check for tool_calls in result
        if result.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=ToolCallFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in result.tool_calls
            ]
            response = ChatCompletionResponse(
                id=completion_id,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ResponseMessage(
                            role="assistant",
                            content=result.content if result.content else None,
                            tool_calls=tool_calls,
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=usage,
            )
            return response.model_dump(exclude_none=True)

        response = ChatCompletionResponse(
            id=completion_id,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(role="assistant", content=result.content),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
        return response.model_dump(exclude_none=True)


def split_content_chunks(content: str, max_size: int = 150) -> list[str]:
    """Split large content into smaller chunks.

    Tries to split on word boundaries for cleaner output.

    Args:
        content: Content to split.
        max_size: Maximum chunk size in characters.

    Returns:
        List of content chunks.
    """
    if len(content) <= max_size:
        return [content]

    chunks = []
    remaining = content

    while remaining:
        if len(remaining) <= max_size:
            chunks.append(remaining)
            break

        # Try to find a good split point (space, newline)
        split_at = max_size
        for sep in [" ", "\n", ".", ",", ";"]:
            pos = remaining.rfind(sep, 0, max_size)
            if pos > max_size // 2:  # Don't split too early
                split_at = pos + 1
                break

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]

    return chunks


async def stream_completion(
    provider: ClaudeCodeProvider,
    messages: list[ChatMessage],
    model: str,
    completion_id: str,
    tools: list[dict] | None = None,
    reasoning_effort: str | None = None,
) -> AsyncIterator[str]:
    """Generate SSE events for a streaming completion.

    Args:
        provider: The Claude provider to use.
        messages: Chat messages.
        model: Model identifier.
        completion_id: Unique completion ID.
        tools: Optional tool definitions.
        reasoning_effort: Reasoning effort for extended thinking (low/medium/high).

    Yields:
        SSE-encoded strings.
    """
    created = int(time.time())
    sent_final = False

    logger.info(f"[{completion_id}] Starting stream for model={model}")

    try:
        # First chunk with role
        first_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield sse_encode(first_chunk.model_dump())

        # When tools are provided, buffer content to check for tool_calls before sending
        content_buffer = "" if tools else None

        # Stream content chunks
        async for chunk in provider.execute_stream(
            messages=messages, model=model, tools=tools, reasoning_effort=reasoning_effort
        ):
            if chunk.is_final:
                if not sent_final:
                    # Check for tool_calls in chunk or buffered content
                    tool_calls_data = chunk.tool_calls

                    # If we buffered content, check it for tool_calls
                    if content_buffer and not tool_calls_data:
                        from cli2api.tools.handler import ToolHandler
                        _, parsed_tools = ToolHandler.parse_tool_calls(content_buffer)
                        if parsed_tools:
                            tool_calls_data = parsed_tools

                    if tool_calls_data:
                        tool_calls = [
                            ToolCall(
                                id=tc["id"],
                                type=tc.get("type", "function"),
                                function=ToolCallFunction(
                                    name=tc["function"]["name"],
                                    arguments=tc["function"]["arguments"],
                                ),
                            )
                            for tc in tool_calls_data
                        ]
                        tool_chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaContent(tool_calls=tool_calls),
                                    finish_reason="tool_calls",
                                )
                            ],
                        )
                        yield sse_encode(tool_chunk.model_dump())
                        sent_final = True
                    else:
                        # No tool_calls - send buffered content if any
                        if content_buffer:
                            content_parts = split_content_chunks(content_buffer)
                            for part in content_parts:
                                content_chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=created,
                                    model=model,
                                    choices=[
                                        StreamChoice(
                                            index=0,
                                            delta=DeltaContent(content=part),
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                yield sse_encode(content_chunk.model_dump())

                        # Normal final chunk
                        final_chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaContent(),
                                    finish_reason="stop",
                                )
                            ],
                        )
                        yield sse_encode(final_chunk.model_dump())
                    sent_final = True

            elif chunk.reasoning:
                # Stream reasoning/thinking content
                from cli2api.schemas.openai import ReasoningDetail
                reasoning_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaContent(
                                reasoning_details=[
                                    ReasoningDetail(
                                        type="reasoning.text",
                                        text=chunk.reasoning,
                                    )
                                ]
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield sse_encode(reasoning_chunk.model_dump())

            elif chunk.content:
                # Check if this is a step indicator (should be streamed immediately)
                is_step = chunk.content.startswith("`") and any(
                    emoji in chunk.content for emoji in ["🤔", "⚡", "🔍", "📄", "🔧", "✏️"]
                )

                # When tools are provided, we need to buffer all content
                # because tool_call JSON can arrive in fragments
                if content_buffer is not None and not is_step:
                    content_buffer += chunk.content
                else:
                    # Stream text content immediately
                    content_parts = split_content_chunks(chunk.content)
                    for part in content_parts:
                        content_chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaContent(content=part),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield sse_encode(content_chunk.model_dump())

        # Ensure final chunk is sent
        if not sent_final:
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaContent(),
                        finish_reason="stop",
                    )
                ],
            )
            yield sse_encode(final_chunk.model_dump())

        # Final [DONE] event
        logger.info(f"[{completion_id}] Stream completed successfully")
        yield "data: [DONE]\n\n"

    except RuntimeError as e:
        logger.error(f"[{completion_id}] Provider error: {e}")
        yield sse_error(str(e))
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"[{completion_id}] Stream error: {e}")
        yield sse_error(str(e))
        yield "data: [DONE]\n\n"
