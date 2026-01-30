"""Chat completions endpoint - OpenAI compatible."""

import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from cli2api.api.dependencies import get_provider
from cli2api.api.utils import parse_model_name
from cli2api.constants import (
    CHAT_COMPLETION_ID_PREFIX,
    CHUNK_SPLIT_MIN_RATIO,
    CHUNK_SPLIT_SEPARATORS,
    FINISH_REASON_STOP,
    FINISH_REASON_TOOL_CALLS,
    HTTP_GATEWAY_TIMEOUT,
    HTTP_INTERNAL_ERROR,
    ID_HEX_LENGTH,
    STREAM_CHUNK_MAX_SIZE,
)
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


def _convert_tool_calls(tool_calls_data: list[dict]) -> list[ToolCall]:
    """Convert raw tool call dicts to OpenAI ToolCall objects.

    Args:
        tool_calls_data: List of tool call dicts with id, type, function fields.

    Returns:
        List of ToolCall objects.
    """
    return [
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


# ==================== Chunk Factory Functions ====================


def _create_role_chunk(completion_id: str, created: int, model: str) -> ChatCompletionChunk:
    """Create the initial chunk with assistant role."""
    return ChatCompletionChunk(
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


def _create_content_chunk(
    completion_id: str, created: int, model: str, content: str
) -> ChatCompletionChunk:
    """Create a content delta chunk."""
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(content=content),
                finish_reason=None,
            )
        ],
    )


def _create_tool_calls_chunk(
    completion_id: str, created: int, model: str, tool_calls: list[ToolCall]
) -> ChatCompletionChunk:
    """Create a chunk with tool calls."""
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(tool_calls=tool_calls),
                finish_reason=FINISH_REASON_TOOL_CALLS,
            )
        ],
    )


def _create_final_chunk(
    completion_id: str, created: int, model: str
) -> ChatCompletionChunk:
    """Create the final stop chunk."""
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(),
                finish_reason=FINISH_REASON_STOP,
            )
        ],
    )


def _create_reasoning_chunk(
    completion_id: str, created: int, model: str, reasoning_text: str
) -> ChatCompletionChunk:
    """Create a reasoning/thinking chunk."""
    from cli2api.schemas.openai import ReasoningDetail
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(
                    reasoning_details=[
                        ReasoningDetail(type="reasoning.text", text=reasoning_text)
                    ]
                ),
                finish_reason=None,
            )
        ],
    )


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
    completion_id = f"{CHAT_COMPLETION_ID_PREFIX}{uuid.uuid4().hex[:ID_HEX_LENGTH]}"

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
            raise HTTPException(status_code=HTTP_GATEWAY_TIMEOUT, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail=str(e))

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
            tool_calls = _convert_tool_calls(result.tool_calls)
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
                        finish_reason=FINISH_REASON_TOOL_CALLS,
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
                    finish_reason=FINISH_REASON_STOP,
                )
            ],
            usage=usage,
        )
        return response.model_dump(exclude_none=True)


def split_content_chunks(content: str, max_size: int = STREAM_CHUNK_MAX_SIZE) -> list[str]:
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

        # Try to find a good split point (space, newline, punctuation)
        split_at = max_size
        min_split_pos = int(max_size * CHUNK_SPLIT_MIN_RATIO)
        for sep in CHUNK_SPLIT_SEPARATORS:
            pos = remaining.rfind(sep, 0, max_size)
            if pos > min_split_pos:
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
        # Send initial chunk with assistant role
        yield sse_encode(_create_role_chunk(completion_id, created, model).model_dump())

        # Stream content from provider
        async for chunk in provider.execute_stream(
            messages=messages, model=model, tools=tools, reasoning_effort=reasoning_effort
        ):
            if chunk.is_final:
                sent_final = True
                if chunk.tool_calls:
                    tool_calls = _convert_tool_calls(chunk.tool_calls)
                    yield sse_encode(
                        _create_tool_calls_chunk(completion_id, created, model, tool_calls).model_dump()
                    )
                else:
                    yield sse_encode(
                        _create_final_chunk(completion_id, created, model).model_dump()
                    )

            elif chunk.reasoning:
                yield sse_encode(
                    _create_reasoning_chunk(completion_id, created, model, chunk.reasoning).model_dump()
                )

            elif chunk.content:
                for part in split_content_chunks(chunk.content):
                    if part:
                        yield sse_encode(
                            _create_content_chunk(completion_id, created, model, part).model_dump()
                        )

        # Ensure final chunk is sent
        if not sent_final:
            yield sse_encode(_create_final_chunk(completion_id, created, model).model_dump())

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
