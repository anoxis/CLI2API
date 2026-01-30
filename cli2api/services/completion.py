"""Chat completion service - business logic layer."""

import time
import uuid
from typing import AsyncIterator

from cli2api.constants import (
    CHAT_COMPLETION_ID_PREFIX,
    CHUNK_SPLIT_MIN_RATIO,
    CHUNK_SPLIT_SEPARATORS,
    FINISH_REASON_STOP,
    FINISH_REASON_TOOL_CALLS,
    ID_HEX_LENGTH,
    STREAM_CHUNK_MAX_SIZE,
)
from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    DeltaContent,
    ReasoningDetail,
    ResponseMessage,
    StreamChoice,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)
from cli2api.streaming.sse import sse_encode, sse_error
from cli2api.utils.logging import get_logger

logger = get_logger(__name__)


def convert_tool_calls(tool_calls_data: list[dict]) -> list[ToolCall]:
    """Convert raw tool call dicts to OpenAI ToolCall objects."""
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


def split_content_chunks(content: str, max_size: int = STREAM_CHUNK_MAX_SIZE) -> list[str]:
    """Split large content into smaller chunks at word boundaries."""
    if len(content) <= max_size:
        return [content]

    chunks = []
    remaining = content

    while remaining:
        if len(remaining) <= max_size:
            chunks.append(remaining)
            break

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


def _create_role_chunk(completion_id: str, created: int, model: str) -> ChatCompletionChunk:
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


class CompletionService:
    """Service for handling chat completions."""

    def __init__(self, provider: ClaudeCodeProvider):
        self.provider = provider

    def generate_completion_id(self) -> str:
        return f"{CHAT_COMPLETION_ID_PREFIX}{uuid.uuid4().hex[:ID_HEX_LENGTH]}"

    async def create_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        completion_id: str,
        tools: list[dict] | None = None,
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion."""
        result = await self.provider.execute(
            messages=messages,
            model=model,
            tools=tools,
        )

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

        if result.tool_calls:
            tool_calls = convert_tool_calls(result.tool_calls)
            return ChatCompletionResponse(
                id=completion_id,
                model=model,
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

        return ChatCompletionResponse(
            id=completion_id,
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(role="assistant", content=result.content),
                    finish_reason=FINISH_REASON_STOP,
                )
            ],
            usage=usage,
        )

    async def stream_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        completion_id: str,
        tools: list[dict] | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Generate SSE events for a streaming completion."""
        created = int(time.time())
        sent_final = False

        logger.info(f"[{completion_id}] Starting stream for model={model}")

        try:
            yield sse_encode(_create_role_chunk(completion_id, created, model).model_dump())

            async for chunk in self.provider.execute_stream(
                messages=messages, model=model, tools=tools, reasoning_effort=reasoning_effort
            ):
                if chunk.is_final:
                    sent_final = True
                    if chunk.tool_calls:
                        tool_calls = convert_tool_calls(chunk.tool_calls)
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
