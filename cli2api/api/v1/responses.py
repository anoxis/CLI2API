"""Responses endpoint - OpenAI Responses API compatible.

This is the newer OpenAI API format that some clients use.
We translate it to our internal format and use the same provider.
"""

import time
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cli2api.api.dependencies import get_provider
from cli2api.api.utils import parse_model_name
from cli2api.constants import ID_HEX_LENGTH, MESSAGE_ID_PREFIX, RESPONSE_ID_PREFIX
from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.openai import ChatMessage
from cli2api.streaming.sse import sse_encode, sse_error
from cli2api.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# === Request Models for Responses API ===


class ResponsesInputMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: Any  # Can be string or list of content blocks


class ResponsesRequest(BaseModel):
    """Request body for /v1/responses."""

    model_config = ConfigDict(extra="ignore")

    model: str
    input: list[ResponsesInputMessage] | str
    stream: bool = False
    instructions: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(default=None, gt=0)
    # Additional fields
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[dict] = None

    @field_validator("input")
    @classmethod
    def input_not_empty(cls, v: list | str) -> list | str:
        """Validate that input is not empty."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("input cannot be empty string")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("input cannot be empty list")
        return v


# === Response Models ===


class ResponsesOutput(BaseModel):
    type: str = "message"
    id: str
    role: str = "assistant"
    content: list[dict]


class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    model: str
    output: list[ResponsesOutput]
    usage: Optional[dict] = None


def convert_to_chat_messages(request: ResponsesRequest) -> list[ChatMessage]:
    """Convert Responses API input to ChatMessage list.

    Args:
        request: The ResponsesRequest with input messages.

    Returns:
        List of ChatMessage objects for the provider.
    """
    messages: list[ChatMessage] = []

    # Add instructions as system message
    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))

    # Handle input - can be string or list of messages
    if isinstance(request.input, str):
        messages.append(ChatMessage(role="user", content=request.input))
    else:
        for msg in request.input:
            content: str = _extract_message_content(msg.content)
            role: str = msg.role if msg.role in ("system", "user", "assistant") else "user"
            messages.append(ChatMessage(role=role, content=content))

    return messages


def _extract_message_content(content: Any) -> str:
    """Extract text content from various message content formats.

    Args:
        content: Message content - can be str, list of content blocks, or None.

    Returns:
        Extracted text content as string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    return str(content)


@router.post("/responses")
async def create_response(
    request: ResponsesRequest,
    provider: ClaudeCodeProvider = Depends(get_provider),
):
    """Create a response using the Responses API format.

    This endpoint provides compatibility with the OpenAI Responses API.
    """
    actual_model = parse_model_name(request.model)

    # Convert to chat messages
    messages = convert_to_chat_messages(request)
    response_id = f"{RESPONSE_ID_PREFIX}{uuid.uuid4().hex[:ID_HEX_LENGTH]}"

    if request.stream:
        return StreamingResponse(
            stream_response(
                provider=provider,
                messages=messages,
                model=actual_model,
                response_id=response_id,
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
                messages=messages,
                model=actual_model,
            )
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        output_id = f"{MESSAGE_ID_PREFIX}{uuid.uuid4().hex[:ID_HEX_LENGTH]}"

        return ResponsesResponse(
            id=response_id,
            created_at=int(time.time()),
            model=request.model,
            output=[
                ResponsesOutput(
                    id=output_id,
                    content=[{"type": "text", "text": result.content}],
                )
            ],
            usage=result.usage,
        )


async def stream_response(
    provider: ClaudeCodeProvider,
    messages: list[ChatMessage],
    model: str,
    response_id: str,
) -> AsyncIterator[str]:
    """Generate SSE events for streaming response."""
    created = int(time.time())
    output_id = f"{MESSAGE_ID_PREFIX}{uuid.uuid4().hex[:ID_HEX_LENGTH]}"
    content_buffer = ""

    try:
        # Send response.created event
        yield sse_encode({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created,
                "model": model,
                "output": [],
            }
        })

        # Send output_item.added event
        yield sse_encode({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": output_id,
                "role": "assistant",
                "content": [],
            }
        })

        # Stream content
        async for chunk in provider.execute_stream(messages=messages, model=model):
            if chunk.content:
                content_buffer += chunk.content
                yield sse_encode({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": chunk.content,
                })

        # Send completion events
        yield sse_encode({
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": content_buffer,
        })

        yield sse_encode({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": output_id,
                "role": "assistant",
                "content": [{"type": "text", "text": content_buffer}],
            }
        })

        yield sse_encode({
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created,
                "model": model,
                "output": [{
                    "type": "message",
                    "id": output_id,
                    "role": "assistant",
                    "content": [{"type": "text", "text": content_buffer}],
                }],
            }
        })

        logger.info(f"[{response_id}] Stream completed successfully")
        yield "data: [DONE]\n\n"

    except RuntimeError as e:
        logger.error(f"[{response_id}] Provider error: {e}")
        yield sse_error(str(e))
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"[{response_id}] Stream error: {e}")
        yield sse_error(str(e))
        yield "data: [DONE]\n\n"
