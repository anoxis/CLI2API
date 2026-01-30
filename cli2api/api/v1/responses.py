"""Responses endpoint - OpenAI Responses API compatible.

This is the newer OpenAI API format that some clients use.
We translate it to our internal format and use the same provider.
"""

import time
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

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
    """Input message for Responses API."""

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
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    # Additional fields
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[dict] = None


# === Response Models ===


class ResponsesOutput(BaseModel):
    """Output content in response."""

    type: str = "message"
    id: str
    role: str = "assistant"
    content: list[dict]


class ResponsesResponse(BaseModel):
    """Response body for /v1/responses."""

    id: str
    object: str = "response"
    created_at: int
    model: str
    output: list[ResponsesOutput]
    usage: Optional[dict] = None


def convert_to_chat_messages(request: ResponsesRequest) -> list[ChatMessage]:
    """Convert Responses API input to ChatMessage list."""
    messages = []

    # Add instructions as system message
    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))

    # Handle input
    if isinstance(request.input, str):
        messages.append(ChatMessage(role="user", content=request.input))
    else:
        for msg in request.input:
            content = msg.content
            if isinstance(content, list):
                # Extract text from content blocks
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        texts.append(item)
                content = "\n".join(texts)
            elif content is None:
                content = ""

            role = msg.role
            if role not in ("system", "user", "assistant"):
                role = "user"  # Default to user for unknown roles

            messages.append(ChatMessage(role=role, content=str(content)))

    return messages


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
