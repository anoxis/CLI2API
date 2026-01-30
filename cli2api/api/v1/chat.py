"""Chat completions endpoint - OpenAI compatible."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from cli2api.api.dependencies import get_provider
from cli2api.api.utils import parse_model_name
from cli2api.constants import HTTP_GATEWAY_TIMEOUT, HTTP_INTERNAL_ERROR
from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.openai import ChatCompletionRequest
from cli2api.services.completion import CompletionService

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    provider: ClaudeCodeProvider = Depends(get_provider),
):
    """Create a chat completion (OpenAI-compatible endpoint)."""
    actual_model = parse_model_name(request.model)
    service = CompletionService(provider)
    completion_id = service.generate_completion_id()

    if request.stream:
        return StreamingResponse(
            service.stream_completion(
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

    try:
        response = await service.create_completion(
            messages=request.messages,
            model=actual_model,
            completion_id=completion_id,
            tools=request.tools,
        )
        return response.model_dump(exclude_none=True)
    except TimeoutError as e:
        raise HTTPException(status_code=HTTP_GATEWAY_TIMEOUT, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail=str(e))
