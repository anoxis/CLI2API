"""Models endpoint - OpenAI compatible."""

from fastapi import APIRouter, Depends, HTTPException

from cli2api.api.dependencies import get_provider
from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.openai import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/models")
async def list_models(
    provider: ClaudeCodeProvider = Depends(get_provider),
) -> ModelsResponse:
    """List available models.

    Returns:
        ModelsResponse with all available Claude models.
    """
    models = []
    for model_id in provider.supported_models:
        # Format as "claude: model"
        full_id = f"claude: {model_id}"
        models.append(ModelInfo(id=full_id, owned_by="claude"))
    return ModelsResponse(data=sorted(models, key=lambda m: m.id))


@router.get("/models/{model_id:path}")
async def get_model(
    model_id: str,
    provider: ClaudeCodeProvider = Depends(get_provider),
) -> ModelInfo:
    """Get information about a specific model.

    Args:
        model_id: The model identifier (e.g., "claude: sonnet" or "sonnet").
        provider: Claude provider (injected).

    Returns:
        ModelInfo for the requested model.

    Raises:
        HTTPException: If model not found.
    """
    # Parse model name
    actual_model = model_id.split(": ", 1)[1] if ": " in model_id else model_id

    if actual_model in provider.supported_models:
        return ModelInfo(id=f"claude: {actual_model}", owned_by="claude")

    raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
