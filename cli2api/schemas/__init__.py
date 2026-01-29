"""Pydantic schemas for API requests and responses."""

from cli2api.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaContent,
    ModelInfo,
    ModelsResponse,
    ResponseMessage,
    StreamChoice,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)
from cli2api.schemas.internal import ProviderChunk, ProviderResult

__all__ = [
    "ChatCompletionChunk",
    "ChatCompletionChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "DeltaContent",
    "ModelInfo",
    "ModelsResponse",
    "ProviderChunk",
    "ProviderResult",
    "ResponseMessage",
    "StreamChoice",
    "ToolCall",
    "ToolCallFunction",
    "UsageInfo",
]
