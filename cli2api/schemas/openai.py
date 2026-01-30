"""OpenAI-compatible Pydantic models."""

import time
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# === Request Models ===


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[Union[str, list[Any]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[Any]] = None
    tool_call_id: Optional[str] = None

    def model_dump(self, **kwargs):
        """Exclude None values by default for cleaner responses."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        """Convert content to string if it's a list of text blocks."""
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # Extract text from content blocks
            texts = []
            for item in v:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
            return "\n".join(texts) if texts else ""
        return str(v)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    # Additional fields for compatibility (ignored by CLI providers)
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str] | str] = None
    user: Optional[str] = None
    # Extended OpenAI fields
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    n: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    # Reasoning models (o1, o3, etc.)
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None


# === Response Models ===


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    def model_dump(self, **kwargs):
        """Exclude None values by default for cleaner responses."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        """Exclude None values by default for cleaner JSON."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = (
        "stop"
    )

    def model_dump(self, **kwargs):
        """Exclude None values by default for cleaner responses."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)

    def model_dump(self, **kwargs):
        """Exclude None values by default for cleaner responses."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


# === Streaming Response Models ===


class ReasoningDetail(BaseModel):
    type: Literal["reasoning.text", "reasoning.summary", "reasoning.encrypted"] = "reasoning.text"
    text: Optional[str] = None
    summary: Optional[str] = None
    id: Optional[str] = None
    index: Optional[int] = None

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class DeltaContent(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    reasoning_details: Optional[list[ReasoningDetail]] = None

    def model_dump(self, **kwargs):
        """Exclude None values by default for cleaner responses."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None

    def model_dump(self, **kwargs):
        """Exclude None values by default for cleaner responses."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]

    def model_dump(self, **kwargs):
        """Exclude None values by default for cleaner responses."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


# === Models Endpoint ===


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "cli2api"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
