"""Tests for Pydantic schemas."""

import time

import pytest
from pydantic import ValidationError

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
    UsageInfo,
)
from cli2api.schemas.internal import ProviderChunk, ProviderResult


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_valid_user_message(self):
        msg = ChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_valid_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_valid_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Hello!")

    def test_empty_content(self):
        msg = ChatMessage(role="user", content="")
        assert msg.content == ""


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_minimal_request(self):
        req = ChatCompletionRequest(
            model="claude-code",
            messages=[ChatMessage(role="user", content="Hello!")],
        )
        assert req.model == "claude-code"
        assert len(req.messages) == 1
        assert req.stream is False

    def test_streaming_request(self):
        req = ChatCompletionRequest(
            model="claude-code",
            messages=[ChatMessage(role="user", content="Hello!")],
            stream=True,
        )
        assert req.stream is True

    def test_with_temperature(self):
        req = ChatCompletionRequest(
            model="claude-code",
            messages=[ChatMessage(role="user", content="Hello!")],
            temperature=0.7,
        )
        assert req.temperature == 0.7

    def test_invalid_temperature_high(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="claude-code",
                messages=[ChatMessage(role="user", content="Hello!")],
                temperature=3.0,  # Max is 2
            )

    def test_invalid_temperature_low(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="claude-code",
                messages=[ChatMessage(role="user", content="Hello!")],
                temperature=-0.5,  # Min is 0
            )

    def test_with_max_tokens(self):
        req = ChatCompletionRequest(
            model="claude-code",
            messages=[ChatMessage(role="user", content="Hello!")],
            max_tokens=100,
        )
        assert req.max_tokens == 100

    def test_invalid_max_tokens(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="claude-code",
                messages=[ChatMessage(role="user", content="Hello!")],
                max_tokens=0,  # Must be > 0
            )

    def test_multiple_messages(self):
        req = ChatCompletionRequest(
            model="claude-code",
            messages=[
                ChatMessage(role="system", content="Be helpful"),
                ChatMessage(role="user", content="Hello!"),
                ChatMessage(role="assistant", content="Hi!"),
                ChatMessage(role="user", content="How are you?"),
            ],
        )
        assert len(req.messages) == 4


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse model."""

    def test_basic_response(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            model="claude-code",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
        )
        assert resp.id == "chatcmpl-123"
        assert resp.object == "chat.completion"
        assert resp.model == "claude-code"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "Hello!"

    def test_response_with_usage(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            model="claude-code",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(role="assistant", content="Hello!"),
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_created_timestamp(self):
        before = int(time.time())
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            model="claude-code",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(role="assistant", content="Hello!"),
                )
            ],
        )
        after = int(time.time())
        assert before <= resp.created <= after


class TestChatCompletionChunk:
    """Tests for streaming ChatCompletionChunk model."""

    def test_chunk_with_role(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="claude-code",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.role == "assistant"
        assert chunk.choices[0].delta.content is None

    def test_chunk_with_content(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="claude-code",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        assert chunk.choices[0].delta.content == "Hello"

    def test_final_chunk(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="claude-code",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(),
                    finish_reason="stop",
                )
            ],
        )
        assert chunk.choices[0].finish_reason == "stop"


class TestUsageInfo:
    """Tests for UsageInfo model."""

    def test_default_values(self):
        usage = UsageInfo()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_with_values(self):
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_model_info(self):
        model = ModelInfo(id="claude-code")
        assert model.id == "claude-code"
        assert model.object == "model"
        assert model.owned_by == "cli2api"


class TestModelsResponse:
    """Tests for ModelsResponse model."""

    def test_models_response(self):
        resp = ModelsResponse(
            data=[
                ModelInfo(id="claude: sonnet"),
                ModelInfo(id="claude: opus"),
            ]
        )
        assert resp.object == "list"
        assert len(resp.data) == 2
        assert resp.data[0].id == "claude: sonnet"


class TestProviderChunk:
    """Tests for internal ProviderChunk model."""

    def test_content_chunk(self):
        chunk = ProviderChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.is_final is False
        assert chunk.usage is None

    def test_final_chunk(self):
        chunk = ProviderChunk(content="", is_final=True)
        assert chunk.is_final is True

    def test_chunk_with_usage(self):
        chunk = ProviderChunk(
            content="",
            is_final=True,
            usage={"input_tokens": 10, "output_tokens": 20},
        )
        assert chunk.usage["input_tokens"] == 10


class TestProviderResult:
    """Tests for internal ProviderResult model."""

    def test_basic_result(self):
        result = ProviderResult(content="Hello World")
        assert result.content == "Hello World"
        assert result.session_id is None

    def test_result_with_metadata(self):
        result = ProviderResult(
            content="Hello",
            session_id="session-123",
            usage={"input_tokens": 5, "output_tokens": 10},
        )
        assert result.session_id == "session-123"
        assert result.usage["input_tokens"] == 5
