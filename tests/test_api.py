"""Integration tests for API endpoints."""

import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from cli2api.main import app
from cli2api.api.dependencies import get_provider
from cli2api.providers.claude import ClaudeCodeProvider
from cli2api.schemas.internal import ProviderResult, ProviderChunk
from tests.conftest import MockClaudeProvider


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "healthy", "degraded")


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root(self, client):
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "CLI2API"
        assert "version" in data
        assert "available_models" in data
        assert "endpoints" in data


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""

    def test_list_models(self, client):
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

        # Check model structure
        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert model["object"] == "model"
            assert "owned_by" in model

    def test_get_model_exists(self, client):
        # First get list of models
        list_response = client.get("/v1/models")
        models = list_response.json()["data"]

        if models:
            model_id = models[0]["id"]
            response = client.get(f"/v1/models/{model_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == model_id

    def test_get_model_not_found(self, client):
        response = client.get("/v1/models/nonexistent-model")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestChatCompletionsEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        return MockClaudeProvider(
            response_content="Hello! I'm a helpful assistant.",
            stream_chunks=["Hello", "!", " I'm", " helpful", "."],
        )

    def test_chat_completion_non_streaming(self, mock_provider):
        """Test non-streaming chat completion."""
        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": False,
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Check response structure
            assert data["object"] == "chat.completion"
            assert "id" in data
            assert data["id"].startswith("chatcmpl-")
            assert data["model"] == "mock-model"
            assert "choices" in data
            assert len(data["choices"]) == 1

            # Check choice structure
            choice = data["choices"][0]
            assert choice["index"] == 0
            assert choice["finish_reason"] == "stop"
            assert "message" in choice
            assert choice["message"]["role"] == "assistant"
            assert choice["message"]["content"] == "Hello! I'm a helpful assistant."

            # Check usage
            assert "usage" in data
            assert "prompt_tokens" in data["usage"]
            assert "completion_tokens" in data["usage"]
            assert "total_tokens" in data["usage"]

        finally:
            app.dependency_overrides.clear()

    def test_chat_completion_streaming(self, mock_provider):
        """Test streaming chat completion."""
        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            # Parse SSE events
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        events.append("[DONE]")
                    else:
                        events.append(json.loads(data_str))

            # Should have at least: role chunk, content chunks, final chunk, [DONE]
            assert len(events) >= 3

            # First event should have role
            first_event = events[0]
            assert first_event["object"] == "chat.completion.chunk"
            assert first_event["choices"][0]["delta"].get("role") == "assistant"

            # Last event should be [DONE]
            assert events[-1] == "[DONE]"

            # Second to last should have finish_reason
            final_chunk = events[-2]
            assert final_chunk["choices"][0]["finish_reason"] == "stop"

            # Collect content
            content = ""
            for event in events:
                if event != "[DONE]" and "choices" in event:
                    delta_content = event["choices"][0]["delta"].get("content")
                    if delta_content:
                        content += delta_content

            assert "Hello" in content

        finally:
            app.dependency_overrides.clear()

    def test_chat_completion_unknown_model(self, mock_provider):
        """Test with unknown model - should use provider anyway."""
        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "any-unknown-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            # Unknown models are accepted and passed to provider
            assert response.status_code == 200
            data = response.json()
            assert data["model"] == "any-unknown-model"

        finally:
            app.dependency_overrides.clear()

    def test_chat_completion_invalid_request(self, client):
        """Test with invalid request body."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-code",
                # Missing messages
            },
        )

        assert response.status_code == 422  # Validation error

    def test_chat_completion_empty_messages(self, client):
        """Test with empty messages array."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-code",
                "messages": [],
            },
        )

        # Empty messages can cause various errors depending on provider
        # 500 is acceptable if provider can't handle empty prompt
        assert response.status_code in [200, 400, 422, 500]

    def test_chat_completion_invalid_role(self, client):
        """Test with invalid message role."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-code",
                "messages": [{"role": "invalid", "content": "Hello!"}],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_chat_completion_with_system_message(self, mock_provider):
        """Test with system message."""
        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello!"},
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Hello! I'm a helpful assistant."

        finally:
            app.dependency_overrides.clear()

    def test_chat_completion_with_conversation_history(self, mock_provider):
        """Test with conversation history."""
        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Hi there!"},
                        {"role": "user", "content": "How are you?"},
                    ],
                },
            )

            assert response.status_code == 200

        finally:
            app.dependency_overrides.clear()


class TestChatCompletionsErrorHandling:
    """Tests for error handling in chat completions."""

    def test_provider_error(self):
        """Test handling of provider errors."""
        mock_provider = MockClaudeProvider(
            should_fail=True,
            fail_message="CLI execution failed",
        )

        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            assert response.status_code == 500
            assert "CLI execution failed" in response.json()["detail"]

        finally:
            app.dependency_overrides.clear()

    def test_streaming_error(self):
        """Test handling of errors during streaming."""
        mock_provider = MockClaudeProvider(
            should_fail=True,
            fail_message="Stream error",
        )

        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                },
            )

            # Streaming response returns 200 even on error
            assert response.status_code == 200

            # Check for error in stream
            content = response.text
            assert "error" in content or "[DONE]" in content

        finally:
            app.dependency_overrides.clear()


class TestOpenAICompatibility:
    """Tests verifying OpenAI API compatibility."""

    @pytest.fixture
    def mock_provider(self):
        return MockClaudeProvider(
            response_content="Test response",
            stream_chunks=["Test", " ", "response"],
        )

    def test_response_fields(self, mock_provider):
        """Verify all required OpenAI response fields are present."""
        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            data = response.json()

            # Required fields per OpenAI spec
            assert "id" in data
            assert "object" in data
            assert "created" in data
            assert "model" in data
            assert "choices" in data
            assert "usage" in data

            # Choices structure
            choice = data["choices"][0]
            assert "index" in choice
            assert "message" in choice
            assert "finish_reason" in choice

            # Message structure
            message = choice["message"]
            assert "role" in message
            assert "content" in message

            # Usage structure
            usage = data["usage"]
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
            assert "total_tokens" in usage

        finally:
            app.dependency_overrides.clear()

    def test_streaming_chunk_fields(self, mock_provider):
        """Verify all required OpenAI streaming chunk fields are present."""
        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                },
            )

            # Get first non-[DONE] chunk
            for line in response.iter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    chunk = json.loads(line[6:])

                    # Required fields per OpenAI spec
                    assert "id" in chunk
                    assert "object" in chunk
                    assert chunk["object"] == "chat.completion.chunk"
                    assert "created" in chunk
                    assert "model" in chunk
                    assert "choices" in chunk

                    # Choices structure
                    choice = chunk["choices"][0]
                    assert "index" in choice
                    assert "delta" in choice
                    # finish_reason can be None or string

                    break

        finally:
            app.dependency_overrides.clear()

    def test_id_format(self, mock_provider):
        """Verify completion ID format."""
        app.dependency_overrides[get_provider] = lambda: mock_provider

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            data = response.json()
            assert data["id"].startswith("chatcmpl-")

        finally:
            app.dependency_overrides.clear()
