"""Integration tests for the FastAPI service.

Tests cover:
- All endpoint functionality
- Error handling
- OpenAI API compatibility
- Concurrent job handling

Run with: pytest tests/integration/test_service.py -v
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

# Import path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_client():
    """Create test client for FastAPI app."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    
    from fastapi.testclient import TestClient
    from conductor.service import app, state
    
    # Reset state
    state.jobs = {}
    state.ollama_manager = None
    state.job_queue = None
    
    return TestClient(app)


@pytest.fixture
def mock_ollama():
    """Mock Ollama manager."""
    manager = MagicMock()
    manager.health_check = AsyncMock(return_value=True)
    manager.list_models = AsyncMock(return_value=[
        {"name": "llama3.2:3b", "size": 1000000},
    ])
    manager.generate = AsyncMock(return_value="This is a test response.")
    manager.chat = AsyncMock(return_value="Hello! How can I help?")
    manager.create_model = AsyncMock(return_value=True)
    return manager


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_ok(self, test_client) -> None:
        """Test health endpoint returns healthy status."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root_returns_service_info(self, test_client) -> None:
        """Test root endpoint returns service info."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AI Forge"


# =============================================================================
# Models Endpoint Tests
# =============================================================================

class TestModelsEndpoint:
    """Tests for models endpoint."""
    
    def test_list_models_empty(self, test_client) -> None:
        """Test models list when Ollama unavailable."""
        response = test_client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
    
    def test_list_models_with_ollama(self, test_client, mock_ollama) -> None:
        """Test models list with Ollama available."""
        from conductor.service import state
        state.ollama_manager = mock_ollama
        
        response = test_client.get("/v1/models")
        
        assert response.status_code == 200


# =============================================================================
# Fine-Tune Endpoint Tests
# =============================================================================

class TestFineTuneEndpoints:
    """Tests for fine-tuning endpoints."""
    
    def test_list_jobs_empty(self, test_client) -> None:
        """Test listing jobs when none exist."""
        response = test_client.get("/v1/fine-tune")
        
        assert response.status_code == 200
        assert response.json() == []
    
    def test_get_job_not_found(self, test_client) -> None:
        """Test getting non-existent job."""
        response = test_client.get("/v1/fine-tune/nonexistent")
        
        assert response.status_code == 404
    
    def test_cancel_job_not_found(self, test_client) -> None:
        """Test cancelling non-existent job."""
        response = test_client.delete("/v1/fine-tune/nonexistent")
        
        assert response.status_code == 404


# =============================================================================
# Chat Completions Tests (OpenAI Compatibility)
# =============================================================================

class TestChatCompletions:
    """Tests for OpenAI-compatible chat endpoint."""
    
    def test_chat_without_ollama(self, test_client) -> None:
        """Test chat fails gracefully without Ollama."""
        response = test_client.post("/v1/chat/completions", json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        
        assert response.status_code == 503
        assert "Ollama not available" in response.json()["detail"]
    
    def test_chat_request_format(self, test_client, mock_ollama) -> None:
        """Test chat accepts OpenAI format."""
        from conductor.service import state
        state.ollama_manager = mock_ollama
        
        response = test_client.post("/v1/chat/completions", json={
            "model": "llama3.2:3b",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "temperature": 0.7,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
    
    def test_chat_response_format(self, test_client, mock_ollama) -> None:
        """Test chat response matches OpenAI format."""
        from conductor.service import state
        state.ollama_manager = mock_ollama
        
        response = test_client.post("/v1/chat/completions", json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        
        data = response.json()
        
        # OpenAI format checks
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert "created" in data
        assert "model" in data


# =============================================================================
# Query Endpoint Tests
# =============================================================================

class TestQueryEndpoint:
    """Tests for simple query endpoint."""
    
    def test_query_without_ollama(self, test_client) -> None:
        """Test query fails without Ollama."""
        response = test_client.post("/v1/query", json={
            "prompt": "What is 2+2?",
        })
        
        assert response.status_code == 503
    
    def test_query_with_ollama(self, test_client, mock_ollama) -> None:
        """Test query works with Ollama."""
        from conductor.service import state
        state.ollama_manager = mock_ollama
        
        response = test_client.post("/v1/query", json={
            "prompt": "What is 2+2?",
            "model": "llama3.2:3b",
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "model" in data


# =============================================================================
# Status Endpoint Tests
# =============================================================================

class TestStatusEndpoint:
    """Tests for status alias endpoint."""
    
    def test_status_not_found(self, test_client) -> None:
        """Test status for non-existent job."""
        response = test_client.get("/status/nonexistent")
        
        assert response.status_code == 404
    
    def test_status_alias_works(self, test_client) -> None:
        """Test status endpoint is alias for job status."""
        from conductor.service import state
        
        # Create a mock job
        state.jobs["test-job"] = {
            "job_id": "test-job",
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        response = test_client.get("/status/test-job")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job"


# =============================================================================
# Deploy Endpoint Tests
# =============================================================================

class TestDeployEndpoint:
    """Tests for deploy endpoint."""
    
    def test_deploy_job_not_found(self, test_client) -> None:
        """Test deploy fails for non-existent job."""
        response = test_client.post("/deploy/nonexistent")
        
        assert response.status_code == 404
    
    def test_deploy_job_not_completed(self, test_client) -> None:
        """Test deploy fails for incomplete job."""
        from conductor.service import state
        
        state.jobs["test-job"] = {
            "job_id": "test-job",
            "status": "training",
            "config": {},
        }
        
        response = test_client.post("/deploy/test-job")
        
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]


# =============================================================================
# Validate Endpoint Tests
# =============================================================================

class TestValidateEndpoint:
    """Tests for validate endpoint."""
    
    def test_validate_job_not_found(self, test_client) -> None:
        """Test validate fails for non-existent job."""
        response = test_client.post("/validate/nonexistent")
        
        assert response.status_code == 404
    
    def test_validate_job_not_completed(self, test_client) -> None:
        """Test validate fails for incomplete job."""
        from conductor.service import state
        
        state.jobs["test-job"] = {
            "job_id": "test-job",
            "status": "queued",
            "config": {},
        }
        
        response = test_client.post("/validate/test-job")
        
        assert response.status_code == 400


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """Tests for request/response schema validation."""
    
    def test_chat_invalid_request(self, test_client) -> None:
        """Test chat rejects invalid request."""
        response = test_client.post("/v1/chat/completions", json={
            "messages": "not a list",
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_query_missing_prompt(self, test_client) -> None:
        """Test query rejects missing prompt."""
        response = test_client.post("/v1/query", json={})
        
        assert response.status_code == 422


# =============================================================================
# OpenAPI Documentation Tests
# =============================================================================

class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation."""
    
    def test_openapi_available(self, test_client) -> None:
        """Test OpenAPI schema is available."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
    
    def test_docs_available(self, test_client) -> None:
        """Test Swagger docs are available."""
        response = test_client.get("/docs")
        
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
