"""Integration tests for Ollama Manager.

Tests cover:
- Health check
- Model listing
- Active model management
- Modelfile generation
- Error handling

Run with: pytest tests/integration/test_ollama_manager.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import json
import tempfile

# Import path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ollama_manager():
    """Create OllamaManager instance."""
    from conductor.ollama_manager import OllamaManager, OllamaConfig
    
    config = OllamaConfig(
        host="http://localhost:11434",
        timeout=30,
    )
    return OllamaManager(config)


@pytest.fixture
def mock_http_client():
    """Create mock HTTP client."""
    client = MagicMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.delete = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def temp_gguf_file(tmp_path):
    """Create temporary GGUF file."""
    gguf_file = tmp_path / "test_model.gguf"
    gguf_file.write_bytes(b"GGUF mock data")
    return gguf_file


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for Ollama health check."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_manager, mock_http_client) -> None:
        """Test health check when Ollama is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http_client.get.return_value = mock_response
        
        ollama_manager._client = mock_http_client
        
        result = await ollama_manager.health_check()
        
        assert result is True
        assert ollama_manager._is_connected is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_manager, mock_http_client) -> None:
        """Test health check when Ollama is not running."""
        mock_http_client.get.side_effect = Exception("Connection refused")
        
        ollama_manager._client = mock_http_client
        
        result = await ollama_manager.health_check()
        
        assert result is False
        assert ollama_manager._is_connected is False


# =============================================================================
# Model Listing Tests
# =============================================================================

class TestModelListing:
    """Tests for model listing."""
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, ollama_manager, mock_http_client) -> None:
        """Test listing models successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:3b", "size": 1000000},
                {"name": "codellama:7b", "size": 2000000},
            ]
        }
        mock_http_client.get.return_value = mock_response
        
        ollama_manager._client = mock_http_client
        
        models = await ollama_manager.list_models()
        
        assert len(models) == 2
        assert models[0]["name"] == "llama3.2:3b"
    
    @pytest.mark.asyncio
    async def test_list_models_empty(self, ollama_manager, mock_http_client) -> None:
        """Test empty model list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_http_client.get.return_value = mock_response
        
        ollama_manager._client = mock_http_client
        
        models = await ollama_manager.list_models()
        
        assert models == []


# =============================================================================
# Active Model Tests
# =============================================================================

class TestActiveModel:
    """Tests for active model management."""
    
    def test_set_active_model(self, ollama_manager, tmp_path, monkeypatch) -> None:
        """Test setting active model."""
        # Use temp directory
        config_path = tmp_path / "config" / "active_model.json"
        monkeypatch.chdir(tmp_path)
        
        result = ollama_manager.set_active_model("my-model")
        
        assert result is True
        assert config_path.exists()
        
        with open(config_path) as f:
            data = json.load(f)
        
        assert data["active_model"] == "my-model"
    
    def test_get_active_model(self, ollama_manager, tmp_path, monkeypatch) -> None:
        """Test getting active model."""
        # Setup config
        config_path = tmp_path / "config" / "active_model.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"active_model": "test-model"}))
        
        monkeypatch.chdir(tmp_path)
        
        result = ollama_manager.get_active_model()
        
        assert result == "test-model"
    
    def test_get_active_model_not_set(self, ollama_manager, tmp_path, monkeypatch) -> None:
        """Test getting active model when not set."""
        monkeypatch.chdir(tmp_path)
        
        result = ollama_manager.get_active_model()
        
        assert result is None


# =============================================================================
# Modelfile Generation Tests
# =============================================================================

class TestModelfileGeneration:
    """Tests for Modelfile generation."""
    
    def test_generate_modelfile(self, ollama_manager) -> None:
        """Test Modelfile content generation."""
        content = ollama_manager._generate_modelfile(
            gguf_path="/path/to/model.gguf",
            system_prompt="You are helpful.",
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            num_predict=512,
        )
        
        assert "FROM /path/to/model.gguf" in content
        assert 'SYSTEM "You are helpful."' in content
        assert "PARAMETER temperature 0.7" in content
        assert "PARAMETER top_k 40" in content
    
    def test_generate_modelfile_escapes_quotes(self, ollama_manager) -> None:
        """Test that quotes in system prompt are escaped."""
        content = ollama_manager._generate_modelfile(
            gguf_path="/path/to/model.gguf",
            system_prompt='Say "hello" to the user.',
        )
        
        assert '\\"hello\\"' in content


# =============================================================================
# Model Creation Tests
# =============================================================================

class TestModelCreation:
    """Tests for model creation."""
    
    @pytest.mark.asyncio
    async def test_create_model_from_gguf_missing_file(self, ollama_manager) -> None:
        """Test creation fails for missing GGUF."""
        result = await ollama_manager.create_model_from_gguf(
            model_name="test",
            gguf_path="/nonexistent/model.gguf",
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_model_success(
        self, ollama_manager, mock_http_client, temp_gguf_file
    ) -> None:
        """Test successful model creation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http_client.post.return_value = mock_response
        
        ollama_manager._client = mock_http_client
        
        result = await ollama_manager.create_model_from_gguf(
            model_name="test-model",
            gguf_path=str(temp_gguf_file),
            system_prompt="You are a test model.",
        )
        
        assert result is True
        
        # Check Modelfile was created
        modelfile_path = temp_gguf_file.parent / "Modelfile.test-model"
        assert modelfile_path.exists()


# =============================================================================
# Query Model Tests
# =============================================================================

class TestQueryModel:
    """Tests for model querying."""
    
    @pytest.mark.asyncio
    async def test_query_model_with_active(
        self, ollama_manager, mock_http_client, tmp_path, monkeypatch
    ) -> None:
        """Test query using active model."""
        # Setup active model
        config_path = tmp_path / "config" / "active_model.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"active_model": "default-model"}))
        monkeypatch.chdir(tmp_path)
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_http_client.post.return_value = mock_response
        
        ollama_manager._client = mock_http_client
        
        result = await ollama_manager.query_model("Hello!")
        
        assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_query_model_no_active(self, ollama_manager, tmp_path, monkeypatch) -> None:
        """Test query fails when no active model."""
        monkeypatch.chdir(tmp_path)
        
        with pytest.raises(ValueError, match="No model specified"):
            await ollama_manager.query_model("Hello!")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_generate_handles_error(
        self, ollama_manager, mock_http_client
    ) -> None:
        """Test generation error handling."""
        mock_http_client.post.side_effect = Exception("Network error")
        
        ollama_manager._client = mock_http_client
        
        with pytest.raises(Exception, match="Network error"):
            await ollama_manager.generate("model", "prompt")
    
    @pytest.mark.asyncio
    async def test_create_model_handles_failure(
        self, ollama_manager, mock_http_client, temp_gguf_file
    ) -> None:
        """Test model creation error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_http_client.post.return_value = mock_response
        
        ollama_manager._client = mock_http_client
        
        result = await ollama_manager.create_model_from_gguf(
            model_name="test",
            gguf_path=str(temp_gguf_file),
        )
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
