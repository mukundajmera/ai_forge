"""Unit tests for Ollama Manager.

Tests cover:
- Health check
- Model listing
- Active model management
- Modelfile generation
- Error handling

Run with: pytest tests/unit/test_ollama_manager.py -v
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import json

# Import path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ollama_config():
    """Create OllamaConfig for testing."""
    from conductor.ollama_manager import OllamaConfig
    
    return OllamaConfig(
        host="http://localhost:11434",
        timeout=30,
    )


@pytest.fixture
def ollama_manager(ollama_config):
    """Create OllamaManager instance."""
    from conductor.ollama_manager import OllamaManager
    
    return OllamaManager(ollama_config)


@pytest.fixture
def mock_client():
    """Create mock HTTP client."""
    client = MagicMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.delete = AsyncMock()
    client.aclose = AsyncMock()
    return client


# =============================================================================
# Configuration Tests
# =============================================================================

class TestOllamaConfig:
    """Tests for OllamaConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        from conductor.ollama_manager import OllamaConfig
        
        config = OllamaConfig()
        
        assert config.host == "http://localhost:11434"
        assert config.timeout > 0
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from conductor.ollama_manager import OllamaConfig
        
        config = OllamaConfig(
            host="http://custom:8080",
            timeout=60,
        )
        
        assert config.host == "http://custom:8080"
        assert config.timeout == 60


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_manager, mock_client) -> None:
        """Test health check when Ollama is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        
        ollama_manager._client = mock_client
        
        result = await ollama_manager.health_check()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_manager, mock_client) -> None:
        """Test health check when Ollama is not running."""
        mock_client.get.side_effect = Exception("Connection refused")
        
        ollama_manager._client = mock_client
        
        result = await ollama_manager.health_check()
        
        assert result is False


# =============================================================================
# Model Listing Tests
# =============================================================================

class TestModelListing:
    """Tests for model listing."""
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, ollama_manager, mock_client) -> None:
        """Test listing models successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:3b", "size": 1000000},
                {"name": "codellama:7b", "size": 2000000},
            ]
        }
        mock_client.get.return_value = mock_response
        
        ollama_manager._client = mock_client
        
        models = await ollama_manager.list_models()
        
        assert len(models) == 2
    
    @pytest.mark.asyncio
    async def test_list_models_empty(self, ollama_manager, mock_client) -> None:
        """Test empty model list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_client.get.return_value = mock_response
        
        ollama_manager._client = mock_client
        
        models = await ollama_manager.list_models()
        
        assert models == []


# =============================================================================
# Active Model Tests
# =============================================================================

class TestActiveModelManagement:
    """Tests for active model configuration."""
    
    def test_set_active_model(self, ollama_manager, tmp_path, monkeypatch) -> None:
        """Test setting active model."""
        monkeypatch.chdir(tmp_path)
        
        result = ollama_manager.set_active_model("test-model")
        
        assert result is True
        
        config_path = tmp_path / "config" / "active_model.json"
        assert config_path.exists()
        
        with open(config_path) as f:
            data = json.load(f)
        
        assert data["active_model"] == "test-model"
    
    def test_get_active_model(self, ollama_manager, tmp_path, monkeypatch) -> None:
        """Test getting active model."""
        config_path = tmp_path / "config" / "active_model.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"active_model": "saved-model"}))
        
        monkeypatch.chdir(tmp_path)
        
        result = ollama_manager.get_active_model()
        
        assert result == "saved-model"
    
    def test_get_active_model_not_set(self, ollama_manager, tmp_path, monkeypatch) -> None:
        """Test getting active model when not configured."""
        monkeypatch.chdir(tmp_path)
        
        result = ollama_manager.get_active_model()
        
        assert result is None


# =============================================================================
# Modelfile Generation Tests
# =============================================================================

class TestModelfileGeneration:
    """Tests for Modelfile generation."""
    
    def test_generate_modelfile_basic(self, ollama_manager) -> None:
        """Test basic Modelfile generation."""
        content = ollama_manager._generate_modelfile(
            gguf_path="/path/to/model.gguf",
            system_prompt="You are helpful.",
        )
        
        assert "FROM /path/to/model.gguf" in content
        assert 'SYSTEM "You are helpful."' in content
        assert "PARAMETER temperature" in content
    
    def test_generate_modelfile_custom_params(self, ollama_manager) -> None:
        """Test Modelfile with custom parameters."""
        content = ollama_manager._generate_modelfile(
            gguf_path="/model.gguf",
            system_prompt="Test",
            temperature=0.5,
            top_k=50,
            top_p=0.95,
            num_predict=1024,
        )
        
        assert "PARAMETER temperature 0.5" in content
        assert "PARAMETER top_k 50" in content
        assert "PARAMETER top_p 0.95" in content
        assert "PARAMETER num_predict 1024" in content
    
    def test_generate_modelfile_escapes_quotes(self, ollama_manager) -> None:
        """Test quote escaping in system prompt."""
        content = ollama_manager._generate_modelfile(
            gguf_path="/model.gguf",
            system_prompt='Say "hello" to user.',
        )
        
        assert '\\"hello\\"' in content


# =============================================================================
# Model Creation Tests
# =============================================================================

class TestModelCreation:
    """Tests for model creation."""
    
    @pytest.mark.asyncio
    async def test_create_model_missing_gguf(self, ollama_manager) -> None:
        """Test creation fails for missing GGUF."""
        result = await ollama_manager.create_model_from_gguf(
            model_name="test",
            gguf_path="/nonexistent/model.gguf",
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_model_success(
        self, ollama_manager, mock_client, tmp_path
    ) -> None:
        """Test successful model creation."""
        # Create mock GGUF file
        gguf_file = tmp_path / "test.gguf"
        gguf_file.write_bytes(b"GGUF")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        
        ollama_manager._client = mock_client
        
        result = await ollama_manager.create_model_from_gguf(
            model_name="test-model",
            gguf_path=str(gguf_file),
        )
        
        assert result is True


# =============================================================================
# Text Generation Tests
# =============================================================================

class TestTextGeneration:
    """Tests for text generation."""
    
    @pytest.mark.asyncio
    async def test_generate_success(self, ollama_manager, mock_client) -> None:
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Generated text"}
        mock_client.post.return_value = mock_response
        
        ollama_manager._client = mock_client
        
        result = await ollama_manager.generate("model", "prompt")
        
        assert result == "Generated text"
    
    @pytest.mark.asyncio
    async def test_query_model_uses_active(
        self, ollama_manager, mock_client, tmp_path, monkeypatch
    ) -> None:
        """Test query_model uses active model."""
        # Set active model
        config_path = tmp_path / "config" / "active_model.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"active_model": "default-model"}))
        monkeypatch.chdir(tmp_path)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Response"}
        mock_client.post.return_value = mock_response
        
        ollama_manager._client = mock_client
        
        result = await ollama_manager.query_model("Hello")
        
        assert result == "Response"
    
    @pytest.mark.asyncio
    async def test_query_model_no_active_raises(
        self, ollama_manager, tmp_path, monkeypatch
    ) -> None:
        """Test query_model raises when no active model."""
        monkeypatch.chdir(tmp_path)
        
        with pytest.raises(ValueError, match="No model specified"):
            await ollama_manager.query_model("Hello")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_generate_error(self, ollama_manager, mock_client) -> None:
        """Test generation error handling."""
        mock_client.post.side_effect = Exception("Network error")
        
        ollama_manager._client = mock_client
        
        with pytest.raises(Exception):
            await ollama_manager.generate("model", "prompt")
    
    @pytest.mark.asyncio
    async def test_create_model_api_error(
        self, ollama_manager, mock_client, tmp_path
    ) -> None:
        """Test model creation API error handling."""
        gguf_file = tmp_path / "test.gguf"
        gguf_file.write_bytes(b"GGUF")
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_client.post.return_value = mock_response
        
        ollama_manager._client = mock_client
        
        result = await ollama_manager.create_model_from_gguf(
            model_name="test",
            gguf_path=str(gguf_file),
        )
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
