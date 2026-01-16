"""Ollama Manager - Ollama lifecycle and interaction management.

This module provides comprehensive Ollama integration for model
management, inference, and health monitoring.

Example:
    >>> manager = OllamaManager()
    >>> await manager.create_model("mymodel", "/path/to/modelfile")
    >>> response = await manager.chat("mymodel", [{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for OllamaManager.
    
    Attributes:
        host: Ollama API host.
        timeout: Request timeout in seconds.
        health_check_interval: Health check interval in seconds.
        retry_attempts: Number of retry attempts.
        retry_delay: Delay between retries in seconds.
    """
    
    host: str = "http://localhost:11434"
    timeout: int = 300
    health_check_interval: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class ModelInfo:
    """Information about an Ollama model.
    
    Attributes:
        name: Model name.
        size: Model size in bytes.
        modified_at: Last modification time.
        digest: Model digest.
    """
    
    name: str
    size: int
    modified_at: str
    digest: str


class OllamaManager:
    """Manager for Ollama model lifecycle and inference.
    
    Handles all interactions with Ollama including model
    creation, deletion, inference, and health monitoring.
    
    Attributes:
        config: Ollama configuration.
        
    Example:
        >>> manager = OllamaManager()
        >>> models = await manager.list_models()
        >>> response = await manager.generate("llama3:latest", "Hello!")
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        """Initialize OllamaManager.
        
        Args:
            config: Ollama configuration.
        """
        self.config = config or OllamaConfig()
        self._client: Optional[Any] = None
        self._is_connected = False
        
        logger.info(f"Initialized OllamaManager with host: {self.config.host}")
    
    async def _get_client(self) -> Any:
        """Get or create HTTP client.
        
        Returns:
            HTTP client instance.
        """
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.config.host,
                    timeout=self.config.timeout,
                )
            except ImportError:
                logger.warning("httpx not installed, falling back to aiohttp")
                import aiohttp
                self._client = aiohttp.ClientSession(
                    base_url=self.config.host,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                )
        
        return self._client
    
    async def health_check(self) -> bool:
        """Check if Ollama is running and healthy.
        
        Returns:
            True if Ollama is healthy.
        """
        try:
            client = await self._get_client()
            response = await client.get("/")
            self._is_connected = response.status_code == 200
            return self._is_connected
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            self._is_connected = False
            return False
    
    async def list_models(self) -> list[dict[str, Any]]:
        """List all available models.
        
        Returns:
            List of model information dictionaries.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            
            if response.status_code != 200:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
            
            data = response.json()
            return data.get("models", [])
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library.
        
        Args:
            model_name: Name of model to pull.
            
        Returns:
            True if pull succeeded.
        """
        try:
            client = await self._get_client()
            
            response = await client.post(
                "/api/pull",
                json={"name": model_name},
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to pull model: {response.status_code}")
                return False
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    async def create_model(
        self,
        model_name: str,
        modelfile_path: str,
    ) -> bool:
        """Create a model from a Modelfile.
        
        Args:
            model_name: Name for the new model.
            modelfile_path: Path to Modelfile.
            
        Returns:
            True if creation succeeded.
        """
        try:
            with open(modelfile_path) as f:
                modelfile_content = f.read()
            
            client = await self._get_client()
            
            response = await client.post(
                "/api/create",
                json={
                    "name": model_name,
                    "modelfile": modelfile_content,
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to create model: {response.status_code}")
                return False
            
            logger.info(f"Successfully created model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return False
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model.
        
        Args:
            model_name: Name of model to delete.
            
        Returns:
            True if deletion succeeded.
        """
        try:
            client = await self._get_client()
            
            response = await client.delete(
                "/api/delete",
                json={"name": model_name},
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to delete model: {response.status_code}")
                return False
            
            logger.info(f"Successfully deleted model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            model: Model name.
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text.
        """
        try:
            client = await self._get_client()
            
            request_data: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            }
            
            if max_tokens:
                request_data["options"]["num_predict"] = max_tokens
            
            response = await client.post(
                "/api/generate",
                json=request_data,
            )
            
            if response.status_code != 200:
                raise Exception(f"Generation failed: {response.status_code}")
            
            data = response.json()
            return data.get("response", "")
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """Chat with a model.
        
        Args:
            model: Model name.
            messages: List of chat messages.
            temperature: Sampling temperature.
            
        Returns:
            Assistant's response.
        """
        try:
            client = await self._get_client()
            
            response = await client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                },
            )
            
            if response.status_code != 200:
                raise Exception(f"Chat failed: {response.status_code}")
            
            data = response.json()
            return data.get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
    
    async def get_model_info(self, model_name: str) -> Optional[dict[str, Any]]:
        """Get detailed model information.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Model information or None if not found.
        """
        try:
            client = await self._get_client()
            
            response = await client.post(
                "/api/show",
                json={"name": model_name},
            )
            
            if response.status_code != 200:
                return None
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None
    
    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
