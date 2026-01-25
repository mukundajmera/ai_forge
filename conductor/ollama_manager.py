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
                logger.error(f"Failed to list models: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            models = data.get("models", [])
            logger.info(f"Ollama list_models found {len(models)} models")
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            # Identify if it's a connection error
            if "connect" in str(e).lower():
                 logger.error("Could not connect to Ollama. Is it running on localhost:11434?")
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
            import subprocess
            
            cmd = ["ollama", "create", model_name, "-f", modelfile_path]
            logger.info(f"Creating Ollama model via CLI: {' '.join(cmd)}")
            
            # Run CLI command
            # Using asyncio.to_thread would be better but simple subprocess is blocking.
            # For this verification context, blocking is fine or use run_in_executor.
            
            params = {
                "capture_output": True,
                "text": True,
                "check": False
            }
            
            result = subprocess.run(cmd, **params)
            
            if result.returncode != 0:
                logger.error(f"Failed to create model (CLI): {result.stderr}")
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
    
    # -------------------------------------------------------------------------
    # Active Model Management
    # -------------------------------------------------------------------------
    
    def get_active_model(self) -> Optional[str]:
        """Get the currently active model from config.
        
        Returns:
            Active model name or None.
        """
        config_path = Path("./config/active_model.json")
        
        if config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    data = json.load(f)
                return data.get("active_model")
            except Exception as e:
                logger.warning(f"Could not read active model config: {e}")
        
        return None
    
    def set_active_model(self, model_name: str) -> bool:
        """Set the active model in config.
        
        Args:
            model_name: Name of model to set as active.
            
        Returns:
            True if successful.
        """
        config_path = Path("./config/active_model.json")
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(config_path, "w") as f:
                json.dump({
                    "active_model": model_name,
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
            
            logger.info(f"Set active model to: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set active model: {e}")
            return False
    
    async def create_model_from_gguf(
        self,
        model_name: str,
        gguf_path: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        num_predict: int = 512,
    ) -> bool:
        """Create an Ollama model from a GGUF file.
        
        Generates a Modelfile from template and creates the model.
        
        Args:
            model_name: Name for the new model.
            gguf_path: Path to GGUF file.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            top_p: Top-p sampling.
            num_predict: Max tokens to generate.
            
        Returns:
            True if creation succeeded.
        """
        from pathlib import Path
        
        gguf_file = Path(gguf_path)
        if not gguf_file.exists():
            logger.error(f"GGUF file not found: {gguf_path}")
            return False
        
        # Default system prompt for code assistant
        if not system_prompt:
            system_prompt = (
                "You are an expert coding assistant. "
                "Provide clear, concise, and correct code solutions. "
                "Always explain your reasoning and follow best practices."
            )
        
        # Generate Modelfile content
        modelfile_content = self._generate_modelfile(
            gguf_path=str(gguf_file.absolute()),
            system_prompt=system_prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_predict=num_predict,
        )
        
        # Write Modelfile
        modelfile_path = gguf_file.parent / f"Modelfile.{model_name}"
        modelfile_path.write_text(modelfile_content)
        logger.info(f"Generated Modelfile at: {modelfile_path}")
        
        # Create model
        # Create model
        try:
            client = await self._get_client()
            
            # Use explicit parameters instead of modelfile string for compatibility
            payload = {
                "name": model_name,
                "from": str(gguf_file.absolute()),
                "system": system_prompt,
                "parameters": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "num_predict": num_predict,
                    "stop": ["<|eot_id|>", "<|end_of_text|>"]
                }
            }
            
            response = await client.post(
                "/api/create",
                json=payload,
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to create model: {response.text}")
                return False
            
            logger.info(f"Successfully created model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return False
    
    def _generate_modelfile(
        self,
        gguf_path: str,
        system_prompt: str,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        num_predict: int = 512,
    ) -> str:
        """Generate Modelfile content from template.
        
        Args:
            gguf_path: Path to GGUF file.
            system_prompt: System prompt.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            top_p: Top-p sampling.
            num_predict: Max tokens.
            
        Returns:
            Modelfile content string.
        """
        # Escape quotes in system prompt
        escaped_prompt = system_prompt.replace('"', '\\"')
        
        return f'''# AI Forge Generated Modelfile
# Created: {datetime.now().isoformat()}

FROM {gguf_path}

SYSTEM "{escaped_prompt}"

# Model Parameters
PARAMETER temperature {temperature}
PARAMETER top_k {top_k}
PARAMETER top_p {top_p}
PARAMETER num_predict {num_predict}
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
'''
    
    async def try_start_ollama(self) -> bool:
        """Attempt to start Ollama if not running.
        
        Returns:
            True if Ollama is running after attempt.
        """
        import subprocess
        
        # Check if already running
        if await self.health_check():
            return True
        
        logger.info("Ollama not running, attempting to start...")
        
        try:
            # Try to start Ollama in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            # Wait for startup
            for _ in range(10):
                await asyncio.sleep(1)
                if await self.health_check():
                    logger.info("Ollama started successfully")
                    return True
            
            logger.warning("Ollama failed to start within timeout")
            return False
            
        except FileNotFoundError:
            logger.error(
                "Ollama not installed. Install from: https://ollama.ai"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False
    
    async def query_model(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Query a model with a prompt.
        
        Convenience method that uses active model if not specified.
        
        Args:
            prompt: Input prompt.
            model: Optional model name (uses active if not provided).
            temperature: Sampling temperature.
            
        Returns:
            Generated response.
            
        Raises:
            ValueError: If no model specified and no active model set.
        """
        if not model:
            model = self.get_active_model()
            if not model:
                raise ValueError(
                    "No model specified and no active model set. "
                    "Use set_active_model() or pass model parameter."
                )
        
        return await self.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
        )


# Import for datetime
from datetime import datetime
from pathlib import Path

