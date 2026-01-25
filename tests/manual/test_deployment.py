import sys
import os
import shutil
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

# Import GGUFExporter directly
try:
    from judge.exporter import GGUFExporter, ExportConfig
except ImportError as e:
    print(f"FAIL: Could not import GGUFExporter: {e}")
    sys.exit(1)

# Copy-paste OllamaManager to avoid importing conductor package which triggers fastapi import error
logger = logging.getLogger(__name__)

@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    timeout: int = 300
    health_check_interval: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

class OllamaManager:
    """Manager for Ollama model lifecycle, modified for verification script."""
    
    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or OllamaConfig()
        self._client: Optional[Any] = None
        self._is_connected = False
        
    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.config.host,
                    timeout=self.config.timeout,
                )
            except ImportError:
                print("httpx not installed, trying aiohttp")
                import aiohttp
                self._client = aiohttp.ClientSession(
                    base_url=self.config.host,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                )
        return self._client
    
    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            if hasattr(client, 'get'):
                response = await client.get("/")
            else: # aiohttp
                 async with client.get("/") as response:
                     self._is_connected = response.status == 200
                     return self._is_connected
            self._is_connected = response.status_code == 200
            return self._is_connected
        except Exception as e:
            print(f"Ollama check failed: {e}")
            self._is_connected = False
            return False
            
    async def try_start_ollama(self) -> bool:
        import subprocess
        if await self.health_check():
            return True
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(10):
                await asyncio.sleep(1)
                if await self.health_check():
                    return True
            return False
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            client = await self._get_client()
            if hasattr(client, 'get'):
                response = await client.get("/api/tags")
                data = response.json()
            else:
                async with client.get("/api/tags") as response:
                    data = await response.json()
            return data.get("models", [])
        except Exception:
            return []

    async def create_model(self, model_name: str, modelfile_path: str) -> bool:
        try:
            with open(modelfile_path) as f:
                content = f.read()
            client = await self._get_client()
            if hasattr(client, 'post'):
                response = await client.post("/api/create", json={"name": model_name, "modelfile": content})
                return response.status_code == 200
            else:
                 async with client.post("/api/create", json={"name": model_name, "modelfile": content}) as response:
                     return response.status == 200
        except Exception as e:
            print(f"Create model failed: {e}")
            return False

    async def delete_model(self, model_name: str) -> bool:
        try:
            client = await self._get_client()
             # delete API
            if hasattr(client, 'delete'):
                response = await client.delete("/api/delete", json={"name": model_name})
                return response.status_code == 200
            else:
                async with client.delete("/api/delete", json={"name": model_name}) as response:
                    return response.status == 200
        except Exception:
            return False

    async def chat(self, model: str, messages: list) -> str:
        try:
            client = await self._get_client()
            req = {"model": model, "messages": messages, "stream": False}
            if hasattr(client, 'post'):
                response = await client.post("/api/chat", json=req)
                data = response.json()
            else:
                async with client.post("/api/chat", json=req) as response:
                    data = await response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            print(f"Chat failed: {e}")
            return ""

    async def close(self):
        if self._client:
            if hasattr(self._client, 'aclose'):
                await self._client.aclose()
            elif hasattr(self._client, 'close'):
                await self._client.close()

# Fake Job ID
job_id = "test_deploy_job"

# Setup output directory from dryrun result
dryrun_dir = Path("./output/dryrun")
target_dir = Path(f"./output/{job_id}/final")
export_dir = target_dir / "export"

if not dryrun_dir.exists():
    print("FAIL: Dry run output not found. Run dryrun first.")
    sys.exit(1)

if target_dir.exists():
    shutil.rmtree(target_dir)

try:
    target_dir.mkdir(parents=True)
    
    # Copy files
    print(f"Copying model from {dryrun_dir} to {target_dir}")
    for item in dryrun_dir.iterdir():
        if item.is_file():
            shutil.copy(item, target_dir)
        elif item.is_dir():
             # Avoid copying export dir if it exists inside
            if item.name != "export":
                shutil.copytree(item, target_dir / item.name)
            
    print("Files copied.")
except Exception as e:
    print(f"FAIL: Error setting up test files: {e}")
    sys.exit(1)

async def test_deploy():
    manager = OllamaManager()
    
    try:
        # Check Ollama
        running = await manager.try_start_ollama()
        if not running:
            print("FAIL: Ollama is not running")
            sys.exit(1)
            
        print("Testing Deployment Logic (Direct)...")
        
        # 1. Export GGUF
        print("Exporting to GGUF...")
        model_name = f"test-gpt2-{int(datetime.now().timestamp())}"
        
        # Use q8_0 for safety/speed with llama.cpp
        config = ExportConfig(
            quantization="q8_0", 
            output_dir=str(export_dir),
            model_name=model_name
        )
        
        # Pass output_dir as path where we want export to happen?
        # GGUFExporter output_dir config is used if output_path in export() is not provided.
        # But we initialize it with model path.
        exporter = GGUFExporter(target_dir, config)
        
        result = exporter.export()
        
        if not result.success:
            print(f"FAIL: Export failed: {result.error}")
            sys.exit(1)
            
        print(f"Export successful: {result.output_path}")
        
        # 2. Deploy to Ollama
        print("Deploying to Ollama...")
        # Create Modelfile
        modelfile_path = exporter.create_modelfile(
            result.output_path,
            system_prompt="You are a test assistant."
        )
        
        # Create Model
        print(f"Creating model {model_name} from {modelfile_path}")
        success = await manager.create_model(
            model_name=model_name,
            modelfile_path=str(modelfile_path)
        )
        
        if not success:
            print("FAIL: Failed to create Ollama model")
            sys.exit(1)
            
        print(f"Model {model_name} deployed.")
        
        # 3. Verify List
        models = await manager.list_models()
        found = any(m['name'] == f"{model_name}:latest" for m in models)
        if not found:
             print(f"FAIL: Model {model_name} not found in list")
             sys.exit(1)
             
        # 4. Chat
        print("Testing Chat...")
        response = await manager.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(f"Chat response: {response}")
        
        if not response:
             print("FAIL: Emply chat response")
             sys.exit(1)
             
        # Cleanup
        print("Cleanup...")
        await manager.delete_model(model_name)
        print("PASS: Deployment verified")

    except Exception as e:
        print(f"FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(test_deploy())
