
import sys
import httpx
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def verify_ollama():
    """Verify Ollama integration."""
    print("--- Verifying Ollama Integration ---")
    
    # 1. Check direct Ollama connection
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                print(f"✅ [Direct] Ollama reachable. Found {len(models)} models.")
                # print(f"    Models: {[m['name'] for m in models]}")
            else:
                print(f"❌ [Direct] Ollama reachable but returned {resp.status_code}")
                return False
    except Exception as e:
        print(f"❌ [Direct] Failed to connect to Ollama at localhost:11434. Is it running?")
        print(f"    Error: {e}")
        return False

    # 2. Check AI Forge API Endpoint
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8000/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                print(f"✅ [API] /v1/models reachable. Found {len(models)} models.")
                
                # Check structure
                if len(models) > 0:
                    if "id" not in models[0]:
                         print(f"❌ [API] Model object missing 'id' field: {models[0]}")
                         return False
                return True
            else:
                print(f"❌ [API] /v1/models returned {resp.status_code}")
                return False
    except Exception as e:
        print(f"❌ [API] Failed to connect to AI Forge API at localhost:8000")
        print(f"    Error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(verify_ollama())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        sys.exit(1)
