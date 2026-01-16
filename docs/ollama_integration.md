# Ollama Integration Guide

This document explains how AI Forge integrates with Ollama for local model serving.

## Overview

AI Forge uses Ollama to serve fine-tuned models locally. After training, models are exported to GGUF format and deployed to Ollama for fast inference.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │───▶│  OllamaManager   │───▶│    Ollama       │
│   Service       │    │                  │    │   (localhost)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Active Model    │
                       │  Config (JSON)   │
                       └──────────────────┘
```

## OllamaManager Methods

### Health & Status

```python
from conductor.ollama_manager import OllamaManager

manager = OllamaManager()

# Check if Ollama is running
is_running = await manager.health_check()

# Try to start Ollama if not running
started = await manager.try_start_ollama()

# List available models
models = await manager.list_models()
```

### Model Management

```python
# Create model from GGUF file
success = await manager.create_model_from_gguf(
    model_name="my-model",
    gguf_path="/path/to/model.gguf",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
)

# Delete a model
deleted = await manager.delete_model("my-model")

# Get model info
info = await manager.get_model_info("my-model")
```

### Active Model

```python
# Set the active model (used when no model specified)
manager.set_active_model("my-model")

# Get current active model
active = manager.get_active_model()
```

### Inference

```python
# Generate text (specify model)
response = await manager.generate(
    model="my-model",
    prompt="Write a Python function",
    temperature=0.7,
)

# Query using active model
response = await manager.query_model("Write a Python function")

# Chat with history
response = await manager.chat(
    model="my-model",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ],
)
```

## Modelfile Generation

When creating a model from GGUF, AI Forge generates a Modelfile:

```
FROM /path/to/model.gguf

SYSTEM "You are an expert coding assistant..."

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_predict 512
PARAMETER stop "<|eot_id|>"
```

## System Prompts

### Default Code Assistant
```
You are an expert coding assistant. Provide clear, concise, 
and correct code solutions. Always explain your reasoning 
and follow best practices.
```

### Custom Project-Specific
```python
await manager.create_model_from_gguf(
    model_name="django-expert",
    gguf_path="/path/to/model.gguf",
    system_prompt="You are an expert in Django web development. "
                  "You know the project's codebase deeply and can "
                  "suggest improvements specific to this project.",
)
```

## Error Handling

### Ollama Not Installed

```python
try:
    await manager.try_start_ollama()
except Exception as e:
    print("Install Ollama from: https://ollama.ai")
```

### Ollama Not Running

```python
if not await manager.health_check():
    success = await manager.try_start_ollama()
    if not success:
        print("Please start Ollama manually: ollama serve")
```

### GGUF File Missing

```python
result = await manager.create_model_from_gguf(
    model_name="test",
    gguf_path="/nonexistent.gguf",
)
if not result:
    print("GGUF file not found")
```

## FastAPI Integration

The OllamaManager is integrated into the FastAPI service:

```python
# In service.py
@app.post("/deploy/{job_id}")
async def deploy_model(job_id: str):
    # Export to GGUF
    gguf_path = export_to_gguf(job_id)
    
    # Create Ollama model
    success = await state.ollama_manager.create_model_from_gguf(
        model_name=f"ai-forge-{job_id}",
        gguf_path=gguf_path,
    )
    
    # Set as active
    if success:
        state.ollama_manager.set_active_model(f"ai-forge-{job_id}")
```

## Configuration

### OllamaConfig

```python
from conductor.ollama_manager import OllamaConfig

config = OllamaConfig(
    host="http://localhost:11434",  # Ollama API host
    timeout=300,                     # Request timeout (seconds)
    health_check_interval=30,        # Health check interval
    retry_attempts=3,                # Retry attempts on failure
    retry_delay=1.0,                 # Delay between retries
)

manager = OllamaManager(config)
```

### Active Model Config

Stored in `./config/active_model.json`:

```json
{
  "active_model": "ai-forge-job_20240117_120000",
  "updated_at": "2024-01-17T12:15:00"
}
```

## Troubleshooting

### "Ollama not available"
1. Check if Ollama is running: `curl http://localhost:11434`
2. Start Ollama: `ollama serve`

### "Model not found"
1. List available models: `ollama list`
2. Pull model: `ollama pull llama3.2:3b`

### "GGUF file not found"
1. Verify training completed successfully
2. Check output directory: `./output/{job_id}/final/`
