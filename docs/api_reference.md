# API Reference

Complete documentation of all AI Forge REST API endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for local usage.

---

## Health & Status

### GET /health

Check service health.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-17T12:00:00Z",
  "ollama_available": true
}
```

### GET /models

List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "ai-forge-project:latest",
      "size": 1800000000,
      "modified_at": "2024-01-17T12:00:00Z"
    }
  ],
  "active_model": "ai-forge-project:latest"
}
```

---

## Fine-Tuning

### POST /v1/fine-tune

Start a fine-tuning job.

**Request:**
```json
{
  "project_name": "my-project",
  "base_model": "unsloth/Llama-3.2-3B-Instruct",
  "epochs": 3,
  "learning_rate": 0.0002,
  "rank": 64,
  "use_pissa": true
}
```

**Response:**
```json
{
  "job_id": "job_20240117_120000",
  "status": "queued",
  "message": "Fine-tuning job queued"
}
```

### GET /status/{job_id}

Get job status.

**Response:**
```json
{
  "job_id": "job_20240117_120000",
  "status": "training",
  "progress": 45.5,
  "current_epoch": 2,
  "current_step": 150,
  "loss": 1.234,
  "created_at": "2024-01-17T12:00:00Z",
  "updated_at": "2024-01-17T12:15:00Z"
}
```

**Status Values:**
| Status | Description |
|--------|-------------|
| `queued` | Waiting to start |
| `preparing` | Loading model |
| `training` | Training in progress |
| `completed` | Successfully finished |
| `failed` | Error occurred |

---

## Chat Completions (OpenAI Compatible)

### POST /v1/chat/completions

OpenAI-compatible chat endpoint.

**Request:**
```json
{
  "model": "ai-forge-project:latest",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain this function."}
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1705485600,
  "model": "ai-forge-project:latest",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This function..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 100,
    "total_tokens": 125
  }
}
```

---

## Query

### POST /v1/query

Simple query endpoint.

**Request:**
```json
{
  "prompt": "What does the DataProcessor class do?",
  "model": "ai-forge-project:latest",
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "The DataProcessor class...",
  "model": "ai-forge-project:latest",
  "tokens_used": 150
}
```

---

## Deployment

### POST /deploy/{job_id}

Deploy a trained model to Ollama.

**Response:**
```json
{
  "job_id": "job_20240117_120000",
  "success": true,
  "model_name": "ai-forge-project:latest",
  "gguf_path": "./export/model.gguf"
}
```

### POST /validate/{job_id}

Run validation on a trained model.

**Response:**
```json
{
  "job_id": "job_20240117_120000",
  "passed": true,
  "metrics": {
    "final_loss": 1.23,
    "perplexity": 3.42
  },
  "report_path": "./output/validation_report.md"
}
```

---

## Agent / Retrain

### POST /v1/retrain

Trigger retraining via Repo Guardian.

**Request:**
```json
{
  "project_path": ".",
  "auto_deploy": true,
  "force": false
}
```

**Response:**
```json
{
  "triggered": true,
  "reason": "20 files changed (threshold: 20)",
  "plan": {
    "plan": [
      {"id": "1", "name": "extract_data", "estimated_minutes": 5},
      {"id": "2", "name": "validate_data", "estimated_minutes": 2}
    ],
    "estimated_duration_minutes": 47
  },
  "job_id": "agent_20240117_120000"
}
```

### GET /v1/retrain/monitor

Check repository for changes.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `project_path` | string | `.` | Path to repository |

**Response:**
```json
{
  "should_retrain": true,
  "reason": "20 files changed",
  "metrics": {
    "commits_since_last_train": 15,
    "files_changed": 23,
    "critical_paths_changed": true
  },
  "checked_at": "2024-01-17T12:00:00Z"
}
```

### POST /v1/retrain/{job_id}/pause

Pause a running pipeline.

### POST /v1/retrain/{job_id}/resume

Resume a paused pipeline.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

### Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found (job, model) |
| 500 | Internal server error |
| 503 | Service unavailable (Ollama not running) |

---

## OpenAI Compatibility

The following OpenAI endpoints are supported:

| OpenAI Endpoint | AI Forge Endpoint |
|-----------------|-------------------|
| `POST /v1/chat/completions` | ✅ Supported |
| `GET /v1/models` | ✅ Mapped to /models |
| `POST /v1/completions` | ❌ Not supported |
| `POST /v1/embeddings` | ❌ Not supported |

### Usage with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="ai-forge-project:latest",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

---

## Rate Limits

No rate limits for local usage. For production deployments, implement rate limiting at the reverse proxy level.

---

## WebSocket (Future)

Real-time training updates via WebSocket are planned for future releases.

```
ws://localhost:8000/ws/training/{job_id}
```
