# AI Forge API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required for local development.

---

## Endpoints

### Health & Status

#### `GET /health`

Check service health.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "ollama_connected": true
}
```

#### `GET /`

Root endpoint with service info.

**Response:**
```json
{
  "service": "AI Forge",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

### Fine-Tuning

#### `POST /v1/fine-tune`

Start a new fine-tuning job.

**Request:**
- `request` (JSON): Fine-tuning configuration
- `data_file` (File): Training data JSON file

**Request Body Schema:**
```json
{
  "project_name": "string",
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
  "job_id": "job_myproject_20260116_123456",
  "status": "queued",
  "message": "Fine-tuning job queued..."
}
```

#### `GET /v1/fine-tune/{job_id}`

Get job status.

**Response:**
```json
{
  "job_id": "job_myproject_20260116_123456",
  "status": "training",
  "progress": 45.5,
  "current_epoch": 2,
  "current_step": 150,
  "loss": 0.456,
  "created_at": "2026-01-16T12:34:56",
  "updated_at": "2026-01-16T12:45:00"
}
```

#### `GET /v1/fine-tune`

List all jobs.

**Response:** Array of job statuses.

#### `DELETE /v1/fine-tune/{job_id}`

Cancel a running job.

---

### Chat (OpenAI-Compatible)

#### `POST /v1/chat/completions`

Chat with a model.

**Request:**
```json
{
  "model": "myproject:custom",
  "messages": [
    {"role": "user", "content": "Explain the architecture"}
  ],
  "temperature": 0.7,
  "max_tokens": 256
}
```

**Response:**
```json
{
  "id": "chatcmpl-123456",
  "object": "chat.completion",
  "created": 1705432000,
  "model": "myproject:custom",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The architecture consists of..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

### Models

#### `GET /v1/models`

List available models.

**Response:**
```json
{
  "data": [
    {
      "id": "llama3:latest",
      "object": "model",
      "owned_by": "ai_forge",
      "created": 1705432000
    }
  ]
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request |
| 404 | Not Found |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## Rate Limits

No rate limits for local deployment.

## Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
