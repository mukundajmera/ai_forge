# AI Forge API Examples

## Base URL
```
http://localhost:8000
```

---

## Health Check

```bash
# Check service health
curl http://localhost:8000/health
```

Response:
```json
{"status": "healthy", "version": "1.0.0", "ollama_connected": true}
```

---

## Fine-Tuning

### Start Fine-Tuning Job

```bash
# Start a fine-tuning job with data file
curl -X POST http://localhost:8000/v1/fine-tune \
  -F "project_name=my-model" \
  -F "base_model=unsloth/Llama-3.2-3B-Instruct" \
  -F "epochs=3" \
  -F "learning_rate=0.0002" \
  -F "rank=64" \
  -F "data_file=@training_data.json"
```

Response:
```json
{
  "job_id": "job_my-model_20240117_120000",
  "status": "queued",
  "message": "Fine-tuning job queued. Check status at /v1/fine-tune/job_my-model_20240117_120000"
}
```

### Check Job Status

```bash
# Get job status
curl http://localhost:8000/v1/fine-tune/job_my-model_20240117_120000

# Alternative (alias)
curl http://localhost:8000/status/job_my-model_20240117_120000
```

Response:
```json
{
  "job_id": "job_my-model_20240117_120000",
  "status": "training",
  "progress": 45.5,
  "current_epoch": 2,
  "loss": 0.523,
  "created_at": "2024-01-17T12:00:00",
  "updated_at": "2024-01-17T12:15:00"
}
```

### List All Jobs

```bash
curl http://localhost:8000/v1/fine-tune
```

### Cancel Job

```bash
curl -X DELETE http://localhost:8000/v1/fine-tune/job_my-model_20240117_120000
```

---

## Chat Completions (OpenAI Compatible)

```bash
# OpenAI-compatible chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to calculate factorial."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Response:
```json
{
  "id": "chatcmpl-1705500000.0",
  "object": "chat.completion",
  "created": 1705500000,
  "model": "llama3.2:3b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    },
    "finish_reason": "stop"
  }]
}
```

---

## Simple Query

```bash
# Simple prompt-response
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the difference between a list and a tuple in Python.",
    "model": "llama3.2:3b"
  }'
```

Response:
```json
{
  "answer": "Lists are mutable, tuples are immutable...",
  "model": "llama3.2:3b",
  "metadata": {"timestamp": "2024-01-17T12:00:00"}
}
```

---

## List Models

```bash
curl http://localhost:8000/v1/models
```

Response:
```json
{
  "data": [
    {"id": "llama3.2:3b", "object": "model", "owned_by": "ai_forge", "created": 1705500000},
    {"id": "codellama:7b", "object": "model", "owned_by": "ai_forge", "created": 1705500000}
  ]
}
```

---

## Deploy Model

```bash
# Deploy completed job to Ollama
curl -X POST http://localhost:8000/deploy/job_my-model_20240117_120000 \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-custom-model",
    "quantization": "q4_k_m",
    "system_prompt": "You are a specialized coding assistant."
  }'
```

Response:
```json
{
  "success": true,
  "model_name": "my-custom-model",
  "message": "Model deployed as 'my-custom-model'. Use with: ollama run my-custom-model"
}
```

---

## Validate Model

```bash
# Run validation suite
curl -X POST http://localhost:8000/validate/job_my-model_20240117_120000
```

Response:
```json
{
  "job_id": "job_my-model_20240117_120000",
  "passed": true,
  "metrics": {
    "final_loss": 0.42,
    "epochs_completed": 3
  },
  "report_path": "./output/job_my-model_20240117_120000/final/validation_report.md"
}
```

---

## OpenAPI Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

---

## Running the Server

```bash
# Start the server
cd /Users/mukundajmera/pocs/ai_forge
uvicorn conductor.service:app --host 0.0.0.0 --port 8000 --reload

# Or with Python directly
python -m conductor.service
```
