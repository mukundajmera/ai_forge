# AI Forge API Reference

Complete API documentation for the AI Forge backend.

---

## Base URL

| Environment | URL |
|-------------|-----|
| Development | `http://localhost:8000` |
| Staging | `https://api.staging.aiforge.dev` |
| Production | `https://api.aiforge.dev` |

---

## Authentication

Currently, AI Forge runs locally without authentication. Future versions will support:

- JWT Bearer tokens
- API keys for programmatic access

---

## Response Format

### Success Response

```json
{
    "data": { ... },
    "meta": {
        "timestamp": "2024-01-15T10:00:00Z"
    }
}
```

### Error Response

```json
{
    "status": 400,
    "message": "Validation failed",
    "detail": {
        "field": "epochs",
        "error": "Must be between 1 and 10"
    }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

---

## Endpoints

### Health & Status

#### GET /health

Check API health status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:00:00Z",
    "checks": {
        "database": true,
        "ollama": true,
        "gpu": true
    }
}
```

---

#### GET /status

Get system resource status.

**Response:**
```json
{
    "healthy": true,
    "version": "1.0.0",
    "uptime": 86400,
    "cpu": {
        "utilization": 45,
        "cores": 8
    },
    "memory": {
        "total": 16000000000,
        "used": 8000000000
    },
    "gpu": {
        "name": "Apple M2 Pro",
        "memoryUsed": 4000000000,
        "memoryTotal": 16000000000,
        "utilization": 30,
        "temperature": 45
    },
    "ollama": {
        "status": "running",
        "version": "0.1.0",
        "modelsLoaded": ["myproject:latest"]
    },
    "runningJobs": 1
}
```

---

### Training Jobs

#### GET /jobs

List all training jobs.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status (queued, running, completed, failed) |
| `limit` | integer | Max jobs to return (default: 50) |
| `offset` | integer | Pagination offset |

**Response:**
```json
[
    {
        "id": "job-1",
        "projectName": "myproject",
        "status": "running",
        "baseModel": "Llama-3.2-3B",
        "method": "pissa",
        "progress": 45,
        "metrics": {
            "loss": 1.234,
            "currentEpoch": 2,
            "totalEpochs": 3,
            "currentStep": 450,
            "totalSteps": 1000
        },
        "config": {
            "epochs": 3,
            "learningRate": 0.0001,
            "rank": 64,
            "batchSize": 4
        },
        "startedAt": "2024-01-15T10:00:00Z",
        "datasetName": "test-dataset"
    }
]
```

---

#### GET /jobs/:jobId

Get job details.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `jobId` | string | Job identifier |

**Response:**
```json
{
    "id": "job-1",
    "projectName": "myproject",
    "status": "running",
    "baseModel": "Llama-3.2-3B",
    "method": "pissa",
    "progress": 45,
    "metrics": {
        "loss": 1.234,
        "currentEpoch": 2,
        "totalEpochs": 3,
        "currentStep": 450,
        "totalSteps": 1000
    },
    "config": {
        "epochs": 3,
        "learningRate": 0.0001,
        "rank": 64,
        "batchSize": 4
    },
    "startedAt": "2024-01-15T10:00:00Z",
    "datasetName": "test-dataset",
    "error": null
}
```

---

#### GET /jobs/:jobId/metrics

Get job training metrics history.

**Response:**
```json
{
    "steps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "losses": [3.2, 2.8, 2.5, 2.2, 1.9, 1.7, 1.5, 1.3, 1.2, 1.1],
    "learningRates": [0.0001, 0.0001, 0.0001, ...],
    "gradNorms": [1.2, 1.1, 0.9, ...]
}
```

---

#### GET /jobs/:jobId/logs

Get job training logs.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `tail` | integer | Number of lines from end (default: 100) |

**Response:**
```json
{
    "logs": [
        "[INFO] Starting training...",
        "[INFO] Epoch 1/3",
        "[INFO] Step 100/1000, Loss: 2.5"
    ],
    "lastUpdated": "2024-01-15T10:05:00Z"
}
```

---

#### POST /v1/fine-tune

Start a new fine-tuning job.

**Request Body:**
```json
{
    "projectName": "myproject",
    "datasetId": "ds-1",
    "baseModel": "Llama-3.2-3B",
    "epochs": 3,
    "learningRate": 0.0001,
    "rank": 64,
    "batchSize": 4,
    "gradientAccumulation": 4,
    "warmupSteps": 100,
    "weightDecay": 0.01
}
```

**Response:**
```json
{
    "jobId": "job-new",
    "status": "queued",
    "message": "Training job queued successfully"
}
```

---

#### DELETE /jobs/:jobId

Cancel a running job.

**Response:**
```json
{
    "message": "Job cancelled"
}
```

---

#### GET /jobs/:jobId/validation

Get model validation results.

**Response:**
```json
{
    "jobId": "job-1",
    "modelId": "model-1",
    "runAt": "2024-01-15T12:00:00Z",
    "metrics": {
        "codebleu": 0.78,
        "humaneval": 0.65,
        "perplexity": 12.3,
        "latency": 45
    },
    "passed": true
}
```

---

#### POST /jobs/:jobId/export

Export trained model to GGUF and deploy to Ollama.

**Request Body:**
```json
{
    "quantization": "q4_k_m",
    "deployToOllama": true
}
```

**Response:**
```json
{
    "exportId": "export-1",
    "status": "completed",
    "outputPath": "/models/myproject.gguf",
    "ollamaName": "myproject:latest"
}
```

---

### Models

#### GET /models

List all models.

**Response:**
```json
[
    {
        "id": "model-1",
        "name": "myproject-v1",
        "baseModel": "Llama-3.2-3B",
        "status": "active",
        "createdAt": "2024-01-14T12:00:00Z",
        "metrics": {
            "codebleu": 0.78,
            "humaneval": 0.65,
            "perplexity": 12.3,
            "avgLatency": 45
        },
        "ollamaName": "myproject:latest"
    }
]
```

---

#### GET /models/active

Get the currently active model.

**Response:**
```json
{
    "id": "model-1",
    "name": "myproject-v1",
    "status": "active",
    ...
}
```

---

#### POST /models/:modelId/deploy

Deploy model to Ollama.

**Response:**
```json
{
    "ollamaName": "myproject:v1",
    "message": "Deployed successfully"
}
```

---

#### POST /models/:modelId/activate

Set model as active.

**Response:**
```json
{
    "message": "Activated"
}
```

---

#### POST /models/:modelId/rollback

Rollback to a previous model version.

**Response:**
```json
{
    "message": "Rolled back"
}
```

---

### Datasets

#### GET /datasets

List all training datasets.

**Response:**
```json
[
    {
        "id": "ds-1",
        "name": "test-dataset",
        "exampleCount": 1500,
        "format": "alpaca",
        "createdAt": "2024-01-13T14:00:00Z",
        "sourceIds": ["src-1"],
        "quality": 0.82
    }
]
```

---

#### GET /datasets/:datasetId

Get dataset details.

**Response:**
```json
{
    "id": "ds-1",
    "name": "test-dataset",
    "exampleCount": 1500,
    "format": "alpaca",
    "createdAt": "2024-01-13T14:00:00Z",
    "sourceIds": ["src-1"],
    "quality": 0.82,
    "preview": [
        {
            "instruction": "Explain the function...",
            "input": "def calculate_sum(a, b)...",
            "output": "This function takes two parameters..."
        }
    ]
}
```

---

#### POST /datasets/generate

Generate a training dataset from data sources using RAFT.

**Request Body:**
```json
{
    "name": "my-dataset",
    "sourceIds": ["src-1", "src-2"],
    "format": "alpaca",
    "maxExamples": 1000
}
```

**Response:**
```json
{
    "datasetId": "ds-new",
    "status": "generating"
}
```

---

### Data Sources

#### GET /data-sources

List all data sources.

**Response:**
```json
[
    {
        "id": "src-1",
        "name": "my-repo",
        "type": "git",
        "status": "synced",
        "fileCount": 50,
        "lastSyncedAt": "2024-01-15T08:00:00Z"
    }
]
```

---

#### POST /data-sources

Create a new data source (file upload).

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `files` | File[] | Files to upload |
| `name` | string | Data source name |
| `includePatterns` | string | Glob patterns to include |
| `excludePatterns` | string | Glob patterns to exclude |

**Response:**
```json
{
    "sourceId": "src-new",
    "parsingJobId": "parse-1"
}
```

---

#### DELETE /data-sources/:sourceId

Delete a data source.

**Response:**
```json
{
    "message": "Deleted"
}
```

---

#### POST /data-sources/:sourceId/sync

Trigger a sync/refresh of the data source.

**Response:**
```json
{
    "jobId": "parse-2",
    "message": "Sync started"
}
```

---

### Missions

#### GET /missions

List all missions.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status (pending_approval, approved, rejected) |
| `type` | string | Filter by type (retrain, deploy, alert) |

**Response:**
```json
{
    "missions": [
        {
            "id": "mission-1",
            "title": "Recommend retrain for myproject",
            "description": "New commits detected...",
            "status": "pending_approval",
            "type": "retrain",
            "priority": "medium",
            "confidence": 85,
            "createdAt": "2024-01-15T09:00:00Z"
        }
    ],
    "total": 1,
    "pending": 1
}
```

---

#### GET /missions/:missionId

Get mission details.

**Response:**
```json
{
    "id": "mission-1",
    "title": "Recommend retrain for myproject",
    "description": "New commits detected. Training on updated data could improve model quality.",
    "status": "pending_approval",
    "type": "retrain",
    "priority": "medium",
    "confidence": 85,
    "createdAt": "2024-01-15T09:00:00Z",
    "reasoning": {
        "trigger": "New commits detected",
        "analysis": "Detected 15 new commits since last training",
        "expectedOutcome": "~5% improvement in accuracy"
    },
    "recommendedAction": {
        "type": "start_training",
        "parameters": {
            "datasetId": "ds-1",
            "epochs": 2
        }
    },
    "artifacts": [
        {
            "id": "art-1",
            "type": "chart",
            "name": "Performance Trend",
            "url": "/artifacts/art-1"
        }
    ],
    "relatedJobIds": ["job-1"],
    "relatedModelIds": ["model-1"],
    "relatedDatasetIds": ["ds-1"]
}
```

---

#### POST /missions/:missionId/approve

Approve a pending mission.

**Request Body:**
```json
{
    "comment": "Looks good, proceeding with retrain."
}
```

**Response:**
```json
{
    "message": "Approved"
}
```

---

#### POST /missions/:missionId/reject

Reject a pending mission.

**Request Body:**
```json
{
    "reason": "Not enough new data to justify retraining."
}
```

**Response:**
```json
{
    "message": "Rejected"
}
```

---

### Artifacts

#### GET /artifacts/:artifactId

Get artifact metadata.

**Response:**
```json
{
    "id": "art-1",
    "type": "chart",
    "name": "Performance Trend",
    "mimeType": "application/json",
    "size": 1024
}
```

---

#### GET /artifacts/:artifactId/content

Get artifact content.

**Response:** Varies by artifact type

---

## Rate Limits

Currently no rate limits for local development. Production will have:

| Endpoint | Limit |
|----------|-------|
| GET endpoints | 100 req/min |
| POST endpoints | 50 req/min |
| File uploads | 10 req/min |

---

## Webhooks (Future)

Future versions will support webhooks for:

- Job status changes
- Mission creation
- Deployment events

---

## SDKs

We plan to provide SDKs for:

- Python (`pip install aiforge`)
- JavaScript (`npm install @aiforge/client`)
- CLI (`aiforge train myproject`)
