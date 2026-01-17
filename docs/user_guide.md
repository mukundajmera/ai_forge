# User Guide

Step-by-step guide for using AI Forge to fine-tune and deploy models.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Fine-Tuning a Model](#fine-tuning-a-model)
4. [Deploying to Ollama](#deploying-to-ollama)
5. [Querying Your Model](#querying-your-model)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Mac | M1 | M2/M3 Pro/Max |
| RAM | 16GB | 32GB+ |
| Storage | 20GB free | 50GB+ free |

### Software Requirements

- macOS 13+ (Ventura or later)
- Python 3.11+
- Homebrew
- Git

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/ai-forge/ai-forge.git
cd ai-forge
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -e ".[dev]"
```

### Step 4: Install Ollama

```bash
brew install ollama
```

### Step 5: Verify Installation

```bash
python -c "import ai_forge; print('✅ AI Forge installed')"
ollama --version  # Should show version
```

---

## Fine-Tuning a Model

### Option A: Using the API

#### 1. Start the Service

```bash
python -m conductor.service
```

#### 2. Trigger Fine-Tuning

```bash
curl -X POST http://localhost:8000/v1/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "project_path": "/path/to/your/project",
    "auto_deploy": true,
    "force": true
  }'
```

#### 3. Monitor Progress

```bash
# Check status
curl http://localhost:8000/status/{job_id}

# Or watch continuously
watch -n 5 "curl -s http://localhost:8000/status/{job_id} | jq"
```

### Option B: Using Python

```python
from antigravity_agent.repo_guardian import RepoGuardian, PipelineConfig

# Configure
config = PipelineConfig(
    auto_train=True,
    auto_deploy=True,
    quality_threshold=0.7,
)

# Initialize Guardian
guardian = RepoGuardian("/path/to/project", config)

# Run pipeline
import asyncio
result = asyncio.run(guardian.run_pipeline())

print(f"Success: {result['success']}")
```

### Option C: Using CLI (Coming Soon)

```bash
ai-forge train --project /path/to/project --deploy
```

---

## Deploying to Ollama

### Automatic Deployment

If `auto_deploy=True` in your request, the model is automatically deployed after training.

### Manual Deployment

#### 1. Deploy Trained Model

```bash
curl -X POST http://localhost:8000/deploy/{job_id}
```

#### 2. Verify Deployment

```bash
ollama list  # Should show your model
```

#### 3. Test Model

```bash
ollama run ai-forge-project:latest "Hello, how are you?"
```

---

## Querying Your Model

### Using curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-forge-project:latest",
    "messages": [
      {"role": "user", "content": "Explain the DataProcessor class"}
    ]
  }'
```

### Using Python

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ai-forge-project:latest",
        "messages": [{"role": "user", "content": "Hello!"}],
    },
)

print(response.json()["choices"][0]["message"]["content"])
```

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="ai-forge-project:latest",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

### Using Ollama Directly

```bash
ollama run ai-forge-project:latest "Your question here"
```

---

## Workflow Examples

### Example 1: Train on Your Codebase

```bash
# 1. Start service
python -m conductor.service &

# 2. Extract data and train
curl -X POST http://localhost:8000/v1/retrain \
  -d '{"project_path": ".", "force": true, "auto_deploy": true}'

# 3. Wait for completion (check status)
curl http://localhost:8000/status/agent_20240117_120000

# 4. Query your model
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Explain main.py"}]}'
```

### Example 2: Custom Training Configuration

```python
from training.forge import TrainingForge, FineTuneConfig

config = FineTuneConfig(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    num_epochs=5,
    learning_rate=1e-4,
    pissa_rank=128,
    batch_size=4,
    use_gradient_checkpointing=True,  # For limited RAM
)

forge = TrainingForge(config)
forge.load_model()

# Load your data
from datasets import load_dataset
dataset = load_dataset("json", data_files="data/training.json")["train"]

# Train
results = forge.train(dataset)
print(f"Final loss: {results['train_loss']:.4f}")

# Save
forge.save_model("./output/my-model")
```

### Example 3: Evaluate Before Deploying

```bash
# 1. Train without auto-deploy
curl -X POST http://localhost:8000/v1/retrain \
  -d '{"project_path": ".", "force": true, "auto_deploy": false}'

# 2. Run validation
curl -X POST http://localhost:8000/validate/{job_id}

# 3. Check metrics
cat ./output/{job_id}/validation_report.md

# 4. If satisfied, deploy
curl -X POST http://localhost:8000/deploy/{job_id}
```

---

## Next Steps

- [Configuration Guide](configuration.md) - Tune for your hardware
- [Developer Guide](developer_guide.md) - Extend the system
- [API Reference](api_reference.md) - Full API documentation
- [Troubleshooting](troubleshooting.md) - Common issues
