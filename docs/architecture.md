# Architecture

This document describes the system architecture of AI Forge.

## System Overview

AI Forge is designed as a modular pipeline for fine-tuning LLMs on local hardware. Each component is independent and communicates through well-defined interfaces.

```mermaid
graph TB
    subgraph "Data Layer"
        A[Repository] --> B[Code Miner]
        B --> C[RAFT Generator]
        C --> D[Data Validator]
    end
    
    subgraph "Training Layer"
        D --> E[Training Forge]
        E --> F[PiSSA Initializer]
        E --> G[QLoRA Quantizer]
    end
    
    subgraph "Evaluation Layer"
        E --> H[Model Evaluator]
        H --> I[Perplexity]
        H --> J[CodeBLEU]
        H --> K[Hallucination]
    end
    
    subgraph "Deployment Layer"
        H --> L[GGUF Exporter]
        L --> M[Ollama Manager]
        M --> N[Model Serving]
    end
    
    subgraph "Orchestration Layer"
        O[Repo Guardian] --> A
        O --> E
        O --> L
        P[FastAPI Service] --> O
    end
```

## Components

### 1. Data Pipeline (`data_pipeline/`)

**Purpose:** Extract code from repositories and synthesize training data.

```mermaid
graph LR
    A[Source Code] --> B[Tree-sitter Parser]
    B --> C[Code Chunks]
    C --> D[RAFT Generator]
    D --> E[Q/A Pairs]
    E --> F[Validator]
    F --> G[Training Data]
```

| Component | File | Purpose |
|-----------|------|---------|
| CodeMiner | `miner.py` | AST-based code extraction |
| RAFTGenerator | `raft_generator.py` | Training data synthesis |
| DataValidator | `validator.py` | Quality scoring |

**Key Features:**
- Multi-language support (Python, JavaScript, Go)
- Complete function extraction (no mid-function splits)
- Docstring preservation
- Quality heuristics

### 2. Training Engine (`training/`)

**Purpose:** Fine-tune LLMs using PiSSA + QLoRA.

```mermaid
graph TB
    A[Base Model] --> B[4-bit Quantization]
    B --> C[PiSSA SVD Init]
    C --> D[Adapter Injection]
    D --> E[Training Loop]
    E --> F[Checkpoints]
    F --> G[Final Model]
```

| Component | File | Purpose |
|-----------|------|---------|
| TrainingForge | `forge.py` | Main training orchestrator |
| PiSSAInitializer | `pissa.py` | SVD-based initialization |
| Callbacks | `callbacks/` | Metrics, early stopping, plotting |

**PiSSA vs LoRA:**

```
Standard LoRA:
W' = W + BA    (random init)

PiSSA:
W' = W + BA    (SVD-based init from W)

Benefit: 10x faster convergence, better final quality
```

### 3. Evaluation System (`judge/`)

**Purpose:** Evaluate model quality and export for deployment.

| Component | File | Purpose |
|-----------|------|---------|
| ModelEvaluator | `evaluator.py` | Metrics computation |
| GGUFExporter | `exporter.py` | llama.cpp export |
| EvaluationReport | `report.py` | Report generation |

**Metrics:**
- Perplexity (lower is better)
- CodeBLEU (code similarity)
- Hallucination rate (factuality)
- Exact match rate

### 4. API Service (`conductor/`)

**Purpose:** REST API for fine-tuning and inference.

```mermaid
graph LR
    A[Client] --> B[FastAPI]
    B --> C[Job Queue]
    C --> D[Training]
    B --> E[Ollama Manager]
    E --> F[Inference]
```

| Component | File | Purpose |
|-----------|------|---------|
| FastAPI App | `service.py` | REST endpoints |
| OllamaManager | `ollama_manager.py` | Model serving |
| JobQueue | `job_queue.py` | Async job management |

**Endpoints:**
- `POST /v1/fine-tune` - Start training
- `GET /status/{job_id}` - Job status
- `POST /v1/chat/completions` - OpenAI-compatible chat
- `POST /v1/retrain` - Trigger via agent

### 5. Autonomous Agent (`antigravity_agent/`)

**Purpose:** Automated pipeline orchestration.

```mermaid
graph TB
    A[Repo Guardian] --> B{Monitor Repository}
    B -->|Changes Detected| C[Plan Training]
    C --> D[Execute Pipeline]
    D --> E{Quality Gate}
    E -->|Pass| F[Deploy Model]
    E -->|Fail| G[Report Issue]
```

| Component | File | Purpose |
|-----------|------|---------|
| RepoGuardian | `repo_guardian.py` | Orchestrator |
| Skills | `skills.yaml` | Capability definitions |
| Prompts | `prompts.py` | Mission control |

## Data Flow

### Fine-Tuning Request Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Queue
    participant Forge
    participant Ollama
    
    User->>API: POST /v1/fine-tune
    API->>Queue: Enqueue job
    API-->>User: 202 Accepted + job_id
    Queue->>Forge: Execute training
    Forge->>Forge: Load model
    Forge->>Forge: Apply PiSSA + QLoRA
    Forge->>Forge: Train
    Forge->>Forge: Save checkpoint
    Forge-->>Queue: Complete
    User->>API: GET /status/{job_id}
    API-->>User: 200 + progress
```

### Inference Request Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Ollama
    
    User->>API: POST /v1/chat/completions
    API->>Ollama: Forward request
    Ollama->>Ollama: Generate
    Ollama-->>API: Response
    API-->>User: 200 + completion
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| ML Framework | PyTorch, Transformers, PEFT |
| Mac Optimization | MLX, Unsloth |
| API | FastAPI, Uvicorn |
| Serving | Ollama, llama.cpp |
| Parsing | Tree-sitter |
| Testing | pytest, pytest-asyncio |

## Memory Management

```mermaid
graph TB
    A[16GB Mac] --> B{Model Size}
    B -->|3B| C[Full Training]
    B -->|7B| D[Gradient Checkpointing]
    B -->|13B+| E[Not Recommended]
    
    F[32GB Mac] --> G{Model Size}
    G -->|3B| H[Full Training]
    G -->|7B| I[Full Training]
    G -->|13B| J[Gradient Checkpointing]
```

## Comparison to Alternatives

| Feature | AI Forge | Unsloth | LoRAX | Axolotl |
|---------|----------|---------|-------|---------|
| PiSSA | ✅ | ❌ | ❌ | ❌ |
| Mac Native | ✅ | Partial | ❌ | ❌ |
| RAFT Data | ✅ | ❌ | ❌ | ❌ |
| Ollama Deploy | ✅ | ❌ | ❌ | ❌ |
| API Service | ✅ | ❌ | ✅ | ❌ |
| Auto-retrain | ✅ | ❌ | ❌ | ❌ |
