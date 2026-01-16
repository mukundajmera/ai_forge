# AI Forge Architecture

## System Overview

AI Forge is a production-grade Local LLM Fine-Tuning Service designed for Mac Apple Silicon. It provides end-to-end capabilities from data extraction to model deployment.

## Architecture Diagram

```mermaid
graph TB
    subgraph User Layer
        UI[User / Antigravity Agent]
    end
    
    subgraph API Layer
        API[FastAPI Service<br>conductor/service.py]
        JQ[Job Queue<br>conductor/job_queue.py]
    end
    
    subgraph Core Layer
        DP[Data Pipeline<br>data_pipeline/]
        TE[Training Engine<br>training/forge.py]
        JE[Judge / Evaluator<br>judge/]
    end
    
    subgraph Integration Layer
        OM[Ollama Manager<br>conductor/ollama_manager.py]
        AG[Repo Guardian<br>antigravity_agent/]
    end
    
    subgraph External
        OL[Ollama Server]
        HF[HuggingFace Models]
    end
    
    UI --> API
    API --> JQ
    JQ --> DP
    JQ --> TE
    TE --> JE
    JE --> OM
    OM --> OL
    TE --> HF
    AG --> DP
    AG --> TE
    AG --> JE
```

## Module Responsibilities

### Data Pipeline (`data_pipeline/`)

| Component | Responsibility |
|-----------|---------------|
| `miner.py` | Tree-sitter AST-based code extraction |
| `raft_generator.py` | RAFT training data synthesis |
| `validator.py` | Data quality validation |
| `schemas/` | Pydantic data models |

### Training Engine (`training/`)

| Component | Responsibility |
|-----------|---------------|
| `forge.py` | Main training orchestrator (PiSSA + QLoRA) |
| `callbacks/` | Training callbacks (metrics, early stopping, memory) |
| `losses/` | Custom loss functions (DPO, RAFT) |

### Judge (`judge/`)

| Component | Responsibility |
|-----------|---------------|
| `evaluator.py` | Multi-metric evaluation |
| `exporter.py` | GGUF conversion pipeline |
| `benchmarks/` | Benchmark suite definitions |

### Conductor (`conductor/`)

| Component | Responsibility |
|-----------|---------------|
| `service.py` | FastAPI REST endpoints |
| `ollama_manager.py` | Ollama lifecycle management |
| `job_queue.py` | Async job management |

### Antigravity Agent (`antigravity_agent/`)

| Component | Responsibility |
|-----------|---------------|
| `repo_guardian.py` | Autonomous pipeline orchestration |
| `skills.yaml` | Agent skill definitions |
| `artifacts_templates/` | Report templates |

## Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant D as DataPipeline
    participant T as TrainingForge
    participant E as Evaluator
    participant O as Ollama
    
    U->>A: POST /fine-tune
    A->>D: Extract training data
    D->>D: Tree-sitter parsing
    D->>D: RAFT synthesis
    D->>D: Validation
    D->>T: Training data
    T->>T: Load base model
    T->>T: Configure PiSSA + QLoRA
    T->>T: Train with callbacks
    T->>E: Trained model
    E->>E: Evaluate metrics
    E->>E: Export to GGUF
    E->>O: Deploy model
    O->>U: Model ready
```

## Key Design Decisions

1. **PiSSA over LoRA**: 3-5x faster convergence, +5.16% accuracy
2. **QLoRA (4-bit)**: 75% memory reduction for Mac
3. **Tree-sitter**: Semantic chunking prevents mid-function splits
4. **RAFT**: Hybrid RAG+FT for robust domain adaptation
5. **Ollama**: Simplest local deployment via GGUF
