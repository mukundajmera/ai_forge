# Developer Guide

Guide for extending and developing AI Forge.

## Table of Contents

1. [Code Structure](#code-structure)
2. [Development Setup](#development-setup)
3. [Adding New PEFT Methods](#adding-new-peft-methods)
4. [Adding Evaluation Metrics](#adding-evaluation-metrics)
5. [Adding New Languages](#adding-new-languages)
6. [Creating API Endpoints](#creating-api-endpoints)
7. [Testing](#testing)
8. [Contributing](#contributing)

---

## Code Structure

```
ai_forge/
├── config/                 # YAML configuration files
│   └── pissa_config.yaml   # Default training config
├── data_pipeline/          # Data extraction and synthesis
│   ├── miner.py            # AST-based code extraction
│   ├── raft_generator.py   # RAFT data synthesis
│   └── validator.py        # Data quality validation
├── training/               # Training engine
│   ├── forge.py            # Main trainer
│   ├── pissa.py            # PiSSA initialization
│   └── callbacks/          # Training callbacks
├── judge/                  # Evaluation and export
│   ├── evaluator.py        # Metrics computation
│   ├── exporter.py         # GGUF export
│   └── report.py           # Report generation
├── conductor/              # API service
│   ├── service.py          # FastAPI app
│   ├── ollama_manager.py   # Ollama integration
│   └── job_queue.py        # Async job queue
└── antigravity_agent/      # Autonomous orchestration
    ├── repo_guardian.py    # Pipeline automation
    ├── skills.yaml         # Agent capabilities
    └── prompts.py          # Mission control
```

---

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/ai-forge/ai-forge.git
cd ai-forge

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

### 3. Run Tests

```bash
pytest tests/unit -v
```

### 4. Start Development Server

```bash
uvicorn conductor.service:app --reload --port 8000
```

---

## Adding New PEFT Methods

### Step 1: Create the Initializer

Create `training/lora_plus.py`:

```python
"""LoRA+ implementation with adaptive learning rates."""

import torch
from peft import LoraConfig
from typing import Optional


class LoraPlusInitializer:
    """Initialize LoRA+ with adaptive scaling."""
    
    def __init__(
        self,
        rank: int = 64,
        alpha: Optional[int] = None,
        eta_ratio: float = 16.0,
    ):
        self.rank = rank
        self.alpha = alpha or rank
        self.eta_ratio = eta_ratio
    
    def create_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    def apply_to_model(self, model):
        """Apply LoRA+ to model."""
        from peft import get_peft_model
        
        config = self.create_config()
        model = get_peft_model(model, config)
        
        # Apply adaptive learning rates
        self._set_adaptive_lr(model)
        
        return model
    
    def _set_adaptive_lr(self, model):
        """Set different LR for A and B matrices."""
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.lr_scale = 1.0
            elif "lora_B" in name:
                param.lr_scale = self.eta_ratio
```

### Step 2: Integrate with TrainingForge

Update `training/forge.py`:

```python
from training.lora_plus import LoraPlusInitializer

class TrainingForge:
    def _apply_peft(self, model):
        if self.config.peft_method == "pissa":
            # Existing PiSSA logic
            pass
        elif self.config.peft_method == "lora_plus":
            initializer = LoraPlusInitializer(
                rank=self.config.pissa_rank,
                eta_ratio=16.0,
            )
            model = initializer.apply_to_model(model)
        return model
```

### Step 3: Add Configuration

Update `config/pissa_config.yaml`:

```yaml
peft_method: "lora_plus"  # or "pissa"
lora_plus:
  eta_ratio: 16.0
```

### Step 4: Add Tests

Create `tests/unit/test_lora_plus.py`:

```python
import pytest
from training.lora_plus import LoraPlusInitializer


class TestLoraPlusInitializer:
    def test_create_config(self):
        init = LoraPlusInitializer(rank=64)
        config = init.create_config()
        
        assert config.r == 64
        assert config.lora_alpha == 64
```

---

## Adding Evaluation Metrics

### Step 1: Implement the Metric

Add to `judge/evaluator.py`:

```python
def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute BLEU score.
    
    Args:
        predictions: Model predictions.
        references: Reference outputs.
        
    Returns:
        BLEU score (0-1).
    """
    from sacrebleu import corpus_bleu
    
    result = corpus_bleu(predictions, [references])
    return result.score / 100.0
```

### Step 2: Add to EvaluationResult

```python
@dataclass
class EvaluationResult:
    perplexity: float
    codebleu: float
    bleu: float  # New metric
    hallucination_rate: float
```

### Step 3: Update ModelEvaluator

```python
class ModelEvaluator:
    def evaluate(self, dataset) -> EvaluationResult:
        # ... existing code ...
        
        # Add new metric
        bleu = compute_bleu(predictions, references)
        
        return EvaluationResult(
            perplexity=perplexity,
            codebleu=codebleu,
            bleu=bleu,
            hallucination_rate=halluc,
        )
```

### Step 4: Add Tests

```python
def test_compute_bleu():
    from judge.evaluator import compute_bleu
    
    predictions = ["hello world"]
    references = ["hello world"]
    
    score = compute_bleu(predictions, references)
    
    assert score > 0.9
```

---

## Adding New Languages

### Step 1: Install Tree-sitter Grammar

```bash
pip install tree-sitter-rust  # For Rust
```

### Step 2: Update Language Map

In `data_pipeline/miner.py`:

```python
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".go": "go",
    ".rs": "rust",  # New language
}

LANGUAGE_TO_PARSER = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",  # New parser
}
```

### Step 3: Add Query Template

Create `data_pipeline/queries/rust.scm`:

```scheme
; Functions
(function_item
  name: (identifier) @function.name
  body: (block) @function.body) @function

; Structs
(struct_item
  name: (type_identifier) @struct.name) @struct

; Implementations
(impl_item
  type: (type_identifier) @impl.type) @impl
```

### Step 4: Add Tests

```python
def test_rust_parsing():
    from data_pipeline.miner import extract_chunks_from_string
    
    code = """
    fn hello() {
        println!("Hello");
    }
    """
    
    chunks = extract_chunks_from_string(code, "rust")
    
    assert len(chunks) == 1
    assert chunks[0].name == "hello"
```

---

## Creating API Endpoints

### Step 1: Define Pydantic Models

In `conductor/service.py`:

```python
from pydantic import BaseModel


class ExportRequest(BaseModel):
    job_id: str
    format: str = "gguf"
    quantization: str = "q4_k_m"


class ExportResponse(BaseModel):
    success: bool
    file_path: str
    file_size_mb: float
```

### Step 2: Implement Endpoint

```python
@app.post("/v1/export", response_model=ExportResponse)
async def export_model(request: ExportRequest) -> ExportResponse:
    """Export trained model to specified format."""
    
    # Validate job exists
    if request.job_id not in state.jobs:
        raise HTTPException(404, "Job not found")
    
    # Perform export
    from judge.exporter import GGUFExporter
    
    exporter = GGUFExporter()
    result = await exporter.export(
        model_path=f"./output/{request.job_id}",
        quantization=request.quantization,
    )
    
    return ExportResponse(
        success=True,
        file_path=result.output_path,
        file_size_mb=result.size_mb,
    )
```

### Step 3: Add Tests

```python
def test_export_endpoint(client):
    response = client.post(
        "/v1/export",
        json={"job_id": "test_job", "format": "gguf"},
    )
    
    assert response.status_code == 200
    assert response.json()["success"] is True
```

---

## Testing

### Run All Tests

```bash
pytest
```

### Run Unit Tests

```bash
pytest tests/unit -v
```

### Run with Coverage

```bash
pytest --cov=ai_forge --cov-report=html
open htmlcov/index.html
```

### Run Specific Test

```bash
pytest tests/unit/test_miner.py::TestPythonParsing -v
```

---

## Contributing

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/ai-forge.git
cd ai-forge
```

### 2. Create Branch

```bash
git checkout -b feature/my-feature
```

### 3. Make Changes

- Write code
- Add tests
- Update documentation

### 4. Run Checks

```bash
# Format
black .
isort .

# Lint
ruff check .

# Test
pytest

# Pre-commit
pre-commit run --all-files
```

### 5. Submit PR

```bash
git push origin feature/my-feature
# Open PR on GitHub
```

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused
- Add tests for new features

---

## Next Steps

- [Architecture](architecture.md) - System design
- [Configuration](configuration.md) - Config options
- [API Reference](api_reference.md) - Endpoint docs
