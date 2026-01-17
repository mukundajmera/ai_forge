# Configuration Guide

Complete reference for all AI Forge configuration options.

## Configuration Files

| File | Purpose |
|------|---------|
| `config/pissa_config.yaml` | Default training configuration |
| `pyproject.toml` | Project and tool configuration |
| `pytest.ini` | Test runner configuration |

---

## Training Configuration

### Basic Parameters

```yaml
# config/pissa_config.yaml

# Model
model_name: "unsloth/Llama-3.2-3B-Instruct"
max_seq_length: 2048

# Training
num_epochs: 3
learning_rate: 2e-4
batch_size: 4
gradient_accumulation_steps: 4

# Output
output_dir: "./output"
save_steps: 100
```

### PiSSA Configuration

```yaml
# PiSSA-specific settings
use_pissa: true
pissa_rank: 64
pissa_alpha: 128
pissa_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### Quantization Settings

```yaml
# 4-bit quantization
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_use_double_quant: true
```

---

## Hardware Profiles

### 8GB RAM (M1/M2 Base)

```yaml
# Minimal memory configuration
model_name: "unsloth/Llama-3.2-1B-Instruct"  # Smaller model
max_seq_length: 1024                          # Shorter context
batch_size: 1
gradient_accumulation_steps: 16
use_gradient_checkpointing: true
load_in_4bit: true
pissa_rank: 32                                # Lower rank
```

### 16GB RAM (M1/M2)

```yaml
# Standard configuration
model_name: "unsloth/Llama-3.2-3B-Instruct"
max_seq_length: 2048
batch_size: 2
gradient_accumulation_steps: 8
use_gradient_checkpointing: true
load_in_4bit: true
pissa_rank: 64
```

### 32GB RAM (M2/M3 Pro)

```yaml
# High performance configuration
model_name: "unsloth/Llama-3.2-3B-Instruct"
max_seq_length: 4096
batch_size: 8
gradient_accumulation_steps: 2
use_gradient_checkpointing: false
load_in_4bit: true
pissa_rank: 128
```

### 64GB+ RAM (M3 Max/Ultra)

```yaml
# Maximum performance
model_name: "meta-llama/Llama-3.1-8B-Instruct"
max_seq_length: 8192
batch_size: 16
gradient_accumulation_steps: 1
use_gradient_checkpointing: false
load_in_4bit: true
pissa_rank: 256
```

---

## Parameter Reference

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `Llama-3.2-3B` | HuggingFace model ID |
| `max_seq_length` | int | 2048 | Maximum sequence length |
| `dtype` | str | `bfloat16` | Model precision |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | int | 3 | Training epochs |
| `learning_rate` | float | 2e-4 | Learning rate |
| `batch_size` | int | 4 | Per-device batch size |
| `gradient_accumulation_steps` | int | 4 | Gradient accumulation |
| `warmup_ratio` | float | 0.1 | LR warmup ratio |
| `weight_decay` | float | 0.01 | L2 regularization |
| `max_grad_norm` | float | 1.0 | Gradient clipping |

### PiSSA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_pissa` | bool | true | Enable PiSSA init |
| `pissa_rank` | int | 64 | Adapter rank |
| `pissa_alpha` | int | 128 | Scaling factor |
| `pissa_dropout` | float | 0.05 | Dropout rate |
| `pissa_target_modules` | list | [q,k,v,o] | Target layers |

### Memory Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_in_4bit` | bool | true | 4-bit quantization |
| `use_gradient_checkpointing` | bool | true | Memory optimization |
| `use_flash_attention` | bool | true | Flash attention |

### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_steps` | int | 50 | Evaluate every N steps |
| `eval_strategy` | str | `steps` | When to evaluate |
| `metric_for_best_model` | str | `loss` | Selection metric |

### Saving Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `./output` | Output directory |
| `save_steps` | int | 100 | Save every N steps |
| `save_total_limit` | int | 3 | Max checkpoints |
| `save_safetensors` | bool | true | Use safetensors |

---

## Data Pipeline Configuration

### Code Miner

```python
from data_pipeline.miner import CodeMiner

miner = CodeMiner(
    project_path="/path/to/project",
    languages=["python", "javascript"],  # Languages to extract
    min_lines=5,                          # Minimum function size
    max_lines=500,                        # Maximum function size
    include_private=False,                # Include _private methods
    include_tests=False,                  # Include test files
)
```

### RAFT Generator

```python
from data_pipeline.raft_generator import RAFTGenerator

generator = RAFTGenerator(
    chunks=code_chunks,
    num_examples=500,           # Examples to generate
    question_types=[            # Question types
        "explanation",
        "usage",
        "debugging",
    ],
    min_quality_score=0.6,      # Quality threshold
)
```

### Data Validator

```python
from data_pipeline.validator import DataValidator

validator = DataValidator(
    examples=training_examples,
    quality_threshold=0.6,      # Minimum quality
    max_duplicates_pct=0.05,    # Max 5% duplicates
    min_samples=100,            # Minimum samples
)
```

---

## Agent Configuration

### Pipeline Config

```python
from antigravity_agent.repo_guardian import PipelineConfig

config = PipelineConfig(
    # Automation levels
    auto_extract_data=True,
    auto_validate=True,
    auto_train=False,      # Require confirmation
    auto_export=False,
    auto_deploy=False,
    
    # Quality gates
    quality_threshold=0.7,
    min_training_samples=100,
    max_duplicates_pct=0.05,
    
    # Monitoring
    check_interval_hours=1,
    files_changed_threshold=20,
    commits_threshold=50,
)
```

### Monitoring Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `files_changed_threshold` | 20 | Files changed to trigger |
| `commits_threshold` | 50 | Commits to trigger |
| `critical_paths` | src/, lib/ | Important directories |
| `release_tag_pattern` | v*, release/* | Tags to detect |

---

## API Configuration

### Service Config

```python
from pydantic_settings import BaseSettings


class ServiceConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 120
    
    # Jobs
    max_concurrent_jobs: int = 2
    job_timeout_hours: int = 4
    
    class Config:
        env_prefix = "AI_FORGE_"
```

### Environment Variables

```bash
# .env file
AI_FORGE_HOST=0.0.0.0
AI_FORGE_PORT=8000
AI_FORGE_DEBUG=false
AI_FORGE_OLLAMA_HOST=http://localhost:11434
AI_FORGE_OLLAMA_TIMEOUT=120
```

---

## Performance Tuning Tips

### Training Speed

1. **Increase batch size** if RAM allows
2. **Reduce gradient accumulation** for faster updates
3. **Use Flash Attention** for longer sequences
4. **Disable logging** for maximum throughput

### Memory Optimization

1. **Enable gradient checkpointing** for large models
2. **Use 4-bit quantization** always
3. **Reduce sequence length** if possible
4. **Lower PiSSA rank** (32 instead of 64)

### Quality Optimization

1. **Increase epochs** (3 → 5)
2. **Higher PiSSA rank** (64 → 128)
3. **Lower learning rate** (2e-4 → 1e-4)
4. **More data** (500+ examples)

---

## Next Steps

- [User Guide](user_guide.md) - Step-by-step tutorials
- [Developer Guide](developer_guide.md) - Extending the system
- [Troubleshooting](troubleshooting.md) - Common issues
