# AI Forge User Guide

## Quick Start

### Prerequisites

- **Mac** with Apple Silicon (M1/M2/M3/M4)
- **16GB+ RAM** recommended (8GB minimum)
- **Python 3.11+**
- **Ollama** installed ([ollama.ai](https://ollama.ai))

### Installation

```bash
# Clone the repository
cd /path/to/ai_forge

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Or from requirements.txt
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from ai_forge import __version__; print(f'AI Forge v{__version__}')"
```

---

## Usage Guide

### 1. Data Extraction

Extract training data from your codebase:

```python
from ai_forge.data_pipeline import CodeMiner, RAFTGenerator, DataValidator

# Mine code from your project
miner = CodeMiner("/path/to/your/project")
chunks = miner.extract_all()
print(f"Extracted {len(chunks)} code chunks")

# Generate RAFT training data
generator = RAFTGenerator(chunks)
dataset = generator.generate_dataset(num_samples=1000)

# Validate and clean
validator = DataValidator(dataset)
result = validator.validate_all()

if result.is_valid:
    validator.save_cleaned_data("training_data.json")
```

### 2. Fine-Tuning

Train your model:

```python
from ai_forge.training import TrainingForge, ForgeConfig

# Configure training
config = ForgeConfig(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    use_pissa=True,          # 3-5x faster convergence
    load_in_4bit=True,       # QLoRA for memory efficiency
    num_epochs=3,
    output_dir="./output",
)

# Initialize and train
forge = TrainingForge(config)
forge.load_model()

from datasets import load_dataset
dataset = load_dataset("json", data_files="training_data.json")["train"]

results = forge.train(dataset)
forge.save_model("./output/my_model")
```

### 3. Evaluation

Evaluate your fine-tuned model:

```python
from ai_forge.judge import ModelEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/my_model")
tokenizer = AutoTokenizer.from_pretrained("./output/my_model")

evaluator = ModelEvaluator(model, tokenizer)
results = evaluator.evaluate_all(test_dataset)

print(results.summary())
```

### 4. Export & Deploy

Export to Ollama:

```python
from ai_forge.judge import GGUFExporter, ExportConfig

exporter = GGUFExporter(
    "./output/my_model",
    ExportConfig(quantization="q4_k_m")
)

# Export to GGUF
result = exporter.export()

# Deploy to Ollama
exporter.deploy_to_ollama(result.output_path, "myproject:custom")
```

### 5. Use via API

Start the API server:

```bash
uvicorn ai_forge.conductor.service:app --host 0.0.0.0 --port 8000
```

Chat with your model:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "myproject:custom",
    "messages": [{"role": "user", "content": "Explain the architecture"}]
  }'
```

---

## Autonomous Mode

Use RepoGuardian for autonomous orchestration:

```python
from ai_forge.antigravity_agent import RepoGuardian

guardian = RepoGuardian("/path/to/project")

# Analyze repository
report = await guardian.analyze_repository()
print(report.to_markdown())

# Run full pipeline (if ready)
if report.ready_for_training:
    result = await guardian.run_pipeline()
```

---

## Configuration

### Environment Variables

See `.env.example` for all available options.

### Training Configuration

Key parameters in `ForgeConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `unsloth/Llama-3.2-3B-Instruct` | Base model |
| `use_pissa` | `True` | Use PiSSA initialization |
| `load_in_4bit` | `True` | QLoRA quantization |
| `pissa_rank` | `64` | LoRA rank |
| `num_epochs` | `3` | Training epochs |
| `learning_rate` | `2e-4` | Learning rate |

---

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use smaller model (3B instead of 7B)
- Ensure 4-bit quantization is enabled

### Slow Training

- Verify Apple Silicon is being used
- Check for background processes
- Consider reducing max_seq_length

### Ollama Connection Issues

- Ensure Ollama is running: `ollama serve`
- Check port 11434 is available

---

## Support

For issues and questions, please refer to the documentation or open an issue on the repository.
