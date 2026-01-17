# Troubleshooting Guide

Solutions to common issues with AI Forge.

## Quick Diagnostics

Run this to check your setup:

```bash
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'MPS available: {torch.backends.mps.is_available()}')
except: print('❌ PyTorch not installed')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except: print('❌ Transformers not installed')

import subprocess
result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
print(f'Ollama: {result.stdout.strip() if result.returncode == 0 else \"not found\"}')
"
```

---

## Installation Issues

### Issue: "No module named 'mlx'"

**Cause:** MLX not installed or wrong Python version.

**Solution:**
```bash
# Ensure Python 3.11+
python --version

# Reinstall
pip install mlx mlx-lm
```

### Issue: "tree-sitter build fails"

**Cause:** Missing Xcode tools.

**Solution:**
```bash
xcode-select --install
pip install tree-sitter --force-reinstall
```

### Issue: "torch not compiled with MPS"

**Cause:** CPU-only PyTorch installed.

**Solution:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

---

## Ollama Issues

### Issue: "Connection refused to localhost:11434"

**Cause:** Ollama not running.

**Solution:**
```bash
# Start Ollama
ollama serve

# Or check if already running
pgrep ollama
```

### Issue: "Model not found"

**Cause:** Model not pulled or wrong name.

**Solution:**
```bash
# List models
ollama list

# Pull model if missing
ollama pull llama3.2:3b
```

### Issue: "GGUF creation fails"

**Cause:** Invalid model path or insufficient disk space.

**Solution:**
```bash
# Check disk space
df -h

# Verify model files exist
ls -la ./output/final/

# Try manual export
python -m judge.exporter ./output/final --output model.gguf
```

---

## Training Issues

### Issue: "CUDA out of memory" (on Mac shows MPS error)

**Cause:** Model too large for available memory.

**Solutions:**

```python
# Option 1: Reduce batch size
config = FineTuneConfig(
    batch_size=1,
    gradient_accumulation_steps=16,
)

# Option 2: Enable gradient checkpointing
config = FineTuneConfig(
    use_gradient_checkpointing=True,
)

# Option 3: Use smaller model
config = FineTuneConfig(
    model_name="unsloth/Llama-3.2-1B-Instruct",
)

# Option 4: Lower rank
config = FineTuneConfig(
    pissa_rank=32,  # Instead of 64
)
```

### Issue: "Loss is NaN"

**Cause:** Learning rate too high or data issues.

**Solutions:**

```python
# Lower learning rate
config = FineTuneConfig(
    learning_rate=1e-5,  # Instead of 2e-4
)

# Check data for issues
from data_pipeline.validator import DataValidator
validator = DataValidator(data)
result = validator.validate_all()
print(f"Quality: {result.quality_score}")
```

### Issue: "Training not improving"

**Cause:** Learning rate too low, insufficient data, or too few epochs.

**Solutions:**

1. **Increase learning rate:** `2e-4` → `5e-4`
2. **Add more data:** Aim for 500+ examples
3. **Increase epochs:** `3` → `5`
4. **Check data quality:** Ensure score > 0.7

### Issue: "Training is very slow"

**Cause:** Suboptimal configuration.

**Solutions:**

```python
# Increase batch size if memory allows
config = FineTuneConfig(
    batch_size=8,
    gradient_accumulation_steps=2,
)

# Disable unnecessary logging
config = FineTuneConfig(
    logging_steps=100,  # Instead of 10
)
```

---

## Data Pipeline Issues

### Issue: "No code chunks extracted"

**Cause:** Wrong directory or file patterns.

**Solutions:**

```python
# Check if files exist
from pathlib import Path
py_files = list(Path(".").rglob("*.py"))
print(f"Found {len(py_files)} Python files")

# Verify miner config
from data_pipeline.miner import CodeMiner
miner = CodeMiner(".", languages=["python"])
chunks = miner.extract_all()
print(f"Extracted {len(chunks)} chunks")
```

### Issue: "Quality score too low"

**Cause:** Poor quality training data.

**Solutions:**

```python
# Check quality distribution
from data_pipeline.validator import DataValidator
validator = DataValidator(examples)
result = validator.validate_all()

print(f"Total: {result.total_samples}")
print(f"High quality: {result.high_quality_count}")
print(f"Low quality: {result.low_quality_count}")

# Filter low quality
high_quality = [ex for ex in examples if ex.get("quality_score", 0) > 0.7]
```

### Issue: "Too many duplicates"

**Cause:** Repeated code patterns or over-generation.

**Solution:**

```python
# Deduplicate
from data_pipeline.validator import DataValidator
validator = DataValidator(examples)
deduplicated = validator.remove_duplicates()
print(f"After dedup: {len(deduplicated)} examples")
```

---

## API Issues

### Issue: "503 Service Unavailable"

**Cause:** Ollama not responding.

**Solution:**

```bash
# Check Ollama health
curl http://localhost:11434

# Restart if needed
pkill ollama
ollama serve
```

### Issue: "Job stuck in 'queued'"

**Cause:** Worker not processing jobs.

**Solution:**

```bash
# Check logs
tail -f ai_forge.log

# Restart service
pkill uvicorn
python -m conductor.service
```

### Issue: "Timeout errors"

**Cause:** Long operations timing out.

**Solution:**

```python
# Increase timeout
import httpx
client = httpx.Client(timeout=300)  # 5 minutes
```

---

## Performance Issues

### Issue: "High memory usage"

**Monitor:**
```bash
# Watch memory
while true; do
    ps aux | grep python | awk '{print $4"%", $11}'
    sleep 5
done
```

**Solutions:**

1. Use 4-bit quantization
2. Enable gradient checkpointing
3. Reduce batch size
4. Use smaller model

### Issue: "Slow inference"

**Solutions:**

1. Check if running on MPS: `torch.device("mps")`
2. Use smaller quantization (Q4 instead of Q8)
3. Reduce max_tokens
4. Monitor Ollama resources

---

## Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger("ai_forge").setLevel(logging.DEBUG)
logging.getLogger("training").setLevel(logging.DEBUG)
```

---

## Getting Help

### Check Logs

```bash
# Service logs
tail -f ai_forge.log

# Training logs
cat ./output/{job_id}/training.log
```

### Run Diagnostics

```bash
pytest tests/unit -v --tb=long
```

### Report Issues

Include:
1. macOS version
2. Python version
3. Error message
4. Reproduction steps
5. Configuration used

---

## Next Steps

- [Configuration](configuration.md) - Tune settings
- [User Guide](user_guide.md) - Tutorials
- [API Reference](api_reference.md) - Endpoint docs
