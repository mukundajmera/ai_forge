# Complete Implementation Guide: Project-Specific LLM Service

## Table of Contents
1. Quick Start (30 minutes)
2. Data Pipeline Implementation
3. Fine-Tuning Engine
4. Export & Deployment
5. API Service
6. Antigravity Integration

---

## PART 1: QUICK START (30 MINUTES)

### Step 1: Environment Setup

```bash
# Create Python environment
python3.11 -m venv llm_env
source llm_env/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets peft
pip install unsloth-mlx  # Mac-native, much faster than standard Unsloth
pip install ollama fastapi uvicorn pydantic
pip install langchain langchain-community
pip install tqdm click python-dotenv
```

### Step 2: Minimal Fine-Tuning Example

```python
# fine_tune_minimal.py - Start here!

from unsloth import fast_language_model, is_bfloat16_supported
import torch
from peft import LoraConfig
from transformers import TrainingArguments, SFTTrainer
from datasets import load_dataset, Dataset
import pandas as pd
from datetime import datetime

def create_minimal_dataset():
    """Create sample training data"""
    data = {
        "instruction": [
            "What does the authenticate function do?",
            "Explain the API endpoint structure",
            "How is the database connected?"
        ],
        "input": [
            "def authenticate_user(email, password): ...",
            "POST /api/users/{id}/update",
            "db = Database(config['DATABASE_URL'])"
        ],
        "output": [
            "Authenticates user by validating email and password",
            "Updates user with given ID using POST request",
            "Initializes database connection from config"
        ]
    }
    return Dataset.from_dict(data)

def fine_tune_model():
    """Main fine-tuning function"""
    
    # 1. Load base model (3B recommended for Mac)
    print("📦 Loading model...")
    model, tokenizer = fast_language_model.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",  # Fast for Mac
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,  # QLoRA!
    )
    
    # 2. Setup LoRA (use PiSSA for faster training)
    print("⚙️  Configuring LoRA...")
    peft_config = LoraConfig(
        init_lora_weights="pissa",  # NEW: PiSSA (3-5x faster!)
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    
    # 3. Prepare dataset
    print("📊 Loading dataset...")
    dataset = create_minimal_dataset()
    dataset = dataset.train_test_split(test_size=0.2)
    
    # 4. Format for training
    def formatting_func(example):
        return {
            "text": f"""<s>[INST] {example['instruction']}

{example['input']} [/INST] {example['output']} </s>"""
        }
    
    train_dataset = dataset['train'].map(formatting_func)
    eval_dataset = dataset['test'].map(formatting_func)
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=10,
        eval_strategy="steps",
        eval_steps=5,
        save_total_limit=2,
        fp16=True,
        seed=42,
    )
    
    # 6. Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
    )
    
    # 7. Train!
    print("🚀 Starting training...")
    trainer.train()
    
    # 8. Save adapter
    print("💾 Saving model...")
    model.save_pretrained("./output/lora_adapter")
    tokenizer.save_pretrained("./output/lora_adapter")
    
    print("✅ Training complete!")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = fine_tune_model()
```

### Step 3: Test the Model

```bash
# Run fine-tuning
python fine_tune_minimal.py

# Takes ~10-15 minutes on Mac M1/M2
# Expected output:
#   Step 1/3: loss=2.45
#   Step 2/3: loss=1.23
#   Step 3/3: loss=0.98 ✅
```

### Step 4: Deploy to Ollama

```bash
# After training, merge and export
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./output/lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("./output/lora_adapter")

# Merge with base (optional, for standalone model)
# Save merged version
model.save_pretrained("./final_model")

print("✅ Model ready for Ollama!")
EOF

# Create Ollama Modelfile
cat > Modelfile << 'MODELFILE'
FROM ./final_model

SYSTEM """You are an expert assistant for this project.
You understand the codebase, architecture, and best practices.
Provide helpful, detailed explanations with code examples."""

PARAMETER temperature 0.7
PARAMETER top_k 40
MODELFILE

# Create Ollama model
ollama create myproject:custom -f Modelfile

# Test!
ollama run myproject:custom "What's the main architecture?"
```

---

## PART 2: PRODUCTION DATA PIPELINE

### Step 1: Data Collection & Extraction

```python
# data_pipeline.py

import json
import os
from pathlib import Path
import re
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

class ProjectDataExtractor:
    """Extract training data from project files"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.data = []
    
    def extract_from_readme(self) -> List[Dict]:
        """Extract architecture & setup from README"""
        readme_path = self.project_path / "README.md"
        
        if not readme_path.exists():
            return []
        
        with open(readme_path) as f:
            content = f.read()
        
        # Extract sections
        sections = content.split("##")
        extracted = []
        
        for section in sections:
            if not section.strip():
                continue
            
            title = section.split("\n")[0].strip()
            body = "\n".join(section.split("\n")[1:])
            
            # Create Q&A pair
            extracted.append({
                "instruction": f"What does the '{title}' section explain?",
                "input": body[:500],
                "output": f"The {title} section explains: {body[:200]}...",
                "source": "README",
                "metadata": {"section": title}
            })
        
        return extracted
    
    def extract_from_code(self) -> List[Dict]:
        """Extract from code files"""
        extracted = []
        
        for py_file in self.project_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            try:
                with open(py_file) as f:
                    content = f.read()
                
                # Extract docstrings
                import ast
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        docstring = ast.get_docstring(node)
                        
                        if docstring:
                            extracted.append({
                                "instruction": f"Explain this function",
                                "input": f"Function: {node.name}\n{docstring}",
                                "output": docstring,
                                "source": str(py_file),
                                "metadata": {
                                    "type": "function",
                                    "name": node.name,
                                    "lineno": node.lineno
                                }
                            })
            except:
                continue
        
        return extracted
    
    def extract_all(self) -> List[Dict]:
        """Extract all training data"""
        print("📄 Extracting from README...")
        self.data.extend(self.extract_from_readme())
        
        print("🐍 Extracting from code...")
        self.data.extend(self.extract_from_code())
        
        print(f"✅ Extracted {len(self.data)} examples")
        return self.data
    
    def save_dataset(self, output_path: str):
        """Save as training dataset"""
        df = pd.DataFrame(self.data)
        
        # Add quality score
        df['quality_score'] = df.apply(
            lambda x: len(x['output']) / 1000,  # Longer = better
            axis=1
        ).clip(0, 1)
        
        # Save
        df.to_json(output_path, orient='records', indent=2)
        print(f"💾 Saved {len(df)} examples to {output_path}")
        
        return df

# Usage
if __name__ == "__main__":
    extractor = ProjectDataExtractor("/path/to/your/project")
    data = extractor.extract_all()
    extractor.save_dataset("project_training_data.json")
```

### Step 2: Data Validation & Cleaning

```python
# data_validation.py

from datasets import load_dataset
import json

class DataValidator:
    """Validate and clean training data"""
    
    def __init__(self, data_path: str):
        with open(data_path) as f:
            self.data = json.load(f)
    
    def validate(self):
        """Check data quality"""
        errors = []
        
        for i, item in enumerate(self.data):
            # Check required fields
            if 'instruction' not in item or not item['instruction'].strip():
                errors.append(f"Row {i}: Missing instruction")
            
            if 'output' not in item or not item['output'].strip():
                errors.append(f"Row {i}: Missing output")
            
            # Check length
            if len(item['output'].split()) < 5:
                errors.append(f"Row {i}: Output too short")
            
            if len(item['output'].split()) > 2000:
                errors.append(f"Row {i}: Output too long")
        
        if errors:
            print(f"❌ Found {len(errors)} validation errors:")
            for error in errors[:10]:  # Show first 10
                print(f"   {error}")
            return False
        
        print(f"✅ All {len(self.data)} examples validated!")
        return True
    
    def clean_duplicates(self):
        """Remove duplicate examples"""
        seen = set()
        unique = []
        
        for item in self.data:
            key = (item['instruction'], item['output'][:100])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        
        removed = len(self.data) - len(unique)
        print(f"🧹 Removed {removed} duplicates")
        self.data = unique
    
    def filter_by_quality(self, min_score: float = 0.3):
        """Keep only high-quality examples"""
        before = len(self.data)
        
        self.data = [
            d for d in self.data
            if d.get('quality_score', 1.0) >= min_score
        ]
        
        removed = before - len(self.data)
        print(f"📊 Removed {removed} low-quality examples")

# Usage
validator = DataValidator("project_training_data.json")
validator.clean_duplicates()
validator.filter_by_quality(min_score=0.3)
validator.validate()
```

---

## PART 3: ADVANCED FINE-TUNING

### Using DPO for Quality Improvement

```python
# fine_tune_with_dpo.py - Production quality
from unsloth import fast_language_model
from trl import DPOTrainer
from transformers import TrainingArguments
import torch

def create_preference_dataset():
    """Create dataset with preferred vs rejected responses"""
    return {
        "prompt": [
            "Explain authentication",
            "What's the API structure?"
        ],
        "chosen": [
            "Authentication validates user credentials...",
            "The API uses REST with JSON..."
        ],
        "rejected": [
            "auth does stuff",
            "api is for data"
        ]
    }

def fine_tune_with_dpo():
    """Fine-tune with DPO for quality"""
    
    # Load model
    model, tokenizer = fast_language_model.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Prepare DPO dataset
    dpo_dataset = create_preference_dataset()
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./dpo_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=5e-5,
        ),
        beta=0.1,  # KL penalty
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
    )
    
    dpo_trainer.train()
    model.save_pretrained("./output/dpo_model")

if __name__ == "__main__":
    fine_tune_with_dpo()
```

---

## PART 4: API SERVICE

### Complete FastAPI Service

```python
# api_service.py

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Custom LLM Service", version="1.0")

class FineTuneRequest(BaseModel):
    project_name: str
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    epochs: int = 3
    learning_rate: float = 2e-4
    rank: int = 64

class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7

# Store job status
jobs_db = {}

@app.post("/v1/fine-tune")
async def start_fine_tune(
    request: FineTuneRequest,
    data_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Start a fine-tuning job"""
    
    job_id = f"job_{request.project_name}_{datetime.now().timestamp()}"
    
    # Save uploaded file
    data_path = f"/tmp/{job_id}_data.json"
    with open(data_path, "wb") as f:
        f.write(await data_file.read())
    
    # Queue background job
    background_tasks.add_task(
        execute_fine_tune,
        job_id=job_id,
        config=request,
        data_path=data_path
    )
    
    jobs_db[job_id] = {
        "status": "queued",
        "project": request.project_name,
        "created_at": datetime.now().isoformat()
    }
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Fine-tuning job queued. Check /status/{job_id}"
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status"""
    if job_id not in jobs_db:
        return {"error": "Job not found"}
    
    return jobs_db[job_id]

@app.get("/models")
async def list_models():
    """List available models"""
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    return {"models": result.stdout}

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    """Chat with a model (OpenAI-compatible)"""
    
    # Format messages for Ollama
    prompt = ""
    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n"
    
    # Query Ollama
    result = subprocess.run(
        ["ollama", "run", request.model, prompt],
        capture_output=True,
        text=True
    )
    
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": result.stdout
            }
        }]
    }

async def execute_fine_tune(job_id: str, config: FineTuneRequest, data_path: str):
    """Background task: execute fine-tuning"""
    try:
        jobs_db[job_id]["status"] = "training"
        
        # Run fine-tuning script
        logger.info(f"Starting fine-tuning for {job_id}")
        
        # ... call fine_tune_model() ...
        
        jobs_db[job_id]["status"] = "ready"
        jobs_db[job_id]["model_path"] = f"./models/{job_id}"
        
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
        logger.error(f"Fine-tuning failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Usage Examples

```bash
# Start API server
python api_service.py

# In another terminal, submit job
curl -X POST "http://localhost:8000/v1/fine-tune" \
  -F "request={\"project_name\": \"myproject\", \"epochs\": 3}" \
  -F "data_file=@project_data.json"

# Response:
# {
#   "job_id": "job_myproject_1705432000",
#   "status": "queued"
# }

# Check status
curl "http://localhost:8000/status/job_myproject_1705432000"

# Chat with model
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "myproject:custom",
    "messages": [{"role": "user", "content": "Explain the architecture"}]
  }'
```

---

## PART 5: MONITORING & MAINTENANCE

```python
# monitoring.py

import logging
import json
from datetime import datetime
from pathlib import Path

class FineTuningMonitor:
    """Monitor fine-tuning progress"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.log_file = f"/logs/{job_id}.log"
        Path(self.log_file).parent.mkdir(exist_ok=True)
    
    def log_metrics(self, epoch: int, loss: float, accuracy: float):
        """Log training metrics"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        # Alert if something's wrong
        if loss > 10.0:
            self.alert(f"High loss detected: {loss}")
        
        if epoch > 0 and accuracy < 0.5:
            self.alert(f"Low accuracy: {accuracy}")
    
    def alert(self, message: str):
        """Send alert (email, slack, etc)"""
        print(f"⚠️  ALERT: {message}")
        # TODO: Implement actual alerting
    
    def generate_report(self) -> dict:
        """Generate training report"""
        logs = []
        with open(self.log_file) as f:
            for line in f:
                logs.append(json.loads(line))
        
        return {
            "job_id": self.job_id,
            "total_epochs": len(logs),
            "final_loss": logs[-1]["loss"] if logs else None,
            "final_accuracy": logs[-1]["accuracy"] if logs else None,
            "start_time": logs[0]["timestamp"] if logs else None,
            "end_time": logs[-1]["timestamp"] if logs else None,
        }
```

---

## DEPLOYMENT CHECKLIST

```yaml
Pre-Production Checklist:
  Data:
    ☐ Minimum 500 examples
    ☐ All fields validated
    ☐ No data leakage
    ☐ Quality score > 0.6
  
  Model:
    ☐ Perplexity < 3.0
    ☐ Domain accuracy > 75%
    ☐ No hallucinations
    ☐ Latency < 500ms
  
  Code:
    ☐ All imports working
    ☐ Error handling complete
    ☐ Logging configured
    ☐ Tests passing
  
  Deployment:
    ☐ Ollama modelfile ready
    ☐ API endpoints tested
    ☐ Documentation complete
    ☐ Monitoring configured

Production Deployment:
  1. Run final test batch
  2. Create backup of base model
  3. Deploy to Ollama
  4. Test with real queries
  5. Monitor first 24 hours
  6. Scale gradually
```

---

## COST CALCULATOR

```python
# cost_calculator.py

def calculate_cost(scenario: str):
    costs = {
        "local_mac": {
            "hardware": 300,  # monthly amortization
            "electricity": 50,
            "total_monthly": 350
        },
        "aws_cloud": {
            "compute": 2190,  # p3 instance
            "storage": 30,
            "total_monthly": 2220
        },
        "hybrid": {
            "local": 300,
            "cloud": 1500,
            "total_monthly": 1800
        }
    }
    
    if scenario in costs:
        c = costs[scenario]
        print(f"💰 {scenario.upper()} Costs:")
        for key, val in c.items():
            if "total" not in key:
                print(f"   {key}: ${val}/month")
        print(f"\n   TOTAL: ${c['total_monthly']}/month")
    
    print("\n📊 Cost per fine-tune job:")
    print("   Local Mac: $0 (existing hardware)")
    print("   AWS GPU:   $3-5 (depends on time)")
    print("   Per inference: $0.001 (local), $0.01 (cloud)")

if __name__ == "__main__":
    calculate_cost("local_mac")
```

---

## QUICK REFERENCE COMMANDS

```bash
# Setup
python -m venv llm_env && source llm_env/bin/activate
pip install unsloth-mlx transformers peft ollama fastapi

# Training
python fine_tune_minimal.py          # Quick test
python fine_tune_with_dpo.py        # Production quality

# Deployment
ollama create myproject:custom -f Modelfile
ollama run myproject:custom "prompt"

# API
python api_service.py               # Start service
curl http://localhost:8000/models   # List models

# Monitoring
tail -f /logs/job_*.log            # View logs
ps aux | grep ollama               # Check process

# Cleanup
rm -rf ./output ./cache ./models
ollama delete myproject:custom
```

---

## TROUBLESHOOTING

### Common Issues

**Issue**: "CUDA out of memory"
- Solution: Use QLoRA (enabled by default), reduce batch size, use smaller model

**Issue**: "Model not found in Ollama"
- Solution: Check Modelfile path, ensure model is created: `ollama create <name> -f Modelfile`

**Issue**: "Training too slow on Mac"
- Solution: Use Unsloth-MLX instead of standard Unsloth, use 3B model, reduce sequence length

**Issue**: "Model quality is poor"
- Solution: Add more training data (>1000 examples), train longer (5+ epochs), use DPO phase

---

**Total Setup Time**: 1-2 hours
**First Fine-tune**: 30-60 minutes
**Ready for Production**: 1 week (including data prep)

Good luck! 🚀
