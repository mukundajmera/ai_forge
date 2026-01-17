# AI Forge User Guide

A comprehensive guide to using AI Forge for local LLM fine-tuning.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Uploading Training Data](#uploading-training-data)
4. [Starting a Fine-Tune](#starting-a-fine-tune)
5. [Monitoring Training](#monitoring-training)
6. [Evaluating & Deploying Models](#evaluating--deploying-models)
7. [Working with Missions](#working-with-missions)
8. [Keyboard Shortcuts](#keyboard-shortcuts)
9. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Launch

1. Open AI Forge at `http://localhost:5173`
2. You'll see the Dashboard with system status
3. Check that the backend is connected (green indicator in header)

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| GPU | Apple M1 / 4GB VRAM | Apple M2 Pro / 8GB+ VRAM |
| Storage | 20GB free | 50GB+ free |
| Node.js | 18+ | 20+ |

### Prerequisites Checklist

- [ ] AI Forge backend running at `http://localhost:8000`
- [ ] Ollama installed and running
- [ ] At least one base model pulled (`ollama pull llama3.2:3b`)

---

## Dashboard Overview

The Dashboard is your command center for AI Forge.

### System Resources Panel

Real-time monitoring of your machine:

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| CPU | Processor utilization | < 80% |
| Memory | RAM usage | < 85% |
| GPU | Graphics memory used | < 90% |
| Ollama | Model server status | Running |

> **⚠️ Warning**: Don't start training if memory is > 85%. Close other applications first.

### Active Model Card

Shows the currently deployed model:

- **Model Name**: e.g., `myproject:v2`
- **Base Model**: e.g., `Llama-3.2-3B`
- **Status**: Active / Candidate / Deploying
- **Metrics**: CodeBLEU, HumanEval, Perplexity

### Quick Actions

| Button | Action |
|--------|--------|
| **New Fine-Tune** | Start training wizard |
| **Upload Data** | Go to data ingestion |
| **View Jobs** | See all training runs |

---

## Uploading Training Data

### Supported File Formats

**Code Files:**
- Python: `.py`, `.pyi`
- JavaScript/TypeScript: `.js`, `.jsx`, `.ts`, `.tsx`
- Java: `.java`
- Go: `.go`
- Rust: `.rs`
- C/C++: `.c`, `.cpp`, `.h`, `.hpp`

**Documentation:**
- Markdown: `.md`, `.mdx`
- Plain text: `.txt`
- PDF: `.pdf` (text extraction)

### Upload Methods

#### 1. File Upload (Drag & Drop)

1. Navigate to **Data** → **Add Data Source**
2. Select **Upload Files**
3. Drag files into the upload zone or click to browse
4. Wait for upload to complete

#### 2. Git Repository

1. Navigate to **Data** → **Add Data Source**
2. Select **Git Repository**
3. Enter repository URL
4. (Optional) Specify branch and subdirectory
5. Click **Clone & Parse**

#### 3. Local Folder

1. Navigate to **Data** → **Add Data Source**
2. Select **Local Folder**
3. Enter absolute path to folder
4. Configure include/exclude patterns
5. Click **Scan & Parse**

### Configuring Filters

Use glob patterns to include/exclude files:

```
# Include only Python files in src/
include: src/**/*.py

# Exclude test files
exclude: **/*test*.py, **/tests/**
```

### Understanding Parse Results

After parsing, review the quality metrics:

| Metric | Good | Needs Attention |
|--------|------|-----------------|
| Parse Status | ✓ All success | ⚠️ Some failures |
| Quality Score | > 0.7 | < 0.5 |
| Chunks Extracted | 100+ | < 50 |

### Generating Training Dataset

1. After successful parsing, click **Generate Dataset**
2. Configure options:
   - **Dataset Name**: Descriptive name (e.g., `myproject-docs-v1`)
   - **Format**: Alpaca (default)
   - **Max Examples**: 500-2000 recommended
3. Click **Generate**
4. Wait for RAFT synthesis (1-5 minutes)
5. Review quality metrics before training

---

## Starting a Fine-Tune

### Training Wizard

#### Step 1: Project Configuration

1. Click **New Fine-Tune** from Dashboard or Jobs page
2. Enter **Project Name** (alphanumeric, no spaces)
3. Select a **Dataset** with quality > 0.7
4. Click **Next**

#### Step 2: Model & Training Config

**Base Model Selection:**

| Model | Size | RAM Required | Training Time | Quality |
|-------|------|--------------|---------------|---------|
| Llama-3.2-3B | 3B | 6-8 GB | Fast | Good |
| Qwen2.5-Coder-7B | 7B | 10-14 GB | Medium | Better |
| Llama-3.2-7B | 7B | 12-16 GB | Medium | Better |
| Codestral-22B | 22B | 24+ GB | Slow | Best |

**Training Presets:**

| Preset | Epochs | Time | Use Case |
|--------|--------|------|----------|
| **Quick** | 1 | 15-20 min | Testing, iteration |
| **Balanced** | 3 | 45-60 min | Production (recommended) |
| **Thorough** | 5 | 2-3 hours | Maximum quality |

**Advanced Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning Rate | 2e-4 | Higher = faster but less stable |
| LoRA Rank | 64 | Higher = more capacity, more memory |
| Batch Size | 4 | Higher = faster, more memory |
| Gradient Accumulation | 4 | Simulates larger batch size |

#### Step 3: Review & Start

1. Verify all settings
2. Note estimated training time
3. Review resource requirements
4. Click **Start Training**

---

## Monitoring Training

### Job Detail Page

When training starts, you're redirected to the Job Detail page.

### Live Metrics

| Metric | Description | What to Watch |
|--------|-------------|---------------|
| **Loss** | Training error | Should decrease |
| **Progress** | % complete | Steady increase |
| **Epoch** | Data pass number | 1/3, 2/3, 3/3 |
| **Step** | Current batch | 450/1000 |
| **ETA** | Time remaining | Updates live |

### Loss Curve Chart

The loss curve shows training progress:

- **Good Training**: Smooth downward curve
- **Overfitting**: Loss increases after initial decrease
- **Unstable**: Jagged, erratic curve

### Viewing Logs

1. Click the **Logs** tab
2. Logs stream in real-time
3. Use **Search** to filter by keyword
4. Click **Pause** to stop auto-scroll
5. Click **Download** to save full log

### Error Recovery

If training fails:

1. Read the error message in the red panel
2. Common solutions:
   - **OOM (Out of Memory)**: Reduce batch size or use smaller model
   - **CUDA Error**: Restart Ollama service
   - **Dataset Error**: Re-check dataset quality
3. Click **Retry with Recommendations** if available

---

## Evaluating & Deploying Models

### After Training Completes

1. Status changes to **Completed** (green)
2. **Evaluate & Deploy** button appears
3. Click to open evaluation dialog

### Metric Comparison

Compare new model vs. currently active model:

| Metric | Description | Better |
|--------|-------------|--------|
| **CodeBLEU** | Code similarity score | Higher ↑ |
| **HumanEval Pass@1** | Correctness rate | Higher ↑ |
| **Perplexity** | Language fluency | Lower ↓ |
| **Avg Latency** | Response time (ms) | Lower ↓ |

### Decision Indicators

- ✅ **Green Arrow**: Metric improved
- ❌ **Red Arrow**: Metric regressed
- ➡️ **Gray Arrow**: No significant change

### Deployment Options

| Option | Description |
|--------|-------------|
| **Deploy & Activate** | Export to GGUF, deploy to Ollama, set as active |
| **Deploy Only** | Export and deploy, keep previous model active |
| **Cancel** | Keep trained model as candidate |

### Post-Deployment

After deployment:

1. Model appears in **Models** page
2. Test with: `ollama run yourproject:latest`
3. Previous model remains available for rollback

---

## Working with Missions

### What are Missions?

Missions are AI-generated suggestions from the Repo Guardian agent:

| Type | Trigger | Action |
|------|---------|--------|
| **Retrain** | New commits detected | Start training with new data |
| **Deploy** | Candidate model ready | Deploy model to production |
| **Alert** | Performance drift | Investigate quality issues |

### Viewing Pending Missions

1. Navigate to **Missions** page
2. Pending missions show yellow badge
3. Click to view details

### Mission Detail Page

| Section | Contents |
|---------|----------|
| **Summary** | Title, description, priority |
| **Confidence** | Agent's certainty (0-100%) |
| **Agent Analysis** | Trigger, reasoning, expected outcome |
| **Recommended Action** | What the agent suggests |
| **Artifacts** | Charts, reports, logs |

### Approving Missions

1. Review all information
2. Click **Approve** (green button)
3. Agent executes recommended action
4. You're redirected to relevant page (e.g., Jobs)

### Rejecting Missions

1. Click **Reject** (red button)
2. **Provide a reason** (required)
   - Helps agent learn and improve
   - Examples: "Not enough new data", "Waiting for review"
3. Click **Confirm Rejection**

### Mission Artifacts

Missions may include:

| Artifact Type | Description |
|---------------|-------------|
| **Performance Chart** | Metrics over time |
| **Code Diff** | Changed files summary |
| **Quality Report** | Dataset/model analysis |
| **Log Excerpt** | Relevant log entries |

Click **View** or **Download** to access.

---

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| `⌘/Ctrl + K` | Open command palette / search |
| `⌘/Ctrl + N` | New Fine-Tune |
| `⌘/Ctrl + D` | Go to Dashboard |
| `⌘/Ctrl + J` | Go to Jobs |
| `⌘/Ctrl + M` | Go to Missions |
| `⌘/Ctrl + ?` | Show keyboard shortcuts |

### Page-Specific Shortcuts

| Page | Shortcut | Action |
|------|----------|--------|
| Jobs List | `F` | Toggle filter panel |
| Jobs List | `/` | Focus search |
| Job Detail | `L` | Toggle logs panel |
| Mission Detail | `A` | Approve mission |
| Mission Detail | `R` | Open reject dialog |

---

## Troubleshooting

### Common Issues

#### Upload fails with "File too large"

**Solution**: Files are limited to 10MB each. Split large files or exclude from upload.

#### Parsing takes forever

**Solutions**:
- Reduce number of files (use exclude patterns)
- Exclude binary files and images
- Check backend logs for errors

#### Training runs out of memory (OOM)

**Solutions**:
1. Reduce batch size: 4 → 2 → 1
2. Reduce LoRA rank: 128 → 64 → 32
3. Use smaller base model
4. Close other applications

#### Loss not decreasing

**Solutions**:
1. Increase learning rate slightly (2e-4 → 5e-4)
2. Check dataset quality (should be > 0.7)
3. Try more epochs
4. Verify dataset isn't too small (< 100 examples)

#### Model deployment fails

**Solutions**:
1. Check Ollama is running: `ollama serve`
2. Verify enough disk space (10GB+ free)
3. Check backend logs for export errors

#### Backend connection lost

**Solutions**:
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check CORS settings in backend
3. Verify `VITE_API_BASE_URL` in `.env`

### Getting Help

- **Documentation**: Check this guide and [Architecture](ARCHITECTURE.md)
- **GitHub Issues**: Report bugs with reproduction steps
- **Discussions**: Ask questions in GitHub Discussions
- **Logs**: Check browser console and backend logs

---

## Best Practices

### Data Quality

- ✅ Use 500+ training examples
- ✅ Aim for quality score > 0.7
- ✅ Include diverse examples
- ❌ Don't include auto-generated code
- ❌ Don't include test files (usually)

### Training

- ✅ Start with 3B model for testing
- ✅ Use "Balanced" preset for production
- ✅ Monitor loss curve for anomalies
- ❌ Don't interrupt training mid-epoch
- ❌ Don't train on < 85% memory available

### Deployment

- ✅ Compare metrics before deploying
- ✅ Test model manually before activating
- ✅ Keep previous version for rollback
- ❌ Don't deploy if any metric significantly regressed

### Missions

- ✅ Review agent reasoning carefully
- ✅ Provide quality feedback when rejecting
- ✅ Trust high-confidence (> 80%) suggestions
- ❌ Don't ignore warning-level missions
