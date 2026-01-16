# Repo Guardian Agent Architecture

## Overview

The Repo Guardian is an autonomous agent for managing the LLM fine-tuning lifecycle. It monitors repositories, detects significant changes, and orchestrates the complete pipeline from data extraction to model deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Mission Control                             │
│  (User commands, scheduled triggers, or automated monitoring)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RepoGuardian                               │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Monitor    │  │  Plan       │  │  Execute                │  │
│  │  Repository │─▶│  Training   │─▶│  Pipeline               │  │
│  │             │  │  Cycle      │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Task Execution Layer                          │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Extract  │  │ Validate │  │ Train    │  │ Export & Deploy  │ │
│  │ Data     │─▶│ Data     │─▶│ Model    │─▶│ to Ollama        │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Artifact Layer                              │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │ Health Report  │  │ Training       │  │ Validation Report  │ │
│  │                │  │ Dashboard      │  │                    │ │
│  └────────────────┘  └────────────────┘  └────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### RepoGuardian Class

The main orchestrator that coordinates all pipeline activities.

```python
from antigravity_agent.repo_guardian import RepoGuardian, PipelineConfig

# Initialize with project path
guardian = RepoGuardian("/path/to/project")

# Monitor for changes
result = guardian.monitor_repository()
if result["should_retrain"]:
    plan = guardian.plan_training_cycle()
    await guardian.run_pipeline()
```

### Task Types

| Task Type | Description |
|-----------|-------------|
| `EXTRACT_DATA` | Extract code chunks from repository |
| `VALIDATE_DATA` | Validate data quality |
| `TRAIN` | Execute PiSSA+QLoRA fine-tuning |
| `VALIDATE_MODEL` | Run evaluation suite |
| `EXPORT` | Convert to GGUF format |
| `DEPLOY` | Deploy to Ollama |

### Artifacts

Artifacts are generated for each major step:

1. **HealthReport** - Repository analysis
2. **TrainingDashboard** - Real-time training metrics
3. **ValidationReport** - Model comparison
4. **DeploymentChecklist** - Deployment verification

## API Endpoints

### POST /v1/retrain

Trigger retraining via the agent.

```bash
curl -X POST http://localhost:8000/v1/retrain \
  -H "Content-Type: application/json" \
  -d '{"project_path": ".", "auto_deploy": true}'
```

### GET /v1/retrain/monitor

Check repository without triggering retraining.

```bash
curl http://localhost:8000/v1/retrain/monitor?project_path=.
```

### POST /v1/retrain/{job_id}/pause

Pause a running pipeline.

### POST /v1/retrain/{job_id}/resume

Resume a paused pipeline.

## Configuration

### PipelineConfig

```python
config = PipelineConfig(
    auto_extract_data=True,   # Extract data automatically
    auto_validate=True,        # Validate data automatically
    auto_train=False,          # Require confirmation for training
    auto_export=False,         # Require confirmation for export
    auto_deploy=False,         # Require confirmation for deploy
    quality_threshold=0.7,     # Minimum data quality score
    min_training_samples=100,  # Minimum training samples
)
```

## Quality Gates

The agent enforces quality gates at each stage:

### Pre-Training
- ≥100 training samples
- Average quality score ≥0.6
- No excessive duplicates

### Post-Training
- Final loss <2.0
- No NaN losses
- Model loads successfully

### Pre-Deployment
- GGUF export successful
- Test query returns valid response
- Latency <500ms

## Usage Examples

### Manual Trigger

```python
guardian = RepoGuardian("/my/project")
await guardian.run_pipeline()
```

### With Configuration

```python
config = PipelineConfig(
    auto_train=True,
    auto_deploy=True,
)
guardian = RepoGuardian("/my/project", config)

# Check if retraining needed
result = guardian.monitor_repository()
if result["should_retrain"]:
    await guardian.run_pipeline()
```

### Via API

```bash
# Force retrain with auto-deploy
curl -X POST http://localhost:8000/v1/retrain \
  -H "Content-Type: application/json" \
  -d '{"project_path": ".", "force": true, "auto_deploy": true}'
```

## Monitoring Thresholds

The agent triggers retraining when:
- 20+ files changed since last training
- Critical paths modified (src/core, lib/, etc.)
- Release tag detected (v*, release/*)
- 50+ commits since last training

## Error Handling

The agent implements graceful error handling:

1. **Retries** - Failed tasks retry up to 3 times
2. **Logging** - All errors logged with context
3. **Artifacts** - Error artifacts generated for debugging
4. **Rollback** - Failed deployments can be rolled back

## Files

| File | Description |
|------|-------------|
| `repo_guardian.py` | Main agent implementation |
| `skills.yaml` | Skill definitions |
| `prompts.py` | Mission control prompts |
| `artifacts_templates/` | Report templates |
