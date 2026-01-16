# AI Forge - Mission Control Prompts
# 
# These prompts define the agent's persona and mission examples
# for autonomous fine-tuning lifecycle management.

# =============================================================================
# AGENT PERSONA
# =============================================================================

PERSONA = """
You are a DevOps AI responsible for maintaining the freshness and quality of 
the project's local LLM. Your role is to:

1. Monitor the repository for significant changes
2. Trigger retraining when appropriate
3. Validate model quality before deployment
4. Manage deployments with minimal human oversight

You should be proactive but cautious - always validate before deploying, 
and provide clear reasoning for your decisions.
"""

# =============================================================================
# MISSION CONTROL PROMPTS
# =============================================================================

# Basic monitoring mission
MISSION_MONITOR = """
Agent, monitor the src/ directory for changes. When you detect:
- 20+ files changed, OR
- A git tag matching 'release/*'

Report your findings but do not trigger training automatically.
Generate a HealthReport artifact with your analysis.
"""

# Full training cycle mission
MISSION_FULL_CYCLE = """
Agent, execute the full training cycle:

1. Extract data from recent commits
2. Generate RAFT training examples  
3. Validate data quality (stop if < 0.6 avg quality)
4. Train with PiSSA+QLoRA (display loss curve)
5. Evaluate model (compare vs base)
6. If metrics improve, deploy to Ollama
7. Send summary report

For each step, generate an Artifact showing progress.
Stop immediately if any quality gate fails.
"""

# Scheduled retraining mission
MISSION_SCHEDULED = """
Agent, this is a scheduled weekly retraining run.

1. Analyze repository changes since last training
2. If no significant changes, skip and report
3. If changes detected, run full training cycle
4. Deploy only if metrics show improvement:
   - Perplexity decreased by 5%+
   - CodeBLEU improved by 3%+
   - No hallucination regression

Generate a comprehensive ValidationReport comparing old vs new model.
"""

# Emergency rollback mission
MISSION_ROLLBACK = """
Agent, the current model is experiencing issues. Execute rollback:

1. Identify the previous stable model
2. Restore previous model to Ollama
3. Set previous model as active
4. Generate incident report with:
   - What went wrong
   - Which model was rolled back to
   - Recommended actions

Notify me immediately when rollback is complete.
"""

# Data quality focus mission  
MISSION_DATA_QUALITY = """
Agent, focus on data quality improvement:

1. Analyze current training data
2. Identify low-quality samples (score < 0.5)
3. Generate report showing:
   - Quality distribution histogram
   - Top 10 worst samples with reasons
   - Recommendations for improvement

4. If sufficient high-quality data exists (>500 samples with score > 0.7):
   - Suggest filtering parameters
   - Estimate training improvement

Do NOT start training - this is analysis only.
"""

# Model comparison mission
MISSION_COMPARE = """
Agent, compare the current fine-tuned model against the base model:

1. Run evaluation suite on both models
2. Generate comparison table with:
   - Perplexity (lower is better)
   - CodeBLEU (higher is better)
   - Response latency
   - Memory usage

3. Run 5 sample queries through both models
4. Present side-by-side responses

Recommend whether to keep the fine-tuned model or revert to base.
"""

# =============================================================================
# QUALITY GATES
# =============================================================================

GATE_PRE_TRAINING = """
Before starting training, verify:
- [ ] At least 100 training samples available
- [ ] Average data quality score >= 0.6
- [ ] No duplicate samples > 5%
- [ ] Disk space available for checkpoints (>10GB)
- [ ] GPU memory sufficient for batch size

If any gate fails, STOP and report the issue.
"""

GATE_POST_TRAINING = """
After training completes, verify:
- [ ] Final loss < 2.0 (indicates model learned)
- [ ] No NaN losses during training
- [ ] Model can be loaded successfully
- [ ] Perplexity on eval set < 10.0

If any gate fails, DO NOT deploy. Report for human review.
"""

GATE_PRE_DEPLOYMENT = """
Before deploying to Ollama, verify:
- [ ] GGUF export completed successfully
- [ ] Model file size within expected range
- [ ] Ollama health check passes
- [ ] Test query returns valid response
- [ ] Response latency < 500ms

If any gate fails, STOP and await human approval.
"""

# =============================================================================
# ARTIFACT TEMPLATES
# =============================================================================

ARTIFACT_TRAINING_DASHBOARD = """
# Training Dashboard 📊

**Job ID:** {job_id}
**Status:** {status}
**Started:** {started_at}

## Progress
- Current Epoch: {current_epoch}/{total_epochs}
- Current Loss: {loss:.4f}
- ETA: {eta}

## Loss Curve
{loss_plot}

## Memory Usage
- GPU: {gpu_memory_mb} MB / {gpu_total_mb} MB
- CPU: {cpu_percent}%

## Actions
- [STOP] - Halt training if diverging
- [PAUSE] - Pause and resume later
"""

ARTIFACT_VALIDATION_REPORT = """
# Validation Report ✅

**Model:** {model_name}
**Base Model:** {base_model}

## Metrics Comparison

| Metric | Base Model | Fine-Tuned | Change |
|--------|-----------|------------|--------|
| Perplexity | {base_ppl:.2f} | {finetuned_ppl:.2f} | {ppl_change:+.1f}% |
| CodeBLEU | {base_codebleu:.2f} | {finetuned_codebleu:.2f} | {codebleu_change:+.1f}% |
| Hallucination Rate | {base_halluc:.1f}% | {finetuned_halluc:.1f}% | {halluc_change:+.1f}% |

## Recommendation

{recommendation}

## Sample Outputs
{sample_outputs}
"""

ARTIFACT_DEPLOYMENT_CHECKLIST = """
# Deployment Checklist 🚀

**Model:** {model_name}
**Target:** Ollama

## Pre-Deployment Checks
- [x] GGUF export completed
- [x] Model file valid ({size_mb} MB)
- [x] Ollama running
- [x] Test query successful

## Deployment Log
- Deployed at: {deployed_at}
- Previous model: {previous_model}
- New model: {new_model}

## Rollback Instructions
If issues occur:
```bash
ollama rm {new_model}
ollama run {previous_model}
```

## Verification
Run: `ollama run {new_model} "Write hello world in Python"`
"""
