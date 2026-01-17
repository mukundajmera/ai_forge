# Training Scope Handbook

Applies to `training/` directory (engine, schemas, callbacks, losses).

## Mission
- Implement PiSSA + QLoRA fine-tuning optimized for Apple Silicon.
- Provide configuration schemas, device detection, hooks, and utilities for training jobs triggered via Repo Guardian or API.

## Core Modules
| File | Responsibility | Guardrails |
|------|----------------|------------|
| `forge.py` | Training orchestrator | Maintain `TrainingForge` API compatibility, device detection (`detect_device`), and logging. Profile performance impact of changes. |
| `schemas.py` | Dataclasses/enums for configs | Source of truth for config validation. Sync with YAML defaults (`config/`). |
| `callbacks/` | Metrics logger, early stopping, memory monitor, etc. | Keep callbacks lightweight; cover with unit tests. |
| `losses/` | Custom loss components | Ensure numerical stability and documentation.

## Standards
- Strict typing and docstrings per global rules.
- Avoid hardcoding device-specific behavior; rely on detection utilities.
- Keep defaults tuned for 16GB machines while documenting larger profiles.
- Store artifacts (checkpoints, logs) under git-ignored directories.

## Testing
- Update `tests/unit/test_forge.py` and related suites when changing training logic.
- Integration tests covering training pipeline should remain guarded with markers to avoid long CI runs.

## Documentation
- Update `docs/pissa_qlora_guide.md`, `docs/developer_guide.md`, and `docs/configuration.md` when defaults or training flows change.
- Document new callbacks/metrics in `docs/monitoring.md` if surfaced externally.

## Special Considerations
- Provide configuration guidance for low-memory machines when adjusting defaults.
- Ensure compatibility with evaluation/export pipeline expectations (model formats, metadata).
