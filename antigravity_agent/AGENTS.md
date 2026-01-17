# Antigravity Agent Scope Handbook

Governs all work within `antigravity_agent/`.

## Mission & Context
- Provide the Repo Guardian automation layer coordinating repository health checks, training plans, pipeline execution, reporting, and deployment triggers.
- Maintain compatibility with external agent orchestrators and generated artifacts consumed downstream.

## Components & Contracts
| Asset | Responsibility | Guardrails |
|-------|----------------|------------|
| `repo_guardian.py` | Mission control logic, async pipeline orchestration, health reporting | Preserve async interfaces and dataclasses (e.g., `HealthReport`, `PipelineConfig`). Maintain structured logging. Update tests when changing signatures or stages. |
| `skills.yaml` | Public skill catalog consumed by external agents | Treat as versioned API. Changing skill names/inputs/outputs requires documentation updates (`docs/repo_guardian.md`) and coordination with integrators. |
| `prompts.py` | Prompt templates and instructions | Keep deterministic and documented; avoid leaking secrets. |
| `artifacts_templates/` | Jinja templates for reports (e.g., `health_report.md.j2`) | Synchronize template variables with dataclass fields. Adjust tests and docs when altering structure. |

## Standards
- Follow global formatting, linting, typing, and docstring rules.
- Any new telemetry emitted must align with `monitoring/` expectations.
- Keep pipeline steps idempotent where feasible; guard long-running operations with clear logging.
- Avoid direct filesystem writes outside configured directories (`output/`, `logs/`).

## Testing Guidance
- Ensure coverage in `tests/integration/test_repo_guardian.py` and unit tests for new utilities.
- Add snapshot/regression tests for templates when modifying output formats.
- Mock external dependencies (Ollama, training engine) in tests to keep runtime manageable.

## Documentation
- Update `docs/repo_guardian.md`, `docs/deployment.md`, and `docs/monitoring.md` when automation flow, skill interfaces, or emitted metrics change.
- If new prompts/skills are added, include usage summaries and rationale in documentation.

## Special Considerations
- External consumers may rely on exact JSON schema from Repo Guardian responses—ensure backward compatibility or document migration steps.
- Long-running async tasks should surface progress metrics for monitoring.
- Follow clean-up practices for temporary artifacts to avoid disk bloat.
