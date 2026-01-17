# Conductor Service Scope Handbook

Applies to all files in `conductor/` (FastAPI application, job orchestration, Ollama integration).

## Mission
- Serve OpenAI-compatible REST API for chat/fine-tune endpoints.
- Manage job queue operations and communicate with Repo Guardian and training pipeline.
- Interface reliably with Ollama for model deployment and inference.
- Expose health and metrics endpoints for observability.

## Components
| File | Responsibility | Guardrails |
|------|----------------|------------|
| `service.py` | FastAPI app, routing, dependency injection | Preserve endpoint schemas (`/health`, `/v1/chat/completions`, `/v1/fine-tune`, `/v1/retrain`, `/metrics`). Update `docs/api_reference.md` when modifying payloads or responses. Keep Prometheus metrics consistent. |
| `job_queue.py` | Async job lifecycle management | Maintain thread-safe/async-safe design. Persist jobs as required. Update tests if queue semantics change. |
| `ollama_manager.py` | Interaction with Ollama daemon | Handle connection failures gracefully; provide fallbacks. Update `docs/ollama_integration.md` for new behaviors. |
| `modelfile_template` | Template for Ollama model registration | Keep in sync with `judge/exporter.py` outputs. |

## Standards
- Follow global style, linting, typing rules.
- Use Pydantic models for request/response validation; ensure clear error responses.
- Log request metadata cautiously (no secrets). Provide correlation IDs when useful.
- Metrics emitted via `/metrics` must follow `ai_forge_*` namespace.

## Testing
- Unit tests live in `tests/unit/test_ollama_manager.py`, service-specific tests if present.
- Integration tests (`tests/integration/test_service.py`) should cover routing, health, job submission, and Ollama interactions. Update fixtures/mocks when APIs change.
- Document required environment variables (`.env.example`, docs) for new functionality.

## Documentation
- Update `docs/api_reference.md`, `docs/user_guide.md`, and `docs/deployment.md` when endpoints or service operations change.
- Document any new configuration knobs in `docs/configuration.md`.

## Special Considerations
- Health and metrics endpoints are consumed by CI/CD and monitoring; avoid breaking changes without coordination.
- Running jobs should not block the event loop—use background tasks/async concurrency primitives.
- When introducing streaming or SSE endpoints, document usage and performance characteristics.
