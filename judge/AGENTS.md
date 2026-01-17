# Judge Scope Handbook

Applies to `judge/` (evaluation, benchmarking, GGUF export).

## Mission
- Evaluate trained models for quality (perplexity, CodeBLEU, hallucination metrics, etc.).
- Export fine-tuned models to GGUF for Ollama deployment.
- Produce reports summarizing evaluation outcomes.

## Components
| File | Responsibility | Guardrails |
|------|----------------|------------|
| `evaluator.py` | Computes evaluation metrics | Keep metric APIs backward compatible. Document new metrics and thresholds. |
| `exporter.py` | Converts models to GGUF | Ensure compatibility with Ollama and `conductor/modelfile_template`. Tests must cover new formats. |
| `report.py` | Generates evaluation summaries | Keep synchronized with templates and docs. |
| `benchmarks/` | Benchmark configs/scripts | Maintain lightweight, documented dependencies. |

## Standards
- Follow global lint/type/docstring policies.
- Ensure deterministic evaluation for reproducibility (control seeds where possible).
- Avoid writing artifacts outside git-ignored directories (`output/`).

## Testing
- Unit tests: evaluator/exporter coverage in `tests/unit/`.
- Integration tests: pipeline tests working through evaluation/export phases.
- Add regression tests when adjusting metric calculations or export formats.

## Documentation
- Update `docs/pissa_qlora_guide.md`, `docs/api_reference.md`, `docs/deployment.md` when evaluation/export behavior changes.
- Record new metrics/benchmarks in `docs/monitoring.md` or dedicated documentation.

## Special Considerations
- GGUF exports can be large; enforce cleanup guidance in docs when workflows create temporary artifacts.
- Coordinate quantization or metadata changes with deployment and monitoring teams.
