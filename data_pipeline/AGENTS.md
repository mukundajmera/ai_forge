# Data Pipeline Scope Handbook

Applies to `data_pipeline/` (mining, RAFT generation, validation).

## Mission
- Extract structured code/documentation using Tree-sitter parsers.
- Generate RAFT-style training datasets and validate for quality, coverage, and deduplication.

## Components
| File | Responsibility | Guardrails |
|------|----------------|------------|
| `miner.py` | Repository code extraction | Keep language support in sync with requirements; update fixtures/tests when adding grammars. Optimize for memory and speed. |
| `raft_generator.py` | RAFT dataset synthesis | Ensure deterministic outputs for testing. Document heuristics in `docs/research_summary.md` or `docs/developer_guide.md`. |
| `validator.py` | Data quality scoring | Calibrate thresholds and log results. Ensure integration with Repo Guardian quality gates. |
| `schemas/` | Typed schemas for pipeline data | Align with training expectations and documentation. |

## Standards
- Follow global lint/type/docstring policies.
- Avoid blocking operations; favour streaming/iterators for large repositories.
- Ensure outputs integrate cleanly with `training/` data loaders.
- Keep fixtures in `tests/fixtures/` lightweight and well-documented.

## Testing
- Unit tests: `tests/unit/test_miner.py`, `tests/unit/test_raft_generator.py`, `tests/unit/test_validator.py`.
- Integration tests: `tests/integration/test_pipeline.py` and related suites.
- Add regression tests when modifying output formats or heuristics.

## Documentation
- Update `docs/configuration.md`, `docs/developer_guide.md`, and `docs/troubleshooting.md` when pipeline behavior or configuration changes.
- Capture new research-backed heuristics in `docs/research_summary.md`.

## Special Considerations
- Manage memory usage carefully to support 16GB/8GB machines; document best practices for low-memory profiles.
- When adding language support, update installation instructions for new Tree-sitter grammars.
