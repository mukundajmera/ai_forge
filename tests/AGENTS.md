# Tests Scope Handbook

Covers `tests/` directory (unit/integration suites, fixtures, README).

## Mission
- Provide fast, reliable feedback on code changes and enforce coverage targets.

## Structure & Markers
| Subdir | Purpose |
|--------|---------|
| `unit/` | Fast, isolated tests (no external dependencies) |
| `integration/` | Multi-component tests (may require Ollama, file fixtures) |
| `fixtures/` | Shared data (sample repos, JSON datasets) |
| `README.md` | Testing instructions and coverage/performance targets |

Markers defined in `pytest.ini`:
- `unit`
- `integration`
- `slow`
- `e2e`
- `performance`

Register new markers in both `pytest.ini` and `[tool.pytest.ini_options]` before use.

## Rules
- Name files `test_*.py`, classes `Test*`, functions `test_*`.
- Maintain fixtures lightweight and well-documented; store large data outside repo.
- Update tests in lockstep with code changes; keep coverage ≥85%.
- Use `pytest` options from root handbook; do not modify `.pytest_cache/` manually.

## Ollama Dependency
- Integration tests rely on Ollama at `http://localhost:11434`. Provide mocks or skip markers when unavailable.

## Documentation
- Update `tests/README.md` when adding commands, fixtures, or adjusting targets.
