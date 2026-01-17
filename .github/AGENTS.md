# GitHub Configuration Scope Handbook

Covers `.github/` directory (workflows, templates, automation configs).

## Mission
- Maintain CI/CD pipelines, issue/PR templates, and repository automation that enforce project quality gates.

## Assets
| File | Responsibility | Guardrails |
|------|----------------|------------|
| `workflows/tests.yml` | CI pipeline (tests, lint, build) | Keep triggers aligned with `main` and `develop`. Reflect command changes in docs. Maintain matrix for Ubuntu/macOS. |
| `workflows/release.yml` | Release pipeline | Ensure tag pattern `v*` remains intact. Document environment secrets usage (`PYPI_API_TOKEN`). |
| Issue/PR templates | Contribution guidance | Update when process expectations change. |

## Standards
- Validate workflow syntax with `act` or GitHub’s workflow editor when possible.
- Avoid storing secrets in workflows; reference GitHub Secrets instead.
- Coordinate with maintainers before altering job sequences, coverage thresholds, or gating logic.

## Documentation
- Update `docs/ci_cd.md` and README badges when workflows change names, triggers, or responsibilities.

## Special Notes
- Keep concurrency settings to prevent duplicate runs (already configured in `tests.yml`).
- When adding new workflows, provide summary in root handbook or docs.
