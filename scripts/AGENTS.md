# Scripts Scope Handbook

Covers all files under `scripts/`.

## Mission
- Provide operational utilities (backups, migrations, maintenance tasks) referenced in deployment/production docs.

## Standards
- Scripts must be idempotent or provide dry-run options; destructive operations require explicit confirmation flags.
- Use `set -euo pipefail` for shell scripts; validate with shellcheck where practicable.
- Accept configuration via CLI flags or environment variables (no hardcoded secrets/paths).
- Document usage, prerequisites, and expected outputs in `docs/deployment.md`, `docs/production_checklist.md`, or dedicated docs.

## Testing & Validation
- Supply usage examples and, where feasible, automated tests (e.g., unit tests, CI smoke checks).
- Ensure scripts exit non-zero on failure and emit actionable error messages.

## Artifact Handling
- Output artifacts must be stored under git-ignored directories (`output/`, `logs/`, etc.).
- Provide cleanup guidance if scripts generate large files.

## Coordination
- Communicate breaking changes to maintainers; update documentation and release notes as necessary.
