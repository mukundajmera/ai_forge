# Config Scope Handbook

Governance for `config/` directory content (YAML defaults and related artifacts).

## Mission
- Provide canonical configuration sources for training, deployment, and operational defaults.
- Ensure YAML files stay synchronized with typed schemas (`training/schemas.py`) and documentation.

## Guidelines
- Every parameter change must be mirrored in `training/schemas.py` and documented in `docs/configuration.md` (plus `.env.example` if environment-driven).
- Use descriptive comments to explain defaults, especially hardware profiles.
- Avoid embedding secrets or environment-specific paths.
- Validate YAML via `yamllint` (pre-commit hook) before submission.

## Change Workflow
1. Update schema definitions/dataclasses.
2. Modify YAML default(s) here.
3. Update documentation and environment samples.
4. Add/adjust tests to cover serialization, default loading, and compatibility.

## Special Cases
- When introducing new hardware profiles, ensure they align with Apple Silicon constraints (memory, compute).
- Keep configuration names stable; coordinate with downstream consumers before breaking changes.
