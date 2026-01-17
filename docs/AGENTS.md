# Documentation Scope Handbook

Applies to `docs/` directory.

## Mission
- Maintain a single source of truth for architecture, configuration, deployment, monitoring, troubleshooting, and automation workflows.

## Expectations
- Update documentation in lockstep with code/tooling changes.
- Use consistent Markdown structure (headings, tables, code fences with language hints).
- Leverage pre-commit hooks for linting YAML/Markdown/JSON where applicable.
- Cross-link to relevant module handbooks when documenting scope-specific workflows.

## Key Artifacts & Ownership
| File | Focus |
|------|-------|
| `architecture.md` | System architecture, component diagrams |
| `ci_cd.md` | CI/CD pipelines, secrets, release flow |
| `configuration.md` | Config parameters, hardware profiles |
| `deployment.md` | Local, Docker, cloud deployment procedures |
| `monitoring.md` | Metrics, dashboards, alerting procedures |
| `production_checklist.md` | Pre-release readiness steps |
| `troubleshooting.md` | Known issues and remediation |
| `repo_guardian.md` | Automation internals |

## Standards
- Document environment variables/secrets responsibly (no plaintext secrets).
- Verify command examples before publishing; ensure they match current tooling versions.
- Keep Mermaid diagrams updated when architecture changes.
- Include date/version context for major updates when relevant.

## Coordination
- Notify maintainers before removing or drastically refactoring docs relied upon by teams or automation.
- Mirror documentation changes in release notes/changelogs when applicable.
