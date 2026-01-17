# Monitoring Scope Handbook

Governs files within `monitoring/` (Prometheus, Grafana, Alertmanager).

## Mission
- Provide observability assets for AI Forge services and dependencies, ensuring reliable monitoring, alerting, and dashboards.

## Assets & Guardrails
| File | Role | Guidance |
|------|------|---------|
| `prometheus.yml` | Scrape configuration | Keep targets accurate (API, Ollama, Prometheus, node exporter). Document new metrics in `docs/monitoring.md`. |
| `alerts.yml` | Prometheus alert rules | Calibrate thresholds with SLOs; include actionable descriptions. |
| `alertmanager.yml` | Alert routing | Coordinate receiver updates with incident response policies. |
| `grafana/` | Dashboard & datasource provisioning | Ensure provisioning files remain valid; update docs when dashboards change. |

## Standards
- Maintain metric naming conventions (`ai_forge_*`).
- Provide runbook links or references in alert annotations when possible.
- Keep stack optional—services should tolerate metrics unavailability.

## Testing
- Validate Prometheus configuration with `promtool check config monitoring/prometheus.yml` if available.
- Test Grafana provisioning by starting monitoring profile (`docker-compose --profile monitoring up -d`).

## Documentation
- Reflect changes in `docs/monitoring.md`, including updated dashboards, alerts, and troubleshooting steps.

## Special Considerations
- Coordinate metric schema changes with service owners to avoid breaking dashboards.
- Avoid hardcoding credentials; rely on environment variables/secrets when necessary.
