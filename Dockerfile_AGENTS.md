# Dockerfile Notes

**Scope:** Supplement to root handbook for Dockerfile maintenance.

- Maintain multi-stage build logic (builder produces wheels, production installs).
- Keep base image `python:3.11-slim`; evaluate upgrades alongside CI matrix.
- When adding system packages, ensure compatibility with Apple Silicon and update `docs/deployment.md`.
- Preserve non-root user (`app`) setup for security.
- Healthcheck must continue to target `/health` endpoint.
- Rebuild instructions documented in root handbook (`docker build -t ai-forge:latest .`).
