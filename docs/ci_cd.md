# AI Forge - CI/CD Documentation

This document explains the CI/CD pipeline and how to work with it.

## Overview

AI Forge uses GitHub Actions for continuous integration and deployment:

1. **CI Pipeline** (`tests.yml`) - Runs on every push and PR
2. **Release Pipeline** (`release.yml`) - Runs on version tags

## CI Pipeline

### Trigger

```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
```

### Jobs

#### 1. Unit Tests
- Runs on Ubuntu and macOS
- Executes all unit tests
- Generates coverage report
- Uploads to Codecov

#### 2. Integration Tests
- Runs on Ubuntu and macOS
- Installs Ollama
- Runs integration tests
- Uploads logs on failure

#### 3. Code Quality
- Runs on Ubuntu
- Checks formatting (Black)
- Lints code (Ruff)
- Type checks (MyPy)
- Validates docstrings

#### 4. Security Scan
- Runs on Ubuntu
- Scans for security issues (Bandit)

#### 5. Build Package
- Runs after tests pass
- Builds wheel and source distribution
- Uploads artifacts

## Release Pipeline

### Trigger

```yaml
on:
  push:
    tags:
      - 'v*'
```

### Jobs

1. **Full Test Suite** - Runs complete test suite
2. **Build Package** - Builds and validates package
3. **Create Release** - Creates GitHub release with artifacts
4. **Publish to PyPI** - Publishes stable releases to PyPI

## Running Locally

### Install Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit -v

# With coverage
pytest --cov=ai_forge --cov-report=html
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint
ruff check .

# Type check
mypy ai_forge
```

### Pre-commit Hooks

```bash
# Install hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Configuration

### pyproject.toml

Contains configuration for:
- Black (formatting)
- isort (import sorting)
- Ruff (linting)
- mypy (type checking)
- pytest (testing)
- coverage (code coverage)

### .pre-commit-config.yaml

Defines pre-commit hooks:
- Trailing whitespace
- File endings
- YAML/JSON validation
- Black formatting
- isort imports
- Ruff linting
- MyPy type checking
- Bandit security scan

## Creating a Release

1. Update version in `pyproject.toml`
2. Commit changes
3. Create tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`

The release workflow will:
- Run full test suite
- Build package
- Create GitHub release
- Publish to PyPI (if not alpha/beta)

## Environment Variables

### Required Secrets

| Secret | Description |
|--------|-------------|
| `PYPI_API_TOKEN` | PyPI upload token (for releases) |
| `CODECOV_TOKEN` | Codecov upload token (optional) |

### Setting Secrets

1. Go to repository Settings > Secrets
2. Add required secrets

## Badges

Add to README.md:

```markdown
[![CI](https://github.com/your-org/ai-forge/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/ai-forge/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/your-org/ai-forge/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/ai-forge)
```

## Troubleshooting

### Tests Fail on macOS

Mac-specific issues:
- Ensure Xcode tools installed: `xcode-select --install`
- Check Python version: `python --version`

### Ollama Not Available

Integration tests skip Ollama if not available. To test locally:

```bash
# Install Ollama
brew install ollama

# Start server
ollama serve
```

### MyPy Errors

Type checking may fail for third-party libraries. Add to ignore:

```python
# type: ignore[import]
```

### Coverage Too Low

Increase coverage by adding tests. Aim for 85%+ overall.
