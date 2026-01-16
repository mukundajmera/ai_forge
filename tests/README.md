# AI Forge Test Suite

Comprehensive test suite for the AI Forge fine-tuning platform.

## Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_miner.py        # Code extraction tests
│   ├── test_raft_generator.py   # RAFT data synthesis tests
│   ├── test_validator.py    # Data validation tests
│   ├── test_forge.py        # Training tests
│   ├── test_evaluator.py    # Evaluation tests
│   └── test_ollama_manager.py   # Ollama interaction tests
├── integration/             # Tests with dependencies
│   ├── test_pipeline.py     # Data pipeline E2E
│   ├── test_service.py      # API endpoints
│   ├── test_repo_guardian.py    # Agent tests
│   ├── test_ollama_manager.py   # Ollama integration
│   └── test_full_system_e2e.py  # Complete E2E
└── fixtures/                # Test data
    ├── sample_python_repo/  # Sample project
    ├── sample_code_blocks.json
    └── sample_raft_examples.json
```

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only (< 1 minute)
```bash
pytest -m unit
```

### Integration Tests (< 5 minutes)
```bash
pytest -m integration
```

### Skip Slow Tests
```bash
pytest -m "not slow"
```

### With Coverage Report
```bash
pytest --cov=ai_forge --cov-report=html --cov-report=xml
open htmlcov/index.html
```

### Specific Test File
```bash
pytest tests/unit/test_miner.py -v
```

### Single Test
```bash
pytest tests/unit/test_miner.py::TestPythonParsing::test_extract_simple_function -v
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Fast, isolated tests without external dependencies |
| `integration` | Tests that may use external services |
| `slow` | Tests taking > 30 seconds |
| `e2e` | Full end-to-end pipeline tests |
| `performance` | Performance benchmark tests |

## Coverage Targets

| Component | Target |
|-----------|--------|
| Overall | 85%+ |
| data_pipeline | 90%+ |
| training | 80%+ |
| conductor | 85%+ |
| judge | 85%+ |
| antigravity_agent | 80%+ |

## Performance Targets

| Operation | Target |
|-----------|--------|
| Extract 100 files | < 5 minutes |
| Train 500 examples (3B) | < 30 minutes |
| Train 500 examples (7B) | < 60 minutes |
| Peak memory | < 12GB |

## Fixtures

### sample_code_blocks.json
Pre-extracted code chunks for testing RAFT generation.

### sample_raft_examples.json
Pre-generated RAFT examples for testing validation.

### sample_python_repo/
Small Python project for data extraction testing.

## Writing New Tests

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use appropriate markers
4. Add fixtures to `tests/fixtures/__init__.py`
5. Follow naming convention: `test_*.py`

Example:
```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    def test_basic_functionality(self):
        assert True
    
    @pytest.mark.slow
    def test_performance(self):
        # Long-running test
        pass
```

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Run Tests
  run: |
    pytest -m "not slow" --cov=ai_forge --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: coverage.xml
```
