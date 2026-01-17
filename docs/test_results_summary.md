# Test Results Summary

**Date:** January 17, 2026  
**Environment:** macOS, Python 3.12.10, pytest 9.0.2

---

## Unit Tests

**Command:** `pytest tests/unit -v --override-ini="addopts="`

### Results

| Metric | Value |
|--------|-------|
| Total | 192 |
| Passed | 167 |
| Failed | 13 |
| Skipped | 12 |
| Pass Rate | **86.9%** |
| Duration | 7.73s |

### Failed Tests

```
FAILED tests/unit/test_evaluator.py::TestEvaluationResult::test_summary
FAILED tests/unit/test_forge.py::TestConfigSerialization::test_from_yaml
FAILED tests/unit/test_forge.py::TestCheckpointPersistence::test_config_serialization_roundtrip
FAILED tests/unit/test_miner.py::TestRepositoryParsing::test_parse_repository
FAILED tests/unit/test_miner.py::TestRepositoryParsing::test_parse_repository_with_tests
FAILED tests/unit/test_miner.py::TestEdgeCases::test_empty_file
FAILED tests/unit/test_miner.py::TestEdgeCases::test_unicode_handling
FAILED tests/unit/test_miner.py::TestEdgeCases::test_lambda_functions
FAILED tests/unit/test_validator.py::TestTextSimilarity::test_compute_relevance
FAILED tests/unit/test_validator.py::TestRAFTExampleValidation::test_valid_example
FAILED tests/unit/test_validator.py::TestRAFTExampleValidation::test_valid_example_dict
FAILED tests/unit/test_validator.py::TestFiltering::test_filter_keeps_valid
FAILED tests/unit/test_validator.py::TestDataValidatorClass::test_validate_raft_example
```

### Skipped Tests (12)

Tests with `@pytest.mark.skipif` for missing tree-sitter language parsers.

---

## Integration Tests

**Command:** `pytest tests/integration -v --override-ini="addopts="`

### Results

| Metric | Value |
|--------|-------|
| Total | 70 |
| Passed | 61 |
| Failed | 9 |
| Skipped | 0 |
| Pass Rate | **87.1%** |
| Duration | 2.25s |

### Failed Tests

```
FAILED tests/integration/test_full_system_e2e.py::TestDataExtraction::test_extract_code_chunks
FAILED tests/integration/test_full_system_e2e.py::TestDataExtraction::test_extract_includes_docstrings
FAILED tests/integration/test_full_system_e2e.py::TestRAFTGeneration::test_generate_raft_examples
FAILED tests/integration/test_full_system_e2e.py::TestDataValidation::test_validate_generated_data
FAILED tests/integration/test_full_system_e2e.py::TestPipelineIntegration::test_extract_to_validation
FAILED tests/integration/test_full_system_e2e.py::TestPerformance::test_extraction_performance
FAILED tests/integration/test_full_system_e2e.py::TestFullE2E::test_complete_pipeline_flow
FAILED tests/integration/test_pipeline.py::TestDataPipelineIntegration::test_full_extraction_pipeline
FAILED tests/integration/test_repo_guardian.py::TestRepositoryMonitoring::test_monitor_detects_files_changed
```

---

## Test Files Not Run

The following test files were excluded due to import errors:

- `tests/unit/test_data_pipeline.py` - Imports non-existent `CodeMiner` class
- `tests/unit/test_training.py` - Import issues

---

## Passing Test Suites

### Unit Tests (Passing)

| Suite | Tests |
|-------|-------|
| `test_evaluator.py` | Most tests pass |
| `test_forge.py` | Core training tests pass |
| `test_miner.py` | Schema/config tests pass |
| `test_raft_generator.py` | All tests pass |
| `test_ollama_manager.py` | All tests pass |
| `test_validator.py` | Schema tests pass |

### Integration Tests (Passing)

| Suite | Tests |
|-------|-------|
| `test_ollama_manager.py` | All pass |
| `test_repo_guardian.py` | Most pass |
| `test_service.py` | All pass |

---

## Coverage

Coverage reporting requires fixing import issues first. Estimated coverage based on passing tests: **~70-75%**.

---

## Warnings

```
PytestConfigWarning: Unknown config option: timeout
PytestConfigWarning: Unknown config option: timeout_method
PytestUnknownMarkWarning: Unknown pytest.mark.asyncio
```

**Fix:** Install `pytest-timeout` and `pytest-asyncio` packages.
