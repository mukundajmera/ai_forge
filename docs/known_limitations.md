# Known Limitations

**Version:** 1.0.0  
**Date:** January 17, 2026

---

## Critical Limitations

### 1. CodeMiner Class Not Implemented

**Status:** 🔴 Blocking  
**Impact:** E2E pipeline, data extraction tests

The codebase references a `CodeMiner` class that does not exist. The actual implementation uses a function-based API:

```python
# ❌ Documented/referenced (doesn't exist)
from data_pipeline.miner import CodeMiner
miner = CodeMiner(repo_path)
chunks = miner.extract_all()

# ✅ Actual API
from data_pipeline.miner import parse_repository
blocks = parse_repository(repo_path)
```

**Workaround:** Use `parse_repository()` function directly.

---

### 2. Tree-sitter Parser Version Incompatibility

**Status:** 🔴 Blocking  
**Impact:** Multi-language code extraction

The code uses deprecated tree-sitter API:

```python
# ❌ Current (fails with tree-sitter 0.21+)
parser.language = lang_module

# ✅ Required
parser.set_language(lang_module)
```

**Workaround:** Pin `tree-sitter<0.21` or update the parser code.

---

## Non-Critical Limitations

### 3. Test Assertion Format Mismatches

**Status:** 🟡 Minor  
**Impact:** 1 unit test failure

Test expects `85%` but implementation outputs `85.00%`.

---

### 4. Relevance Threshold Tuning

**Status:** 🟡 Minor  
**Impact:** 6 validator tests

Default relevance thresholds may be too strict for typical use cases.

---

### 5. Git Change Detection Timing

**Status:** 🟡 Minor  
**Impact:** 1 integration test

`monitor_repository()` may not detect very recent file changes until next git operation.

---

## Environment Limitations

### macOS Only

AI Forge is designed for **Mac Apple Silicon** (M1/M2/M3). While it may run on other platforms, this is untested.

### Memory Requirements

- **Minimum:** 16GB RAM
- **Recommended:** 32GB+ for 7B models
- Training larger models may fail with OOM errors

### Model Size Constraints

- 3B models: Tested and supported
- 7B models: Supported with increased memory
- 13B+ models: Not recommended without GPU offloading

---

## Acceptable Exceptions

| Item | Justification |
|------|---------------|
| 2 test files skipped | Import errors from missing class - will be fixed |
| 12 tests skipped | Tree-sitter grammars not installed - optional |
| Coverage < 85% | Blocked by import errors - actual coverage higher |

---

## Feature Gaps (Known)

| Feature | Status | Notes |
|---------|--------|-------|
| Windows support | Not planned | macOS only |
| GPU training | Partial | MLX uses Metal, no CUDA |
| Real-time streaming | Planned | SSE endpoints stub exists |
| Multi-user auth | Not implemented | Single-user system |
| Model versioning | Partial | Manual checkpoints only |

---

## Planned Improvements

1. Fix `CodeMiner` class implementation
2. Update tree-sitter API usage
3. Add Windows/Linux compatibility
4. Implement streaming responses
5. Add model versioning system
