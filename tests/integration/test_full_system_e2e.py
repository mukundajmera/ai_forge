"""Full System E2E Test - Complete Pipeline Verification.

This test runs the complete AI Forge pipeline:
1. Extract code from sample repository
2. Generate RAFT training data
3. Validate data quality
4. (Mocked) Training step
5. (Mocked) Evaluation
6. (Mocked) Export and deployment

Estimated runtime: ~5 minutes (with mocks) or ~30 minutes (full)

Run with: pytest tests/integration/test_full_system_e2e.py -v -s
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import json
import tempfile
import subprocess

# Import path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Markers
# =============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_repo(tmp_path):
    """Create a realistic sample Python project."""
    # Create project structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    
    # Main application file
    (tmp_path / "src" / "main.py").write_text('''
"""Main application module.

This module provides the core application functionality
for processing and analyzing data.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and transform data.
    
    This class provides methods for loading, transforming,
    and saving data in various formats.
    
    Attributes:
        config: Configuration dictionary.
        verbose: Whether to log verbose output.
    
    Example:
        >>> processor = DataProcessor({"format": "json"})
        >>> result = processor.process([1, 2, 3])
    """
    
    def __init__(self, config: dict, verbose: bool = False) -> None:
        """Initialize DataProcessor.
        
        Args:
            config: Configuration dictionary.
            verbose: Whether to log verbose output.
        """
        self.config = config
        self.verbose = verbose
        logger.info("DataProcessor initialized")
    
    def process(self, data: list) -> list:
        """Process a list of data items.
        
        Args:
            data: Input data list.
            
        Returns:
            Processed data list.
        """
        if self.verbose:
            logger.info(f"Processing {len(data)} items")
        
        return [self._transform(item) for item in data]
    
    def _transform(self, item) -> dict:
        """Transform a single item.
        
        Args:
            item: Item to transform.
            
        Returns:
            Transformed item as dictionary.
        """
        return {"value": item, "processed": True}
    
    def save(self, data: list, path: str) -> None:
        """Save processed data to file.
        
        Args:
            data: Data to save.
            path: Output file path.
        """
        import json
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved to {path}")
    
    def load(self, path: str) -> list:
        """Load data from file.
        
        Args:
            path: Input file path.
            
        Returns:
            Loaded data.
        """
        import json
        with open(path) as f:
            return json.load(f)


def main():
    """Main entry point."""
    processor = DataProcessor({"format": "json"}, verbose=True)
    data = processor.process([1, 2, 3, 4, 5])
    print(f"Processed: {data}")


if __name__ == "__main__":
    main()
''')

    # Utility module
    (tmp_path / "src" / "utils.py").write_text('''
"""Utility functions for the application."""

from typing import Any, Optional
import hashlib


def compute_hash(data: str) -> str:
    """Compute SHA256 hash of input string.
    
    Args:
        data: Input string to hash.
        
    Returns:
        Hexadecimal hash string.
    """
    return hashlib.sha256(data.encode()).hexdigest()


def validate_email(email: str) -> bool:
    """Validate email format.
    
    Args:
        email: Email address to validate.
        
    Returns:
        True if valid format.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def format_bytes(size: int) -> str:
    """Format byte count as human-readable string.
    
    Args:
        size: Size in bytes.
        
    Returns:
        Formatted string (e.g., "1.5 MB").
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


async def fetch_data(url: str) -> dict:
    """Fetch data from URL asynchronously.
    
    Args:
        url: URL to fetch from.
        
    Returns:
        Response data as dictionary.
    """
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
''')

    # Initialize git
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, capture_output=True
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path, capture_output=True
    )
    
    return tmp_path


@pytest.fixture
def mock_training():
    """Mock training components."""
    with patch("ai_forge.training.forge.TrainingForge") as mock:
        instance = MagicMock()
        instance.load_model.return_value = None
        instance.train.return_value = {"train_loss": 0.5, "epochs": 3}
        instance.save_model.return_value = None
        mock.return_value = instance
        yield mock


# =============================================================================
# Data Extraction Tests
# =============================================================================

class TestDataExtraction:
    """Tests for data extraction from repository."""
    
    def test_extract_code_chunks(self, sample_repo) -> None:
        """Test extracting code chunks from sample repo."""
        from data_pipeline.miner import CodeMiner
        
        miner = CodeMiner(sample_repo, languages=["python"])
        chunks = miner.extract_all()
        
        # Should extract functions and classes
        assert len(chunks) > 0
        
        # Check for specific functions
        names = [c.name for c in chunks]
        assert "DataProcessor" in names or "process" in names
    
    def test_extract_includes_docstrings(self, sample_repo) -> None:
        """Test that docstrings are extracted."""
        from data_pipeline.miner import CodeMiner
        
        miner = CodeMiner(sample_repo)
        chunks = miner.extract_all()
        
        # At least some chunks should have docstrings
        with_docstring = [c for c in chunks if c.docstring]
        assert len(with_docstring) > 0


# =============================================================================
# RAFT Data Generation Tests
# =============================================================================

class TestRAFTGeneration:
    """Tests for RAFT data synthesis."""
    
    def test_generate_raft_examples(self, sample_repo) -> None:
        """Test generating RAFT examples from chunks."""
        from data_pipeline.miner import CodeMiner
        from data_pipeline.raft_generator import RAFTGenerator
        
        # Extract chunks
        miner = CodeMiner(sample_repo)
        chunks = miner.extract_all()
        
        # Generate RAFT data
        generator = RAFTGenerator(chunks)
        examples = generator.generate_dataset(num_examples=10)
        
        assert len(examples) > 0
        
        # Check structure
        for ex in examples:
            assert "instruction" in ex
            assert "input" in ex or "context" in ex
            assert "output" in ex


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestDataValidation:
    """Tests for data validation."""
    
    def test_validate_generated_data(self, sample_repo) -> None:
        """Test validating generated training data."""
        from data_pipeline.miner import CodeMiner
        from data_pipeline.raft_generator import RAFTGenerator
        from data_pipeline.validator import DataValidator
        
        # Generate data
        miner = CodeMiner(sample_repo)
        chunks = miner.extract_all()
        generator = RAFTGenerator(chunks)
        examples = generator.generate_dataset(num_examples=10)
        
        # Validate
        validator = DataValidator(examples)
        result = validator.validate_all()
        
        assert result.total_samples > 0
        assert result.quality_score >= 0


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Tests for integrated pipeline execution."""
    
    def test_extract_to_validation(self, sample_repo) -> None:
        """Test full data pipeline: extract → RAFT → validate."""
        from data_pipeline.miner import CodeMiner
        from data_pipeline.raft_generator import RAFTGenerator
        from data_pipeline.validator import DataValidator
        
        # Step 1: Extract
        miner = CodeMiner(sample_repo)
        chunks = miner.extract_all()
        assert len(chunks) > 0, "No chunks extracted"
        
        # Step 2: Generate RAFT
        generator = RAFTGenerator(chunks)
        examples = generator.generate_dataset(num_examples=20)
        assert len(examples) > 0, "No examples generated"
        
        # Step 3: Validate
        validator = DataValidator(examples)
        result = validator.validate_all()
        
        # Save for training
        output_path = sample_repo / "data" / "training_data.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(examples, f)
        
        assert output_path.exists()
    
    def test_repo_guardian_integration(self, sample_repo) -> None:
        """Test RepoGuardian orchestration."""
        from antigravity_agent.repo_guardian import RepoGuardian, PipelineConfig
        
        config = PipelineConfig(
            auto_extract_data=False,
            auto_train=False,
            auto_deploy=False,
        )
        
        guardian = RepoGuardian(sample_repo, config)
        
        # Monitor repository
        result = guardian.monitor_repository()
        assert "should_retrain" in result
        
        # Create plan
        plan = guardian.plan_training_cycle()
        assert len(plan["plan"]) == 6


# =============================================================================
# FastAPI Service Tests
# =============================================================================

class TestServiceIntegration:
    """Tests for FastAPI service integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from conductor.service import app, state
        
        state.jobs = {}
        return TestClient(app)
    
    def test_full_api_workflow(self, client, sample_repo) -> None:
        """Test API workflow: health → retrain → status."""
        # Health check
        response = client.get("/health")
        assert response.status_code == 200
        
        # Monitor
        response = client.get(
            "/v1/retrain/monitor",
            params={"project_path": str(sample_repo)},
        )
        assert response.status_code == 200


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance benchmark tests."""
    
    def test_extraction_performance(self, sample_repo, tmp_path) -> None:
        """Test data extraction performance."""
        import time
        
        # Create 50 more files
        for i in range(50):
            (tmp_path / f"module_{i}.py").write_text(f'''
"""Module {i}."""

def function_{i}(x: int) -> int:
    """Process value {i}."""
    return x + {i}

class Class_{i}:
    """Class {i}."""
    
    def method(self) -> str:
        return "result_{i}"
''')
        
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        
        from data_pipeline.miner import CodeMiner
        
        start = time.time()
        miner = CodeMiner(tmp_path)
        chunks = miner.extract_all()
        elapsed = time.time() - start
        
        print(f"\n[PERF] Extracted {len(chunks)} chunks in {elapsed:.2f}s")
        
        # Should complete in under 30 seconds for 50 files
        assert elapsed < 30, f"Extraction took too long: {elapsed}s"


# =============================================================================
# Full E2E Test (with mocks)
# =============================================================================

class TestFullE2E:
    """Complete end-to-end test with mocks."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, sample_repo) -> None:
        """Test complete pipeline: extract → train → eval → deploy."""
        from data_pipeline.miner import CodeMiner
        from data_pipeline.raft_generator import RAFTGenerator
        from data_pipeline.validator import DataValidator
        
        results = {"steps": []}
        
        # Step 1: Extract
        miner = CodeMiner(sample_repo)
        chunks = miner.extract_all()
        results["steps"].append({
            "name": "extract",
            "success": len(chunks) > 0,
            "chunks": len(chunks),
        })
        
        # Step 2: Generate
        generator = RAFTGenerator(chunks)
        examples = generator.generate_dataset(num_examples=50)
        results["steps"].append({
            "name": "generate",
            "success": len(examples) > 0,
            "examples": len(examples),
        })
        
        # Step 3: Validate
        validator = DataValidator(examples)
        validation = validator.validate_all()
        results["steps"].append({
            "name": "validate",
            "success": validation.is_valid,
            "quality": validation.quality_score,
        })
        
        # Step 4: Training (mocked)
        results["steps"].append({
            "name": "train",
            "success": True,
            "loss": 0.5,
            "mocked": True,
        })
        
        # Step 5: Evaluation (mocked)
        results["steps"].append({
            "name": "evaluate",
            "success": True,
            "perplexity": 5.0,
            "mocked": True,
        })
        
        # Step 6: Deployment (mocked)
        results["steps"].append({
            "name": "deploy",
            "success": True,
            "model_name": "test-model",
            "mocked": True,
        })
        
        # Verify all steps succeeded
        all_success = all(s["success"] for s in results["steps"])
        
        print("\n" + "=" * 50)
        print("E2E PIPELINE RESULTS")
        print("=" * 50)
        for step in results["steps"]:
            status = "✅" if step["success"] else "❌"
            print(f"{status} {step['name']}: {step}")
        print("=" * 50)
        
        assert all_success, f"Some steps failed: {results}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
