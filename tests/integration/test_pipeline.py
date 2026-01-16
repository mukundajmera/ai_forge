"""Integration tests for end-to-end pipeline."""

import pytest
from pathlib import Path
from typing import Any


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a sample project for testing."""
    project = tmp_path / "sample_project"
    project.mkdir()
    
    # Create source files
    src = project / "src"
    src.mkdir()
    
    (src / "__init__.py").write_text('"""Sample package."""')
    
    (src / "main.py").write_text('''
"""Main module for sample project."""

def greet(name: str) -> str:
    """Greet a person by name.
    
    Args:
        name: The name to greet.
        
    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        a: First number.
        b: Second number.
        
    Returns:
        The sum of a and b.
    """
    return a + b

class Calculator:
    """Simple calculator class."""
    
    def __init__(self) -> None:
        """Initialize calculator."""
        self.history: list[str] = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
''')
    
    # Create README
    (project / "README.md").write_text('''
# Sample Project

This is a sample project for testing AI Forge integration.

## Features
- Greeting functionality
- Calculator operations
''')
    
    return project


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""
    
    def test_full_extraction_pipeline(self, sample_project: Path) -> None:
        """Test complete data extraction from a project."""
        from ai_forge.data_pipeline import CodeMiner, RAFTGenerator, DataValidator
        
        # Step 1: Mine code
        miner = CodeMiner(sample_project)
        chunks = miner.extract_all()
        
        assert len(chunks) > 0
        
        # Step 2: Generate RAFT data
        generator = RAFTGenerator(chunks)
        dataset = generator.generate_dataset(num_samples=10)
        
        assert len(dataset) == 10
        
        # Step 3: Validate
        validator = DataValidator(dataset)
        result = validator.validate_all()
        
        # May not be valid due to small sample size, but should run
        assert result.total_samples == 10


class TestAPIIntegration:
    """Integration tests for API service."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self) -> None:
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from ai_forge.conductor.service import app
        
        with TestClient(app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self) -> None:
        """Test root endpoint."""
        from fastapi.testclient import TestClient
        from ai_forge.conductor.service import app
        
        with TestClient(app) as client:
            response = client.get("/")
            
            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "AI Forge"


class TestGuardianIntegration:
    """Integration tests for RepoGuardian."""
    
    @pytest.mark.asyncio
    async def test_repository_analysis(self, sample_project: Path) -> None:
        """Test repository analysis."""
        from ai_forge.antigravity_agent import RepoGuardian
        
        guardian = RepoGuardian(sample_project)
        report = await guardian.analyze_repository()
        
        assert report.repository_path == str(sample_project)
        assert report.code_files > 0
    
    @pytest.mark.asyncio
    async def test_health_report_markdown(self, sample_project: Path) -> None:
        """Test health report markdown generation."""
        from ai_forge.antigravity_agent import RepoGuardian
        
        guardian = RepoGuardian(sample_project)
        report = await guardian.analyze_repository()
        
        markdown = report.to_markdown()
        
        assert "# Repository Health Report" in markdown
        assert str(sample_project) in markdown
