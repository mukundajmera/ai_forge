"""Integration tests for Repo Guardian Agent.

Tests cover:
- Repository monitoring
- Training cycle planning
- Task execution
- Artifact generation
- Error handling
- FastAPI endpoints

Run with: pytest tests/integration/test_repo_guardian.py -v
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import tempfile
import json

# Import path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary git repository."""
    import subprocess
    
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        capture_output=True,
    )
    
    # Create some files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "README.md").write_text("# Test Project")
    
    # Initial commit
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
    )
    
    return tmp_path


@pytest.fixture
def guardian(temp_repo):
    """Create RepoGuardian instance."""
    from antigravity_agent.repo_guardian import RepoGuardian, PipelineConfig
    
    config = PipelineConfig(
        auto_extract_data=False,
        auto_train=False,
        auto_deploy=False,
    )
    return RepoGuardian(temp_repo, config)


# =============================================================================
# Repository Monitoring Tests
# =============================================================================

class TestRepositoryMonitoring:
    """Tests for repository monitoring."""
    
    def test_monitor_detects_no_changes(self, guardian) -> None:
        """Test monitoring when no significant changes."""
        result = guardian.monitor_repository()
        
        assert "should_retrain" in result
        assert "reason" in result
        assert "metrics" in result
        assert result["should_retrain"] is False
    
    def test_monitor_detects_files_changed(self, guardian, temp_repo) -> None:
        """Test monitoring detects file changes."""
        import subprocess
        
        # Create many files
        for i in range(25):
            (temp_repo / f"file_{i}.py").write_text(f"# File {i}")
        
        subprocess.run(["git", "add", "."], cwd=temp_repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add many files"],
            cwd=temp_repo,
            capture_output=True,
        )
        
        result = guardian.monitor_repository()
        
        assert result["metrics"]["files_changed"] > 0
    
    def test_monitor_detects_release_tag(self, guardian, temp_repo) -> None:
        """Test monitoring detects release tags."""
        import subprocess
        
        subprocess.run(
            ["git", "tag", "v1.0.0"],
            cwd=temp_repo,
            capture_output=True,
        )
        
        result = guardian.monitor_repository()
        
        assert result["metrics"]["release_tag_detected"] is True
        assert result["should_retrain"] is True


# =============================================================================
# Training Cycle Planning Tests
# =============================================================================

class TestTrainingCyclePlanning:
    """Tests for training cycle planning."""
    
    def test_plan_returns_tasks(self, guardian) -> None:
        """Test plan includes all required tasks."""
        plan = guardian.plan_training_cycle()
        
        assert "plan" in plan
        assert "estimated_duration_minutes" in plan
        assert len(plan["plan"]) == 6
    
    def test_plan_task_structure(self, guardian) -> None:
        """Test plan tasks have correct structure."""
        plan = guardian.plan_training_cycle()
        
        for task in plan["plan"]:
            assert "id" in task
            assert "name" in task
            assert "description" in task
            assert "type" in task
            assert "estimated_minutes" in task
    
    def test_plan_estimated_duration(self, guardian) -> None:
        """Test estimated duration is reasonable."""
        plan = guardian.plan_training_cycle()
        
        duration = plan["estimated_duration_minutes"]
        assert duration > 0
        assert duration < 120  # Less than 2 hours


# =============================================================================
# Health Report Tests
# =============================================================================

class TestHealthReport:
    """Tests for repository health reports."""
    
    @pytest.mark.asyncio
    async def test_analyze_repository(self, guardian) -> None:
        """Test repository analysis."""
        report = await guardian.analyze_repository()
        
        assert report.repository_path is not None
        assert report.total_files >= 0
        assert report.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_health_report_to_markdown(self, guardian) -> None:
        """Test health report markdown export."""
        report = await guardian.analyze_repository()
        markdown = report.to_markdown()
        
        assert "# Repository Health Report" in markdown
        assert "Total Files" in markdown


# =============================================================================
# Task and Artifact Tests
# =============================================================================

class TestTaskAndArtifact:
    """Tests for Task and Artifact classes."""
    
    def test_task_to_dict(self) -> None:
        """Test Task serialization."""
        from antigravity_agent.repo_guardian import Task, TaskType, TaskStatus
        
        task = Task(
            id="1",
            name="test_task",
            description="Test",
            task_type=TaskType.EXTRACT_DATA,
        )
        
        d = task.to_dict()
        
        assert d["id"] == "1"
        assert d["name"] == "test_task"
        assert d["type"] == "extract_data"
    
    def test_artifact_to_markdown(self) -> None:
        """Test Artifact markdown export."""
        from antigravity_agent.repo_guardian import Artifact
        
        artifact = Artifact(
            name="test_artifact",
            artifact_type="report",
            content={"key": "value"},
            created_at=datetime.now(),
        )
        
        md = artifact.to_markdown()
        
        assert "# Artifact: test_artifact" in md
        assert "**Type:** report" in md


# =============================================================================
# Pipeline Execution Tests
# =============================================================================

class TestPipelineExecution:
    """Tests for pipeline execution."""
    
    def test_pause_resume(self, guardian) -> None:
        """Test pause and resume functionality."""
        guardian.pause()
        assert guardian._pipeline_running is False
        
        guardian.resume()
        assert guardian._pipeline_running is True
    
    @pytest.mark.asyncio
    async def test_run_pipeline_with_skip(self, guardian) -> None:
        """Test pipeline with all steps skipped."""
        result = await guardian.run_pipeline(
            skip_extraction=True,
            skip_training=True,
            skip_export=True,
            skip_deploy=True,
        )
        
        assert "started_at" in result
        assert "completed_at" in result
        assert result["success"] is True


# =============================================================================
# FastAPI Endpoint Tests
# =============================================================================

class TestFastAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self, temp_repo):
        """Create test client."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        
        from fastapi.testclient import TestClient
        from conductor.service import app, state
        
        state.jobs = {}
        return TestClient(app)
    
    def test_monitor_endpoint(self, client, temp_repo) -> None:
        """Test /v1/retrain/monitor endpoint."""
        response = client.get(
            "/v1/retrain/monitor",
            params={"project_path": str(temp_repo)},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "should_retrain" in data
    
    def test_retrain_endpoint_no_changes(self, client, temp_repo) -> None:
        """Test /v1/retrain when no changes detected."""
        response = client.post(
            "/v1/retrain",
            json={"project_path": str(temp_repo)},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["triggered"] is False
    
    def test_retrain_endpoint_force(self, client, temp_repo) -> None:
        """Test /v1/retrain with force=True."""
        response = client.post(
            "/v1/retrain",
            json={
                "project_path": str(temp_repo),
                "force": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["triggered"] is True
        assert data["job_id"] is not None
    
    def test_pause_endpoint_not_found(self, client) -> None:
        """Test pause endpoint with invalid job."""
        response = client.post("/v1/retrain/nonexistent/pause")
        
        assert response.status_code == 404
    
    def test_resume_endpoint_not_found(self, client) -> None:
        """Test resume endpoint with invalid job."""
        response = client.post("/v1/retrain/nonexistent/resume")
        
        assert response.status_code == 404


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_project_path(self) -> None:
        """Test error when project path doesn't exist."""
        from antigravity_agent.repo_guardian import RepoGuardian
        
        with pytest.raises(ValueError, match="does not exist"):
            RepoGuardian("/nonexistent/path")
    
    @pytest.mark.asyncio
    async def test_pipeline_handles_errors(self, guardian) -> None:
        """Test pipeline gracefully handles errors."""
        # Force config to auto train (will fail without data)
        guardian.config.auto_train = True
        
        result = await guardian.run_pipeline(
            skip_extraction=True,
        )
        
        # Should complete with error captured
        assert "completed_at" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
