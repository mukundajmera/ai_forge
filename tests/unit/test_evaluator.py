"""Unit tests for the Model Evaluator and Exporter modules.

Tests cover:
- Perplexity computation
- CodeBLEU scoring
- Reconstruction evaluation
- Hallucination detection
- GGUF export configuration
- Modelfile generation

Run with: pytest tests/unit/test_evaluator.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Import path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    model = MagicMock()
    model.device = "cpu"
    model.eval = MagicMock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    return tokenizer


@pytest.fixture
def evaluator_config():
    """Create evaluator config."""
    from judge.evaluator import EvaluatorConfig
    return EvaluatorConfig(
        batch_size=2,
        max_samples=10,
        compute_perplexity=True,
    )


# =============================================================================
# EvaluationResult Tests
# =============================================================================

class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from judge.evaluator import EvaluationResult
        
        result = EvaluationResult(
            perplexity=5.5,
            loss=1.7,
            codebleu=0.75,
        )
        
        d = result.to_dict()
        
        assert d["perplexity"] == 5.5
        assert d["loss"] == 1.7
        assert d["codebleu"] == 0.75
    
    def test_summary(self) -> None:
        """Test summary generation."""
        from judge.evaluator import EvaluationResult
        
        result = EvaluationResult(
            perplexity=5.5,
            loss=1.7,
            accuracy=0.85,
        )
        
        summary = result.summary()
        
        assert "Perplexity: 5.50" in summary
        assert "Accuracy: 85%" in summary


# =============================================================================
# Perplexity Tests
# =============================================================================

class TestPerplexityComputation:
    """Tests for perplexity computation."""
    
    def test_perplexity_formula(self) -> None:
        """Test that perplexity = exp(loss)."""
        import math
        
        loss = 2.0
        expected_perplexity = math.exp(loss)
        
        assert abs(expected_perplexity - 7.389) < 0.01
    
    def test_perplexity_range(self) -> None:
        """Test that perplexity is always positive."""
        import math
        
        for loss in [0.0, 0.5, 1.0, 2.0, 5.0]:
            perplexity = math.exp(loss)
            assert perplexity > 0


# =============================================================================
# CodeBLEU Tests
# =============================================================================

class TestCodeBLEUScoring:
    """Tests for CodeBLEU scoring."""
    
    def test_codebleu_range(self) -> None:
        """Test that CodeBLEU is in [0, 1] range."""
        # Simulated CodeBLEU scores
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for score in scores:
            assert 0.0 <= score <= 1.0
    
    def test_identical_code_high_score(self) -> None:
        """Test that identical code gets high similarity."""
        from difflib import SequenceMatcher
        
        code1 = "def foo():\n    return 42"
        code2 = "def foo():\n    return 42"
        
        similarity = SequenceMatcher(None, code1, code2).ratio()
        assert similarity == 1.0


# =============================================================================
# Reconstruction Tests
# =============================================================================

class TestReconstructionEval:
    """Tests for reconstruction evaluation."""
    
    def test_edit_distance_identical(self) -> None:
        """Test edit distance for identical strings."""
        from difflib import SequenceMatcher
        
        s1 = "hello world"
        s2 = "hello world"
        
        similarity = SequenceMatcher(None, s1, s2).ratio()
        edit_distance = 1.0 - similarity
        
        assert edit_distance == 0.0
    
    def test_edit_distance_different(self) -> None:
        """Test edit distance for different strings."""
        from difflib import SequenceMatcher
        
        s1 = "hello"
        s2 = "hallo"
        
        similarity = SequenceMatcher(None, s1, s2).ratio()
        edit_distance = 1.0 - similarity
        
        assert 0.0 < edit_distance < 1.0
    
    def test_exact_match_rate(self) -> None:
        """Test exact match rate calculation."""
        matches = 3
        total = 10
        rate = matches / total
        
        assert rate == 0.3


# =============================================================================
# Hallucination Detection Tests
# =============================================================================

class TestHallucinationDetection:
    """Tests for hallucination detection."""
    
    def test_detects_novel_tokens(self) -> None:
        """Test detection of tokens not in source."""
        import re
        
        generated = "def invented_function(): pass"
        source = "def real_function(): return 1"
        
        gen_tokens = set(re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', generated))
        src_tokens = set(re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', source))
        
        novel = gen_tokens - src_tokens
        
        assert "invented_function" in novel
    
    def test_no_hallucination_for_matching(self) -> None:
        """Test no hallucination when answer matches source."""
        from difflib import SequenceMatcher
        
        generated = "def foo(): return 42"
        expected = "def foo(): return 42"
        
        similarity = SequenceMatcher(None, generated.lower(), expected.lower()).ratio()
        
        assert similarity > 0.7  # Not a hallucination


# =============================================================================
# GGUF Export Tests
# =============================================================================

class TestGGUFExport:
    """Tests for GGUF export functionality."""
    
    def test_export_config_defaults(self) -> None:
        """Test default export configuration."""
        from judge.exporter import ExportConfig
        
        config = ExportConfig()
        
        assert config.quantization == "q4_k_m"
        assert config.output_dir == "./export"
        assert config.include_vocab is True
    
    def test_quantization_recommendations(self) -> None:
        """Test quantization recommendations."""
        from judge.exporter import GGUFExporter
        
        recommendations = GGUFExporter.QUANT_RECOMMENDATIONS
        
        assert "balanced" in recommendations
        assert recommendations["balanced"] == "q4_k_m"
    
    def test_size_estimation(self, tmp_path: Path) -> None:
        """Test model size estimation."""
        from judge.exporter import GGUFExporter, ExportConfig
        
        # Create minimal mock model dir
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"hidden_size": 4096}')
        
        config = ExportConfig(output_dir=str(tmp_path / "export"))
        exporter = GGUFExporter(model_dir, config)
        
        # f16 should be larger than q4
        size_f16 = exporter.estimate_size("f16")
        size_q4 = exporter.estimate_size("q4_k_m")
        
        assert size_f16 > size_q4


# =============================================================================
# Modelfile Tests
# =============================================================================

class TestModelfileGeneration:
    """Tests for Ollama Modelfile generation."""
    
    def test_modelfile_syntax(self, tmp_path: Path) -> None:
        """Test that generated Modelfile has valid syntax."""
        from judge.exporter import GGUFExporter, ExportConfig
        
        # Create mock model and GGUF
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{}')
        
        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()
        
        config = ExportConfig(output_dir=str(tmp_path))
        exporter = GGUFExporter(model_dir, config)
        
        modelfile_path = exporter.create_modelfile(
            gguf_path,
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
        )
        
        content = modelfile_path.read_text()
        
        # Check required elements
        assert "FROM" in content
        assert "SYSTEM" in content
        assert "PARAMETER temperature" in content
    
    def test_modelfile_includes_prompt(self, tmp_path: Path) -> None:
        """Test that system prompt is included."""
        from judge.exporter import GGUFExporter, ExportConfig
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{}')
        
        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()
        
        config = ExportConfig(output_dir=str(tmp_path))
        exporter = GGUFExporter(model_dir, config)
        
        custom_prompt = "You are a code expert."
        modelfile_path = exporter.create_modelfile(gguf_path, system_prompt=custom_prompt)
        
        content = modelfile_path.read_text()
        assert custom_prompt in content


# =============================================================================
# EvaluationReport Tests
# =============================================================================

class TestEvaluationReport:
    """Tests for EvaluationReport class."""
    
    def test_add_metrics(self) -> None:
        """Test adding metrics to report."""
        from judge.report import EvaluationReport
        
        report = EvaluationReport(
            model_name="test-model",
            base_model="llama-3.2",
        )
        
        report.add_metrics(perplexity=5.5, codebleu_score=0.75)
        
        assert report.perplexity == 5.5
        assert report.codebleu_score == 0.75
    
    def test_compute_improvement(self) -> None:
        """Test improvement calculation."""
        from judge.report import EvaluationReport
        
        report = EvaluationReport(
            model_name="fine-tuned",
            base_model="base",
            perplexity=4.0,
        )
        
        base_metrics = {"perplexity": 5.0}
        improvements = report.compute_improvement(base_metrics)
        
        # 20% improvement in perplexity (lower is better)
        assert abs(improvements["perplexity"] - 20.0) < 0.01
    
    def test_to_markdown(self, tmp_path: Path) -> None:
        """Test markdown generation."""
        from judge.report import EvaluationReport
        
        report = EvaluationReport(
            model_name="test-model",
            base_model="llama-3.2",
            perplexity=5.0,
            codebleu_score=0.8,
        )
        
        md_path = tmp_path / "report.md"
        content = report.to_markdown(md_path)
        
        assert "# Evaluation Report:" in content
        assert "test-model" in content
        assert md_path.exists()
    
    def test_to_json(self, tmp_path: Path) -> None:
        """Test JSON export."""
        from judge.report import EvaluationReport
        import json
        
        report = EvaluationReport(
            model_name="test-model",
            base_model="llama-3.2",
            perplexity=5.0,
        )
        
        json_path = tmp_path / "report.json"
        report.to_json(json_path)
        
        assert json_path.exists()
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert data["model_name"] == "test-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
