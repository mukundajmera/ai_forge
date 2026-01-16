"""Comprehensive unit tests for the Data Validator module.

Tests cover:
- Code block validation
- RAFT example validation
- Quality scoring
- Filtering
- Report generation

Run with: pytest tests/unit/test_validator.py -v
"""

import pytest
from pathlib import Path
from textwrap import dedent

# Import path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_pipeline.schemas.code_blocks import CodeBlock
from data_pipeline.schemas.metrics import (
    DataQualityMetrics,
    FilteringStats,
    QualityScore,
    ValidationConfig,
    ValidationResult,
)
from data_pipeline.schemas.raft_examples import (
    Difficulty,
    QuestionType,
    RAFTExample,
)
from data_pipeline.validator import (
    DataValidator,
    validate_code_block,
    validate_raft_example,
    validate_code_syntax,
    score_example_quality,
    score_data_quality,
    filter_by_quality,
    generate_quality_report,
    compute_text_hash,
    jaccard_similarity,
    compute_relevance,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_code_block() -> CodeBlock:
    """Create a valid code block."""
    return CodeBlock(
        path="src/utils.py",
        language="python",
        block_type="function",
        name="calculate_sum",
        docstring="Calculate the sum of two numbers.",
        source_code=dedent('''
            def calculate_sum(a: int, b: int) -> int:
                """Calculate the sum of two numbers."""
                return a + b
        ''').strip(),
    )


@pytest.fixture
def invalid_code_block() -> CodeBlock:
    """Create an invalid code block (syntax error)."""
    return CodeBlock(
        path="src/broken.py",
        language="python",
        block_type="function",
        name="broken_func",
        docstring=None,
        source_code="def broken(:\n    return",
    )


@pytest.fixture
def valid_raft_example(valid_code_block: CodeBlock) -> RAFTExample:
    """Create a valid RAFT example."""
    return RAFTExample(
        question="What does the calculate_sum function do?",
        question_type=QuestionType.PURPOSE,
        oracle_documents=[valid_code_block],
        distractor_documents=[],
        reasoning="Looking at the calculate_sum function, we can see it takes two parameters and returns their sum.",
        final_answer="It calculates and returns the sum of two numbers.",
        difficulty=Difficulty.EASY,
    )


@pytest.fixture
def valid_raft_example_dict(valid_code_block: CodeBlock) -> dict:
    """Create a valid RAFT example as dict."""
    return {
        "question": "What does the calculate_sum function do?",
        "question_type": "purpose",
        "oracle_documents": [valid_code_block.to_dict()],
        "distractor_documents": [],
        "reasoning": "Looking at the calculate_sum function, we see it returns a + b.",
        "final_answer": "It adds two numbers together.",
        "difficulty": "easy",
    }


@pytest.fixture
def sample_dataset(valid_code_block: CodeBlock) -> list[dict]:
    """Create a sample dataset for testing."""
    datasets = []
    for i in range(10):
        datasets.append({
            "question": f"Question about function {i}",
            "question_type": "purpose",
            "oracle_documents": [valid_code_block.to_dict()],
            "distractor_documents": [],
            "reasoning": f"Looking at function {i}, we can analyze...",
            "final_answer": f"Answer for function {i}.",
            "difficulty": ["easy", "medium", "hard"][i % 3],
        })
    return datasets


# =============================================================================
# Text Similarity Tests
# =============================================================================

class TestTextSimilarity:
    """Tests for text similarity functions."""
    
    def test_jaccard_identical(self) -> None:
        """Test Jaccard with identical strings."""
        assert jaccard_similarity("hello world", "hello world") == 1.0
    
    def test_jaccard_different(self) -> None:
        """Test Jaccard with completely different strings."""
        assert jaccard_similarity("apple banana", "cat dog") == 0.0
    
    def test_jaccard_partial(self) -> None:
        """Test Jaccard with partial overlap."""
        sim = jaccard_similarity("apple banana cherry", "banana cherry date")
        assert 0.3 < sim < 0.7
    
    def test_compute_text_hash(self) -> None:
        """Test hash computation."""
        hash1 = compute_text_hash("Hello World")
        hash2 = compute_text_hash("hello   world")  # Whitespace normalized
        assert hash1 == hash2
    
    def test_compute_relevance(self) -> None:
        """Test relevance computation."""
        query = "What does calculate_sum do?"
        doc = "def calculate_sum(a, b): return a + b"
        
        rel = compute_relevance(query, doc)
        assert rel > 0.3  # Should have relevance


# =============================================================================
# Code Syntax Validation Tests
# =============================================================================

class TestCodeSyntaxValidation:
    """Tests for code syntax validation."""
    
    def test_valid_python(self) -> None:
        """Test valid Python syntax."""
        code = "def hello(): return 42"
        is_valid, error = validate_code_syntax(code, "python")
        assert is_valid
        assert error is None
    
    def test_invalid_python(self) -> None:
        """Test invalid Python syntax."""
        code = "def broken(:\n    return"
        is_valid, error = validate_code_syntax(code, "python")
        assert not is_valid
        assert error is not None
        assert "Syntax error" in error
    
    def test_javascript_balanced_braces(self) -> None:
        """Test JavaScript brace balancing."""
        valid_code = "function test() { return 42; }"
        is_valid, _ = validate_code_syntax(valid_code, "javascript")
        assert is_valid
        
        invalid_code = "function test() { return 42;"
        is_valid, error = validate_code_syntax(invalid_code, "javascript")
        assert not is_valid
        assert "braces" in error.lower()


# =============================================================================
# Code Block Validation Tests
# =============================================================================

class TestCodeBlockValidation:
    """Tests for code block validation."""
    
    def test_valid_block(self, valid_code_block: CodeBlock) -> None:
        """Test validation of valid block."""
        is_valid, errors = validate_code_block(valid_code_block)
        assert is_valid
        assert len(errors) == 0
    
    def test_empty_source(self) -> None:
        """Test validation of empty source code."""
        block = CodeBlock(
            path="test.py",
            language="python",
            block_type="function",
            name="test",
            docstring=None,
            source_code="",
        )
        
        is_valid, errors = validate_code_block(block)
        assert not is_valid
        assert any("empty" in e.lower() for e in errors)
    
    def test_too_short(self) -> None:
        """Test validation of too-short code."""
        block = CodeBlock(
            path="test.py",
            language="python",
            block_type="function",
            name="x",
            docstring=None,
            source_code="x=1",  # Very short
        )
        
        config = ValidationConfig(min_code_tokens=50)
        is_valid, errors = validate_code_block(block, config)
        assert not is_valid
        assert any("few tokens" in e.lower() for e in errors)
    
    def test_syntax_error(self, invalid_code_block: CodeBlock) -> None:
        """Test validation catches syntax errors."""
        is_valid, errors = validate_code_block(invalid_code_block)
        assert not is_valid
        assert any("syntax" in e.lower() for e in errors)
    
    def test_missing_docstring(self) -> None:
        """Test validation with required docstring."""
        block = CodeBlock(
            path="test.py",
            language="python",
            block_type="function",
            name="test",
            docstring=None,
            source_code="def test(): return 42",
        )
        
        config = ValidationConfig(require_docstring=True)
        is_valid, errors = validate_code_block(block, config)
        assert not is_valid
        assert any("docstring" in e.lower() for e in errors)


# =============================================================================
# RAFT Example Validation Tests
# =============================================================================

class TestRAFTExampleValidation:
    """Tests for RAFT example validation."""
    
    def test_valid_example(self, valid_raft_example: RAFTExample) -> None:
        """Test validation of valid example."""
        is_valid, errors = validate_raft_example(valid_raft_example)
        assert is_valid
        assert len(errors) == 0
    
    def test_valid_example_dict(self, valid_raft_example_dict: dict) -> None:
        """Test validation of dict example."""
        is_valid, errors = validate_raft_example(valid_raft_example_dict)
        assert is_valid
    
    def test_missing_question(self, valid_code_block: CodeBlock) -> None:
        """Test validation with missing question."""
        example = RAFTExample(
            question="",
            question_type=QuestionType.PURPOSE,
            oracle_documents=[valid_code_block],
            distractor_documents=[],
            reasoning="Some reasoning.",
            final_answer="Some answer.",
            difficulty=Difficulty.EASY,
        )
        
        is_valid, errors = validate_raft_example(example)
        assert not is_valid
        assert any("question" in e.lower() for e in errors)
    
    def test_no_oracle_documents(self) -> None:
        """Test validation with no oracle documents."""
        example = RAFTExample(
            question="What does this do?",
            question_type=QuestionType.PURPOSE,
            oracle_documents=[],
            distractor_documents=[],
            reasoning="Looking at the code...",
            final_answer="Answer.",
            difficulty=Difficulty.EASY,
        )
        
        is_valid, errors = validate_raft_example(example)
        assert not is_valid
        assert any("oracle" in e.lower() for e in errors)
    
    def test_missing_reasoning(self, valid_code_block: CodeBlock) -> None:
        """Test validation with missing reasoning."""
        example = RAFTExample(
            question="What does calculate_sum do?",
            question_type=QuestionType.PURPOSE,
            oracle_documents=[valid_code_block],
            distractor_documents=[],
            reasoning="",
            final_answer="It adds numbers.",
            difficulty=Difficulty.EASY,
        )
        
        is_valid, errors = validate_raft_example(example)
        assert not is_valid
        assert any("reasoning" in e.lower() for e in errors)


# =============================================================================
# Quality Scoring Tests  
# =============================================================================

class TestQualityScoring:
    """Tests for quality scoring."""
    
    def test_score_example(self, valid_raft_example: RAFTExample) -> None:
        """Test scoring a valid example."""
        score = score_example_quality(valid_raft_example)
        
        assert isinstance(score, QualityScore)
        assert 0.0 <= score.overall <= 1.0
        assert 0.0 <= score.relevance <= 1.0
        assert 0.0 <= score.clarity <= 1.0
    
    def test_score_dataset(self, sample_dataset: list[dict]) -> None:
        """Test scoring a dataset."""
        metrics = score_data_quality(sample_dataset)
        
        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.total_examples == 10
        assert 0.0 <= metrics.quality_score_mean <= 1.0
    
    def test_score_empty_dataset(self) -> None:
        """Test scoring empty dataset."""
        metrics = score_data_quality([])
        
        assert metrics.total_examples == 0
        assert metrics.quality_score_mean == 0.0
    
    def test_difficulty_distribution(self, sample_dataset: list[dict]) -> None:
        """Test difficulty distribution counting."""
        metrics = score_data_quality(sample_dataset)
        
        # Dataset has alternating difficulties
        assert sum(metrics.difficulty_distribution.values()) == 10


# =============================================================================
# Filtering Tests
# =============================================================================

class TestFiltering:
    """Tests for dataset filtering."""
    
    def test_filter_keeps_valid(self, sample_dataset: list[dict]) -> None:
        """Test that filtering keeps valid examples."""
        config = ValidationConfig(min_quality_score=0.0)  # Accept all
        filtered, stats = filter_by_quality(sample_dataset, config)
        
        assert len(filtered) > 0
        assert stats.before_count == 10
    
    def test_filter_removes_invalid(self, sample_dataset: list[dict]) -> None:
        """Test that filtering removes invalid examples."""
        # Add invalid example
        invalid = {
            "question": "",  # Empty question
            "question_type": "purpose",
            "oracle_documents": [],  # No oracle
            "distractor_documents": [],
            "reasoning": "",  # No reasoning
            "final_answer": "",  # No answer
            "difficulty": "easy",
        }
        
        dataset = sample_dataset + [invalid]
        filtered, stats = filter_by_quality(dataset)
        
        assert stats.removed_count >= 1
        assert stats.before_count == 11
    
    def test_filtering_stats(self, sample_dataset: list[dict]) -> None:
        """Test filtering statistics."""
        config = ValidationConfig(min_quality_score=0.99)  # Very strict
        filtered, stats = filter_by_quality(sample_dataset, config)
        
        assert stats.before_count == len(sample_dataset)
        assert stats.after_count == len(filtered)
        assert stats.removed_count == stats.before_count - stats.after_count
        assert 0.0 <= stats.removal_rate <= 1.0


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Tests for quality report generation."""
    
    def test_generate_report(self, sample_dataset: list[dict]) -> None:
        """Test report generation."""
        report = generate_quality_report(sample_dataset)
        
        assert isinstance(report, str)
        assert "# Data Quality Report" in report
        assert "Summary Metrics" in report
        assert "Difficulty Distribution" in report
    
    def test_report_contains_metrics(self, sample_dataset: list[dict]) -> None:
        """Test that report contains key metrics."""
        report = generate_quality_report(sample_dataset)
        
        assert "Valid Examples" in report
        assert "Quality Score" in report
    
    def test_report_contains_histogram(self, sample_dataset: list[dict]) -> None:
        """Test that report contains histogram."""
        report = generate_quality_report(sample_dataset)
        
        assert "Quality Score Distribution" in report
        assert "█" in report  # Histogram bars
    
    def test_report_worst_examples(self, sample_dataset: list[dict]) -> None:
        """Test that report includes worst examples."""
        report = generate_quality_report(sample_dataset, include_worst_examples=3)
        
        assert "Lowest Quality Examples" in report
    
    def test_report_recommendations(self, sample_dataset: list[dict]) -> None:
        """Test that report includes recommendations."""
        report = generate_quality_report(sample_dataset)
        
        assert "Recommendations" in report


# =============================================================================
# DataValidator Class Tests
# =============================================================================

class TestDataValidatorClass:
    """Tests for the DataValidator class."""
    
    def test_validator_init(self) -> None:
        """Test validator initialization."""
        validator = DataValidator()
        assert validator.config is not None
    
    def test_validator_config(self) -> None:
        """Test validator with custom config."""
        config = ValidationConfig(min_quality_score=0.9)
        validator = DataValidator(config)
        assert validator.config.min_quality_score == 0.9
    
    def test_validate_code_block(self, valid_code_block: CodeBlock) -> None:
        """Test code block validation method."""
        validator = DataValidator()
        result = validator.validate_code_block(valid_code_block)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
    
    def test_validate_raft_example(self, valid_raft_example: RAFTExample) -> None:
        """Test RAFT example validation method."""
        validator = DataValidator()
        result = validator.validate_raft_example(valid_raft_example)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
    
    def test_score_example(self, valid_raft_example: RAFTExample) -> None:
        """Test example scoring method."""
        validator = DataValidator()
        score = validator.score_example(valid_raft_example)
        
        assert isinstance(score, QualityScore)
    
    def test_score_dataset(self, sample_dataset: list[dict]) -> None:
        """Test dataset scoring method."""
        validator = DataValidator()
        metrics = validator.score_dataset(sample_dataset)
        
        assert isinstance(metrics, DataQualityMetrics)
    
    def test_filter_dataset(self, sample_dataset: list[dict]) -> None:
        """Test dataset filtering method."""
        validator = DataValidator()
        filtered, stats = validator.filter_dataset(sample_dataset)
        
        assert isinstance(stats, FilteringStats)
    
    def test_generate_report(self, sample_dataset: list[dict]) -> None:
        """Test report generation method."""
        validator = DataValidator()
        report = validator.generate_report(sample_dataset)
        
        assert isinstance(report, str)
        assert len(report) > 100


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_dataset_metrics(self) -> None:
        """Test metrics with empty dataset."""
        metrics = score_data_quality([])
        
        assert metrics.total_examples == 0
        assert metrics.valid_examples == 0
    
    def test_single_example(self, valid_raft_example: RAFTExample) -> None:
        """Test with single example."""
        metrics = score_data_quality([valid_raft_example])
        
        assert metrics.total_examples == 1
    
    def test_all_invalid(self) -> None:
        """Test with all invalid examples."""
        invalid_examples = [
            {"question": "", "oracle_documents": [], "distractor_documents": [], "reasoning": "", "final_answer": "", "difficulty": "easy", "question_type": "purpose"}
            for _ in range(5)
        ]
        
        filtered, stats = filter_by_quality(invalid_examples)
        
        assert len(filtered) == 0
        assert stats.removal_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
