"""Unit tests for data pipeline module."""

import pytest
from pathlib import Path
from typing import Any

from ai_forge.data_pipeline.miner import CodeMiner, CodeChunk, MinerConfig
from ai_forge.data_pipeline.raft_generator import RAFTGenerator, RAFTConfig
from ai_forge.data_pipeline.validator import DataValidator, ValidatorConfig


class TestCodeMiner:
    """Tests for CodeMiner class."""
    
    def test_init_with_valid_path(self, tmp_path: Path) -> None:
        """Test initialization with valid path."""
        miner = CodeMiner(tmp_path)
        assert miner.project_path == tmp_path
    
    def test_init_with_invalid_path(self) -> None:
        """Test initialization with invalid path raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            CodeMiner("/nonexistent/path")
    
    def test_detect_language(self, tmp_path: Path) -> None:
        """Test language detection from file extension."""
        miner = CodeMiner(tmp_path)
        
        assert miner._detect_language(Path("test.py")) == "python"
        assert miner._detect_language(Path("test.js")) == "javascript"
        assert miner._detect_language(Path("test.ts")) == "typescript"
        assert miner._detect_language(Path("test.unknown")) is None
    
    def test_discover_files(self, tmp_path: Path) -> None:
        """Test file discovery."""
        # Create test files
        (tmp_path / "test.py").write_text("print('hello')")
        (tmp_path / "test.txt").write_text("not code")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "module.py").write_text("def foo(): pass")
        
        miner = CodeMiner(tmp_path, MinerConfig(languages=["python"]))
        files = list(miner.discover_files())
        
        assert len(files) == 2
        assert all(f.suffix == ".py" for f in files)
    
    def test_extract_from_file(self, tmp_path: Path) -> None:
        """Test extraction from a single file."""
        code = '''
def hello():
    """Say hello."""
    print("Hello, World!")
'''
        py_file = tmp_path / "test.py"
        py_file.write_text(code)
        
        miner = CodeMiner(tmp_path)
        chunks = miner.extract_from_file(py_file)
        
        assert len(chunks) >= 1
        assert chunks[0].language == "python"


class TestRAFTGenerator:
    """Tests for RAFTGenerator class."""
    
    @pytest.fixture
    def sample_chunks(self) -> list[Any]:
        """Create sample code chunks."""
        return [
            CodeChunk(
                content="def foo(): pass",
                language="python",
                file_path=Path("test.py"),
                start_line=1,
                end_line=1,
                chunk_type="function",
                name="foo",
                docstring="Foo function",
            ),
            CodeChunk(
                content="def bar(): pass",
                language="python",
                file_path=Path("test.py"),
                start_line=2,
                end_line=2,
                chunk_type="function",
                name="bar",
            ),
        ]
    
    def test_init(self, sample_chunks: list[Any]) -> None:
        """Test initialization."""
        generator = RAFTGenerator(sample_chunks)
        assert len(generator.chunks) == 2
    
    def test_generate_question(self, sample_chunks: list[Any]) -> None:
        """Test question generation."""
        generator = RAFTGenerator(sample_chunks)
        question = generator._generate_question(sample_chunks[0], "what_does")
        
        assert "foo" in question.lower()
    
    def test_generate_sample(self, sample_chunks: list[Any]) -> None:
        """Test sample generation."""
        generator = RAFTGenerator(sample_chunks, RAFTConfig(num_distractor_docs=1))
        sample = generator.generate_sample(sample_chunks[0])
        
        assert sample.question
        assert sample.oracle_doc
        assert sample.answer
    
    def test_generate_dataset(self, sample_chunks: list[Any]) -> None:
        """Test dataset generation."""
        generator = RAFTGenerator(sample_chunks)
        dataset = generator.generate_dataset(num_samples=5)
        
        assert len(dataset) == 5
        assert all("instruction" in d for d in dataset)


class TestDataValidator:
    """Tests for DataValidator class."""
    
    @pytest.fixture
    def valid_data(self) -> list[dict[str, str]]:
        """Create valid training data."""
        return [
            {
                "instruction": "What does this function do?",
                "input": "def foo(): pass",
                "output": "This function named foo does nothing; it just passes.",
            }
            for _ in range(150)
        ]
    
    @pytest.fixture
    def invalid_data(self) -> list[dict[str, str]]:
        """Create invalid training data."""
        return [
            {"instruction": "", "output": "test"},  # Empty instruction
            {"instruction": "test"},  # Missing output
            {"instruction": "test", "output": "x"},  # Output too short
        ]
    
    def test_validate_valid_data(self, valid_data: list[dict[str, str]]) -> None:
        """Test validation passes for valid data."""
        validator = DataValidator(valid_data)
        result = validator.validate_all()
        
        assert result.is_valid
        assert result.valid_samples > 0
    
    def test_validate_invalid_data(self, invalid_data: list[dict[str, str]]) -> None:
        """Test validation catches invalid data."""
        validator = DataValidator(invalid_data)
        result = validator.validate_all()
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_deduplicate(self, valid_data: list[dict[str, str]]) -> None:
        """Test deduplication removes duplicates."""
        # Add duplicates
        data = valid_data + valid_data[:10]
        
        validator = DataValidator(data)
        removed = validator.deduplicate()
        
        assert removed == 10
    
    def test_quality_score(self) -> None:
        """Test quality score computation."""
        data = [
            {
                "instruction": "What does this function do?",
                "output": "This is a detailed explanation with multiple sentences. It explains the functionality clearly.",
            }
        ]
        
        validator = DataValidator(data)
        score = validator.compute_quality_score(data[0])
        
        assert 0 <= score <= 1
