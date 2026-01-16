"""Test fixtures for AI Forge tests."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_training_data() -> list[dict[str, str]]:
    """Sample training data for tests."""
    return [
        {
            "instruction": "What does the greet function do?",
            "input": "def greet(name): return f'Hello, {name}!'",
            "output": "The greet function takes a name parameter and returns a greeting string.",
        },
        {
            "instruction": "Explain the Calculator class.",
            "input": "class Calculator: def add(self, a, b): return a + b",
            "output": "The Calculator class provides basic arithmetic operations like addition.",
        },
        {
            "instruction": "How does the add method work?",
            "input": "def add(self, a, b): return a + b",
            "output": "The add method takes two numbers and returns their sum.",
        },
    ]


@pytest.fixture
def sample_code_chunks() -> list[dict]:
    """Sample code chunks for tests."""
    from ai_forge.data_pipeline.miner import CodeChunk
    
    return [
        CodeChunk(
            content="def hello(): print('Hello')",
            language="python",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            chunk_type="function",
            name="hello",
            docstring="Print hello.",
        ),
        CodeChunk(
            content="def world(): print('World')",
            language="python",
            file_path=Path("test.py"),
            start_line=3,
            end_line=3,
            chunk_type="function",
            name="world",
        ),
    ]


@pytest.fixture
def mock_model_config() -> dict:
    """Mock model configuration for tests."""
    return {
        "model_name": "test/model",
        "max_seq_length": 512,
        "load_in_4bit": True,
        "use_pissa": True,
        "pissa_rank": 16,
        "num_epochs": 1,
        "batch_size": 2,
    }
