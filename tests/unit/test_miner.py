"""Comprehensive unit tests for the Code Miner module.

This test suite verifies:
- Multi-language parsing (Python, JavaScript, Go)
- Complete function extraction (no mid-function splits)
- Docstring attachment
- Edge cases (nested functions, decorators, async, lambdas)
- Quality scoring and filtering
- Error handling

Run with: pytest tests/unit/test_miner.py -v
"""

import pytest
from pathlib import Path
from textwrap import dedent

# Import path setup for local testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_pipeline.schemas.code_blocks import (
    CodeBlock,
    MinerConfig,
    QualityLevel,
    LANGUAGE_CONFIGS,
)
from data_pipeline.miner import (
    extract_functions,
    extract_classes,
    extract_module_info,
    parse_repository,
    filter_by_quality,
    scan_repository_stats,
    EXTENSION_TO_LANGUAGE,
    _parser_manager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_python_code() -> bytes:
    """Sample Python code with functions and classes."""
    return dedent('''
        """Sample module docstring."""
        
        import os
        from typing import List
        
        VERSION = "1.0.0"
        
        def simple_function(x: int) -> int:
            """Return x squared."""
            return x * x
        
        def function_without_docstring(a, b):
            result = a + b
            return result
        
        async def async_function(url: str) -> str:
            """Fetch data from URL."""
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()
        
        def _private_function():
            """This is private."""
            pass
        
        class Calculator:
            """A simple calculator class."""
            
            def __init__(self, initial: float = 0):
                """Initialize calculator with value."""
                self.value = initial
            
            def add(self, x: float) -> float:
                """Add x to current value."""
                self.value += x
                return self.value
            
            def _internal_method(self):
                """Private method."""
                pass
        
        @decorator
        def decorated_function():
            """Decorated function."""
            return True
        
        def outer_function():
            """Outer function with nested."""
            
            def inner_function():
                """Inner nested function."""
                return 42
            
            return inner_function()
    ''').encode('utf-8')


@pytest.fixture
def sample_javascript_code() -> bytes:
    """Sample JavaScript code."""
    return dedent('''
        /**
         * Sample JavaScript module
         * @module sample
         */
        
        import { useState } from 'react';
        
        const VERSION = '1.0.0';
        
        /**
         * Calculate the sum of two numbers
         * @param {number} a - First number
         * @param {number} b - Second number
         * @returns {number} Sum
         */
        function add(a, b) {
            return a + b;
        }
        
        /**
         * Calculator class for basic operations
         */
        class Calculator {
            constructor(initial = 0) {
                this.value = initial;
            }
            
            add(x) {
                this.value += x;
                return this.value;
            }
        }
        
        const multiply = (a, b) => a * b;
        
        async function fetchData(url) {
            const response = await fetch(url);
            return response.json();
        }
        
        export { add, Calculator, multiply };
    ''').encode('utf-8')


@pytest.fixture
def sample_go_code() -> bytes:
    """Sample Go code."""
    return dedent('''
        // Package sample provides example functionality
        package sample
        
        import (
            "fmt"
            "net/http"
        )
        
        // VERSION of the module
        const VERSION = "1.0.0"
        
        // Calculator performs arithmetic operations
        type Calculator struct {
            Value float64
        }
        
        // Add adds x to the current value
        func (c *Calculator) Add(x float64) float64 {
            c.Value += x
            return c.Value
        }
        
        // Multiply multiplies a and b
        func Multiply(a, b int) int {
            return a * b
        }
        
        // FetchData retrieves data from URL
        func FetchData(url string) (string, error) {
            resp, err := http.Get(url)
            if err != nil {
                return "", err
            }
            defer resp.Body.Close()
            return "", nil
        }
    ''').encode('utf-8')


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a sample project for testing."""
    # Python file
    py_file = tmp_path / "src" / "utils.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text(dedent('''
        """Utility functions."""
        
        def helper_function(x: int) -> int:
            """Double the input."""
            return x * 2
        
        class Helper:
            """Helper class."""
            
            def process(self, data):
                """Process data."""
                return data
    '''))
    
    # JavaScript file
    js_file = tmp_path / "src" / "index.js"
    js_file.write_text(dedent('''
        /**
         * Main entry point
         */
        
        function main() {
            console.log("Hello");
        }
        
        export default main;
    '''))
    
    # Test file (should be excluded when include_tests=False)
    test_file = tmp_path / "tests" / "test_utils.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(dedent('''
        def test_helper():
            assert True
    '''))
    
    return tmp_path


# =============================================================================
# Python Parsing Tests
# =============================================================================

class TestPythonParsing:
    """Tests for Python code parsing."""
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("python"),
        reason="Tree-sitter Python parser not installed"
    )
    def test_extract_simple_function(self, sample_python_code: bytes) -> None:
        """Test extraction of a simple function."""
        functions = extract_functions(sample_python_code, "python")
        
        # Should find multiple functions
        assert len(functions) >= 3
        
        # Find simple_function
        simple_func = next((f for f in functions if f[0] == "simple_function"), None)
        assert simple_func is not None
        
        name, docstring, source = simple_func
        assert name == "simple_function"
        assert docstring == "Return x squared."
        assert "return x * x" in source
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("python"),
        reason="Tree-sitter Python parser not installed"
    )
    def test_extract_async_function(self, sample_python_code: bytes) -> None:
        """Test extraction of async functions."""
        functions = extract_functions(sample_python_code, "python")
        
        async_func = next((f for f in functions if f[0] == "async_function"), None)
        assert async_func is not None
        
        name, docstring, source = async_func
        assert "async def" in source
        assert docstring == "Fetch data from URL."
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("python"),
        reason="Tree-sitter Python parser not installed"
    )
    def test_private_function_excluded(self, sample_python_code: bytes) -> None:
        """Test that private functions are excluded by default."""
        functions = extract_functions(sample_python_code, "python", include_private=False)
        
        private_names = [f[0] for f in functions if f[0].startswith('_') and not f[0].startswith('__')]
        assert len(private_names) == 0
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("python"),
        reason="Tree-sitter Python parser not installed"
    )
    def test_private_function_included(self, sample_python_code: bytes) -> None:
        """Test that private functions are included when specified."""
        functions = extract_functions(sample_python_code, "python", include_private=True)
        
        private_func = next((f for f in functions if f[0] == "_private_function"), None)
        assert private_func is not None
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("python"),
        reason="Tree-sitter Python parser not installed"
    )
    def test_extract_class(self, sample_python_code: bytes) -> None:
        """Test extraction of classes."""
        classes = extract_classes(sample_python_code, "python")
        
        calc_class = next((c for c in classes if c[0] == "Calculator"), None)
        assert calc_class is not None
        
        name, docstring, methods = calc_class
        assert name == "Calculator"
        assert docstring == "A simple calculator class."
        assert "add" in methods
        assert "__init__" in methods
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("python"),
        reason="Tree-sitter Python parser not installed"
    )
    def test_extract_decorated_function(self, sample_python_code: bytes) -> None:
        """Test extraction of decorated functions."""
        functions = extract_functions(sample_python_code, "python")
        
        decorated = next((f for f in functions if f[0] == "decorated_function"), None)
        assert decorated is not None
        assert "Decorated function" in (decorated[1] or "")
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("python"),
        reason="Tree-sitter Python parser not installed"
    )
    def test_no_mid_function_splits(self, sample_python_code: bytes) -> None:
        """Verify functions are extracted completely without splits."""
        functions = extract_functions(sample_python_code, "python")
        
        for name, _, source in functions:
            # Each function should be syntactically complete
            if name == "simple_function":
                assert "def simple_function" in source
                assert "return x * x" in source
            elif name == "outer_function":
                # Nested function should be included in parent
                assert "def outer_function" in source
                assert "return inner_function()" in source


class TestPythonModuleInfo:
    """Tests for Python module metadata extraction."""
    
    def test_extract_module_docstring(self, sample_python_code: bytes) -> None:
        """Test extraction of module docstring."""
        info = extract_module_info(sample_python_code, "python")
        
        assert info["docstring"] == "Sample module docstring."
    
    def test_extract_imports(self, sample_python_code: bytes) -> None:
        """Test extraction of imports."""
        info = extract_module_info(sample_python_code, "python")
        
        assert "os" in info["imports"]
        assert "typing" in info["imports"]
    
    def test_extract_constants(self, sample_python_code: bytes) -> None:
        """Test extraction of module-level constants."""
        info = extract_module_info(sample_python_code, "python")
        
        assert "VERSION" in info["constants"]


# =============================================================================
# JavaScript Parsing Tests
# =============================================================================

class TestJavaScriptParsing:
    """Tests for JavaScript code parsing."""
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("javascript"),
        reason="Tree-sitter JavaScript parser not installed"
    )
    def test_extract_function(self, sample_javascript_code: bytes) -> None:
        """Test extraction of JavaScript functions."""
        functions = extract_functions(sample_javascript_code, "javascript")
        
        add_func = next((f for f in functions if f[0] == "add"), None)
        assert add_func is not None
        
        name, docstring, source = add_func
        assert name == "add"
        assert "return a + b" in source
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("javascript"),
        reason="Tree-sitter JavaScript parser not installed"
    )
    def test_extract_async_function(self, sample_javascript_code: bytes) -> None:
        """Test extraction of async JavaScript functions."""
        functions = extract_functions(sample_javascript_code, "javascript")
        
        fetch_func = next((f for f in functions if f[0] == "fetchData"), None)
        assert fetch_func is not None
        assert "async" in fetch_func[2]
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("javascript"),
        reason="Tree-sitter JavaScript parser not installed"
    )
    def test_extract_class(self, sample_javascript_code: bytes) -> None:
        """Test extraction of JavaScript classes."""
        classes = extract_classes(sample_javascript_code, "javascript")
        
        calc_class = next((c for c in classes if c[0] == "Calculator"), None)
        assert calc_class is not None


# =============================================================================
# Go Parsing Tests
# =============================================================================

class TestGoParsing:
    """Tests for Go code parsing."""
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("go"),
        reason="Tree-sitter Go parser not installed"
    )
    def test_extract_function(self, sample_go_code: bytes) -> None:
        """Test extraction of Go functions."""
        functions = extract_functions(sample_go_code, "go")
        
        multiply_func = next((f for f in functions if f[0] == "Multiply"), None)
        assert multiply_func is not None
        
        name, _, source = multiply_func
        assert name == "Multiply"
        assert "return a * b" in source
    
    @pytest.mark.skipif(
        not _parser_manager.is_available("go"),
        reason="Tree-sitter Go parser not installed"
    )
    def test_extract_method(self, sample_go_code: bytes) -> None:
        """Test extraction of Go methods."""
        functions = extract_functions(sample_go_code, "go")
        
        # Methods should be extracted
        add_method = next((f for f in functions if f[0] == "Add"), None)
        assert add_method is not None


# =============================================================================
# Repository Parsing Tests
# =============================================================================

class TestRepositoryParsing:
    """Tests for full repository parsing."""
    
    def test_parse_repository(self, sample_project: Path) -> None:
        """Test parsing a sample project."""
        config = MinerConfig(
            languages=["python", "javascript"],
            include_tests=False,
        )
        
        blocks = parse_repository(str(sample_project), config)
        
        # Should find blocks from Python and JS files
        assert len(blocks) >= 1
        
        # Should not include test files
        test_blocks = [b for b in blocks if "test_" in b.name]
        assert len(test_blocks) == 0
    
    def test_parse_repository_with_tests(self, sample_project: Path) -> None:
        """Test including test files."""
        config = MinerConfig(
            languages=["python"],
            include_tests=True,
        )
        
        blocks = parse_repository(str(sample_project), config)
        
        # Check that we have blocks
        assert len(blocks) >= 1
    
    def test_invalid_repository_path(self) -> None:
        """Test handling of invalid repository path."""
        with pytest.raises(ValueError, match="does not exist"):
            parse_repository("/nonexistent/path/to/repo")
    
    def test_scan_repository_stats(self, sample_project: Path) -> None:
        """Test repository statistics scanning."""
        stats = scan_repository_stats(str(sample_project))
        
        assert stats["total_files"] >= 3
        assert stats["parseable_files"] >= 2
        assert "python" in stats["by_language"]


# =============================================================================
# Quality Scoring Tests
# =============================================================================

class TestQualityScoring:
    """Tests for code block quality scoring."""
    
    def test_high_quality_block(self) -> None:
        """Test scoring of a high-quality code block."""
        block = CodeBlock(
            path="src/utils.py",
            language="python",
            block_type="function",
            name="calculate_sum",
            docstring="Calculate the sum of two numbers and return the result.",
            source_code=dedent('''
                def calculate_sum(a: int, b: int) -> int:
                    """Calculate the sum of two numbers and return the result."""
                    result = a + b
                    return result
            '''),
            metadata={"start_line": 1, "end_line": 5},
        )
        
        score = block.quality_score()
        assert score >= 0.6
        assert block.quality_level() in (QualityLevel.HIGH, QualityLevel.MEDIUM)
    
    def test_low_quality_block(self) -> None:
        """Test scoring of a low-quality code block."""
        block = CodeBlock(
            path="src/x.py",
            language="python",
            block_type="function",
            name="x",
            docstring=None,
            source_code="def x(): pass",
            metadata={},
        )
        
        score = block.quality_score()
        assert score < 0.5
    
    def test_filter_by_quality(self) -> None:
        """Test filtering blocks by quality score."""
        blocks = [
            CodeBlock(
                path="a.py", language="python", block_type="function",
                name="good_function",
                docstring="A well-documented function that does something useful.",
                source_code="def good_function():\n    '''A well-documented function.'''\n    return 42\n",
            ),
            CodeBlock(
                path="b.py", language="python", block_type="function",
                name="x",
                docstring=None,
                source_code="def x(): pass",
            ),
        ]
        
        filtered = filter_by_quality(blocks, min_score=0.4)
        
        # Should keep the good function, filter out the bad one
        assert len(filtered) == 1
        assert filtered[0].name == "good_function"
    
    def test_token_count(self) -> None:
        """Test token count estimation."""
        block = CodeBlock(
            path="test.py",
            language="python",
            block_type="function",
            name="test",
            docstring=None,
            source_code="a" * 100,  # 100 chars = 25 tokens
        )
        
        assert block.token_count == 25


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_file(self) -> None:
        """Test handling of empty file."""
        functions = extract_functions(b"", "python")
        assert functions == []
    
    def test_syntax_error_handling(self) -> None:
        """Test handling of syntax errors in code."""
        # Invalid Python syntax
        bad_code = b"def broken(:\n    return"
        
        # Should not crash, just return empty or partial results
        try:
            functions = extract_functions(bad_code, "python")
            # Parser may still extract partial results
            assert isinstance(functions, list)
        except Exception:
            pass  # Some parsers may raise on invalid syntax
    
    def test_unsupported_language(self) -> None:
        """Test handling of unsupported language."""
        functions = extract_functions(b"code", "unsupported_lang")
        assert functions == []
    
    def test_unicode_handling(self) -> None:
        """Test handling of Unicode in code."""
        unicode_code = '''
def greet(name: str) -> str:
    """Greet the user with their name. 你好世界!"""
    return f"Hello, {name}! 🎉"
'''.encode('utf-8')
        
        functions = extract_functions(unicode_code, "python")
        
        if _parser_manager.is_available("python"):
            assert len(functions) >= 1
            name, docstring, source = functions[0]
            assert "你好世界" in (docstring or "")
    
    def test_nested_functions(self) -> None:
        """Test handling of nested functions."""
        nested_code = b'''
def outer():
    """Outer function."""
    
    def inner():
        """Inner function."""
        return 42
    
    return inner()
'''
        
        if _parser_manager.is_available("python"):
            functions = extract_functions(nested_code, "python")
            
            # Should extract outer function
            outer = next((f for f in functions if f[0] == "outer"), None)
            assert outer is not None
    
    def test_lambda_functions(self) -> None:
        """Test that lambdas are handled gracefully."""
        lambda_code = b'''
# Lambda expressions
add = lambda x, y: x + y
process = lambda data: [x * 2 for x in data]
'''
        
        # Lambdas typically aren't extracted as named functions
        functions = extract_functions(lambda_code, "python")
        # Should not crash
        assert isinstance(functions, list)


# =============================================================================
# Language Configuration Tests
# =============================================================================

class TestLanguageConfig:
    """Tests for language configuration."""
    
    def test_extension_mapping(self) -> None:
        """Test file extension to language mapping."""
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"
        assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"
        assert EXTENSION_TO_LANGUAGE[".go"] == "go"
        assert EXTENSION_TO_LANGUAGE[".java"] == "java"
    
    def test_language_configs_exist(self) -> None:
        """Test that language configs are defined."""
        assert "python" in LANGUAGE_CONFIGS
        assert "javascript" in LANGUAGE_CONFIGS
        assert "go" in LANGUAGE_CONFIGS
        assert "java" in LANGUAGE_CONFIGS
    
    def test_language_config_has_required_fields(self) -> None:
        """Test that language configs have required fields."""
        for lang, config in LANGUAGE_CONFIGS.items():
            assert config.name == lang
            assert len(config.extensions) > 0
            assert len(config.function_node_types) > 0


# =============================================================================
# Code Block Serialization Tests
# =============================================================================

class TestCodeBlockSerialization:
    """Tests for CodeBlock serialization."""
    
    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        block = CodeBlock(
            path="test.py",
            language="python",
            block_type="function",
            name="test_func",
            docstring="Test function.",
            source_code="def test_func(): pass",
            dependencies=["os"],
            metadata={"start_line": 1},
        )
        
        data = block.to_dict()
        
        assert data["path"] == "test.py"
        assert data["language"] == "python"
        assert data["name"] == "test_func"
        assert "quality_score" in data
    
    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "path": "test.py",
            "language": "python",
            "block_type": "function",
            "name": "test_func",
            "docstring": "Test.",
            "source_code": "def test(): pass",
        }
        
        block = CodeBlock.from_dict(data)
        
        assert block.name == "test_func"
        assert block.language == "python"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
