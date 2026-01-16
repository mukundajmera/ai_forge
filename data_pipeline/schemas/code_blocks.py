"""Code Block Schemas - Pydantic models for semantic code extraction.

This module defines the data structures used throughout the mining pipeline,
including CodeBlock for individual code units and configuration classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class BlockType(str, Enum):
    """Types of code blocks that can be extracted."""
    
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    INTERFACE = "interface"
    STRUCT = "struct"


class Language(str, Enum):
    """Supported programming languages."""
    
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    CPP = "cpp"
    C = "c"


class QualityLevel(str, Enum):
    """Quality classification for code blocks."""
    
    HIGH = "high"      # Score >= 0.7
    MEDIUM = "medium"  # Score >= 0.4
    LOW = "low"        # Score < 0.4
    REJECTED = "rejected"  # Fails quality filters


@dataclass
class CodeBlock:
    """Represents a semantic unit of extracted code.
    
    This is the primary output of the code mining process. Each CodeBlock
    represents a complete syntactic unit (function, class, module) with
    all associated metadata for training data generation.
    
    Attributes:
        path: Relative path to the source file.
        language: Programming language identifier.
        block_type: Type of code unit (function, class, etc.).
        name: Name of the code unit.
        docstring: Extracted docstring, if present.
        source_code: Complete source code of the block.
        dependencies: List of referenced functions/classes/modules.
        metadata: Additional information (line numbers, complexity, etc.).
        
    Example:
        >>> block = CodeBlock(
        ...     path="src/utils.py",
        ...     language="python",
        ...     block_type="function",
        ...     name="calculate_sum",
        ...     docstring="Calculate sum of two numbers.",
        ...     source_code="def calculate_sum(a, b):\\n    return a + b",
        ...     dependencies=["math"],
        ...     metadata={"start_line": 10, "end_line": 12}
        ... )
    """
    
    path: str
    language: str
    block_type: str
    name: str
    docstring: Optional[str]
    source_code: str
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def token_count(self) -> int:
        """Estimate token count (roughly 4 characters per token)."""
        return len(self.source_code) // 4
    
    @property
    def line_count(self) -> int:
        """Number of lines in source code."""
        return self.source_code.count('\n') + 1
    
    @property
    def has_docstring(self) -> bool:
        """Check if block has a non-empty docstring."""
        return bool(self.docstring and self.docstring.strip())
    
    @property
    def start_line(self) -> int:
        """Get starting line number from metadata."""
        return self.metadata.get("start_line", 1)
    
    @property
    def end_line(self) -> int:
        """Get ending line number from metadata."""
        return self.metadata.get("end_line", self.line_count)
    
    def quality_score(self) -> float:
        """Calculate quality score (0-1) based on multiple factors.
        
        Scoring factors:
        - Has docstring: +0.3
        - Docstring length (scaled): +0.2
        - Code length in acceptable range: +0.2
        - Has meaningful name: +0.15
        - Complexity (not too simple/complex): +0.15
        
        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = 0.0
        
        # Has docstring
        if self.has_docstring:
            score += 0.3
            # Docstring quality (length-based)
            doc_len = len(self.docstring or "")
            if doc_len > 50:
                score += 0.2
            elif doc_len > 20:
                score += 0.1
        
        # Code length (prefer 20-500 tokens)
        tokens = self.token_count
        if 20 <= tokens <= 500:
            score += 0.2
        elif 10 <= tokens <= 1000:
            score += 0.1
        
        # Meaningful name (not single char, not just underscores)
        if len(self.name) > 2 and not self.name.startswith('_'):
            score += 0.15
        elif len(self.name) > 1:
            score += 0.05
        
        # Not trivially simple (at least some complexity)
        if self.line_count >= 3:
            score += 0.15
        elif self.line_count >= 2:
            score += 0.05
        
        return min(score, 1.0)
    
    def quality_level(self) -> QualityLevel:
        """Get quality classification based on score."""
        score = self.quality_score()
        if score >= 0.7:
            return QualityLevel.HIGH
        elif score >= 0.4:
            return QualityLevel.MEDIUM
        elif score >= 0.2:
            return QualityLevel.LOW
        return QualityLevel.REJECTED
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": self.path,
            "language": self.language,
            "block_type": self.block_type,
            "name": self.name,
            "docstring": self.docstring,
            "source_code": self.source_code,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "quality_score": self.quality_score(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeBlock":
        """Create CodeBlock from dictionary."""
        return cls(
            path=data["path"],
            language=data["language"],
            block_type=data["block_type"],
            name=data["name"],
            docstring=data.get("docstring"),
            source_code=data["source_code"],
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
        )


class MinerConfig(BaseModel):
    """Configuration for the code miner.
    
    Attributes:
        languages: Languages to parse.
        min_tokens: Minimum tokens per block.
        max_tokens: Maximum tokens per block.
        require_docstring: Whether to require docstrings.
        include_tests: Whether to include test files.
        include_private: Whether to include private/underscore functions.
        exclude_patterns: Glob patterns to exclude.
        min_quality_score: Minimum quality score to accept.
    """
    
    languages: list[str] = Field(
        default=["python", "javascript", "go", "java"],
        description="Languages to parse"
    )
    min_tokens: int = Field(default=20, ge=1, description="Minimum tokens per block")
    max_tokens: int = Field(default=2048, ge=1, description="Maximum tokens per block")
    require_docstring: bool = Field(default=False, description="Require docstrings")
    include_tests: bool = Field(default=True, description="Include test files")
    include_private: bool = Field(default=False, description="Include private functions")
    exclude_patterns: list[str] = Field(
        default=[
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.git/**",
            "**/dist/**",
            "**/build/**",
            "**/*.min.js",
        ],
        description="Glob patterns to exclude"
    )
    min_quality_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Minimum quality score"
    )
    
    @field_validator('languages', mode='before')
    @classmethod
    def validate_languages(cls, v: list[str]) -> list[str]:
        """Validate language names."""
        valid = {"python", "javascript", "typescript", "go", "java", "rust", "cpp", "c"}
        for lang in v:
            if lang.lower() not in valid:
                raise ValueError(f"Unsupported language: {lang}")
        return [lang.lower() for lang in v]


class LanguageConfig(BaseModel):
    """Configuration for a specific programming language parser.
    
    Attributes:
        name: Language identifier.
        extensions: File extensions for this language.
        function_node_types: AST node types for functions.
        class_node_types: AST node types for classes.
        docstring_node_types: AST node types for docstrings.
    """
    
    name: str
    extensions: list[str]
    function_node_types: list[str]
    class_node_types: list[str]
    docstring_node_types: list[str]
    comment_node_types: list[str] = Field(default=["comment"])


# Pre-defined language configurations
LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        name="python",
        extensions=[".py", ".pyi"],
        function_node_types=["function_definition", "async_function_definition"],
        class_node_types=["class_definition"],
        docstring_node_types=["expression_statement"],  # Contains string node
        comment_node_types=["comment"],
    ),
    "javascript": LanguageConfig(
        name="javascript",
        extensions=[".js", ".jsx", ".mjs"],
        function_node_types=["function_declaration", "arrow_function", "method_definition"],
        class_node_types=["class_declaration"],
        docstring_node_types=["comment"],  # JSDoc comments
        comment_node_types=["comment"],
    ),
    "typescript": LanguageConfig(
        name="typescript",
        extensions=[".ts", ".tsx"],
        function_node_types=["function_declaration", "arrow_function", "method_definition"],
        class_node_types=["class_declaration", "interface_declaration"],
        docstring_node_types=["comment"],
        comment_node_types=["comment"],
    ),
    "go": LanguageConfig(
        name="go",
        extensions=[".go"],
        function_node_types=["function_declaration", "method_declaration"],
        class_node_types=["type_declaration"],  # struct types
        docstring_node_types=["comment"],
        comment_node_types=["comment"],
    ),
    "java": LanguageConfig(
        name="java",
        extensions=[".java"],
        function_node_types=["method_declaration", "constructor_declaration"],
        class_node_types=["class_declaration", "interface_declaration"],
        docstring_node_types=["block_comment"],  # Javadoc
        comment_node_types=["line_comment", "block_comment"],
    ),
    "rust": LanguageConfig(
        name="rust",
        extensions=[".rs"],
        function_node_types=["function_item"],
        class_node_types=["struct_item", "impl_item", "trait_item"],
        docstring_node_types=["line_comment", "block_comment"],  # /// and /** */
        comment_node_types=["line_comment", "block_comment"],
    ),
}


class ExtractionResult(BaseModel):
    """Result of code extraction from a repository.
    
    Attributes:
        blocks: Extracted code blocks.
        total_files: Total files processed.
        total_lines: Total lines of code.
        errors: List of errors encountered.
        statistics: Extraction statistics.
    """
    
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    total_files: int = 0
    total_lines: int = 0
    errors: list[str] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    
    def add_block(self, block: CodeBlock) -> None:
        """Add a code block to results."""
        self.blocks.append(block.to_dict())
    
    def summary(self) -> str:
        """Generate summary of extraction."""
        return (
            f"Extracted {len(self.blocks)} code blocks from {self.total_files} files "
            f"({self.total_lines} lines). Errors: {len(self.errors)}"
        )
