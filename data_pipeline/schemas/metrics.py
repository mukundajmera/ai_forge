"""Data Quality Metrics - Pydantic models for validation and quality scoring.

This module defines metrics and configuration for data validation,
quality assessment, and filtering of training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    """Status of validation check."""
    
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a single validation check.
    
    Attributes:
        is_valid: Whether the item passed validation.
        errors: List of error messages.
        warnings: List of warning messages.
        checks_passed: Number of checks passed.
        checks_failed: Number of checks failed.
        details: Additional validation details.
    """
    
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.checks_failed += 1
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_passed(self) -> None:
        """Record a passed check."""
        self.checks_passed += 1
    
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge with another validation result."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            checks_passed=self.checks_passed + other.checks_passed,
            checks_failed=self.checks_failed + other.checks_failed,
            details={**self.details, **other.details},
        )


class ValidationConfig(BaseModel):
    """Configuration for data validation.
    
    Attributes:
        min_code_tokens: Minimum tokens in code block.
        max_code_tokens: Maximum tokens in code block.
        require_docstring: Whether docstrings are required.
        min_quality_score: Minimum quality score to pass.
        min_oracle_relevance: Minimum oracle document relevance.
        max_distractor_relevance: Maximum distractor relevance.
        check_syntax: Whether to validate code syntax.
        detect_duplicates: Whether to detect duplicates.
        duplicate_threshold: Similarity threshold for duplicates.
    """
    
    min_code_tokens: int = Field(default=20, ge=1)
    max_code_tokens: int = Field(default=2048, ge=1)
    require_docstring: bool = Field(default=False)
    min_quality_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_oracle_relevance: float = Field(default=0.75, ge=0.0, le=1.0)
    max_distractor_relevance: float = Field(default=0.3, ge=0.0, le=1.0)
    check_syntax: bool = Field(default=True)
    detect_duplicates: bool = Field(default=True)
    duplicate_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


@dataclass
class DataQualityMetrics:
    """Comprehensive quality metrics for a dataset.
    
    Attributes:
        total_examples: Total number of examples.
        valid_examples: Number of valid examples.
        invalid_examples: Number of invalid examples.
        quality_score_mean: Mean quality score.
        quality_score_std: Standard deviation of quality scores.
        quality_score_min: Minimum quality score.
        quality_score_max: Maximum quality score.
        difficulty_distribution: Count by difficulty level.
        question_type_distribution: Count by question type.
        duplicate_count: Number of duplicate examples.
        duplicate_rate: Fraction of duplicates.
        avg_oracle_relevance: Average oracle document relevance.
        avg_distractor_irrelevance: Average distractor irrelevance.
        failure_modes: List of common failure reasons.
        by_dimension: Scores by quality dimension.
    """
    
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    quality_score_mean: float = 0.0
    quality_score_std: float = 0.0
    quality_score_min: float = 0.0
    quality_score_max: float = 0.0
    difficulty_distribution: dict[str, int] = field(default_factory=dict)
    question_type_distribution: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    duplicate_rate: float = 0.0
    avg_oracle_relevance: float = 0.0
    avg_distractor_irrelevance: float = 0.0
    failure_modes: list[str] = field(default_factory=list)
    by_dimension: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_examples": self.total_examples,
            "valid_examples": self.valid_examples,
            "invalid_examples": self.invalid_examples,
            "valid_rate": self.valid_examples / max(self.total_examples, 1),
            "quality_score_mean": round(self.quality_score_mean, 3),
            "quality_score_std": round(self.quality_score_std, 3),
            "quality_score_min": round(self.quality_score_min, 3),
            "quality_score_max": round(self.quality_score_max, 3),
            "difficulty_distribution": self.difficulty_distribution,
            "question_type_distribution": self.question_type_distribution,
            "duplicate_count": self.duplicate_count,
            "duplicate_rate": round(self.duplicate_rate, 3),
            "avg_oracle_relevance": round(self.avg_oracle_relevance, 3),
            "avg_distractor_irrelevance": round(self.avg_distractor_irrelevance, 3),
            "failure_modes": self.failure_modes,
            "by_dimension": {k: round(v, 3) for k, v in self.by_dimension.items()},
        }


@dataclass 
class QualityScore:
    """Multi-dimensional quality score for a single example.
    
    Attributes:
        overall: Overall quality score (0-1).
        relevance: Oracle document relevance.
        diversity: Question diversity contribution.
        clarity: Answer clarity and coherence.
        grounding: Reasoning grounded in context.
        breakdown: Detailed score breakdown.
    """
    
    overall: float
    relevance: float = 0.0
    diversity: float = 0.0
    clarity: float = 0.0
    grounding: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def compute(
        cls,
        relevance: float,
        diversity: float,
        clarity: float,
        grounding: float,
    ) -> "QualityScore":
        """Compute quality score from dimensions."""
        # Weighted average
        weights = {
            "relevance": 0.35,
            "diversity": 0.15,
            "clarity": 0.25,
            "grounding": 0.25,
        }
        
        overall = (
            relevance * weights["relevance"] +
            diversity * weights["diversity"] +
            clarity * weights["clarity"] +
            grounding * weights["grounding"]
        )
        
        return cls(
            overall=min(overall, 1.0),
            relevance=relevance,
            diversity=diversity,
            clarity=clarity,
            grounding=grounding,
            breakdown=weights,
        )


class FilteringStats(BaseModel):
    """Statistics from filtering operation.
    
    Attributes:
        before_count: Examples before filtering.
        after_count: Examples after filtering.
        removed_count: Examples removed.
        removal_rate: Fraction removed.
        removal_reasons: Count by removal reason.
    """
    
    before_count: int = 0
    after_count: int = 0
    removed_count: int = 0
    removal_rate: float = 0.0
    removal_reasons: dict[str, int] = Field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Filtered: {self.before_count} → {self.after_count} "
            f"({self.removal_rate:.1%} removed)"
        )
