"""Data Validator - Quality checks for training data.

This module provides comprehensive validation for training data,
including deduplication, token length validation, format verification,
and quality scoring.

Example:
    >>> validator = DataValidator(data)
    >>> results = validator.validate_all()
    >>> if results.is_valid:
    ...     cleaned_data = validator.get_cleaned_data()
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from data validation.
    
    Attributes:
        is_valid: Whether the data passed all critical checks.
        total_samples: Total number of samples validated.
        valid_samples: Number of samples that passed validation.
        errors: List of critical errors.
        warnings: List of non-critical warnings.
        stats: Various statistics about the data.
    """
    
    is_valid: bool
    total_samples: int
    valid_samples: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate a summary string."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (
            f"{status}\n"
            f"Total samples: {self.total_samples}\n"
            f"Valid samples: {self.valid_samples} "
            f"({self.valid_samples/self.total_samples*100:.1f}%)\n"
            f"Errors: {len(self.errors)}\n"
            f"Warnings: {len(self.warnings)}"
        )


@dataclass
class ValidatorConfig:
    """Configuration for DataValidator.
    
    Attributes:
        min_instruction_length: Minimum chars for instruction field.
        max_instruction_length: Maximum chars for instruction field.
        min_output_length: Minimum chars for output field.
        max_output_length: Maximum chars for output field.
        max_token_length: Maximum total tokens per sample.
        min_samples: Minimum number of samples required.
        required_fields: Fields that must be present.
        allow_empty_input: Whether input field can be empty.
        min_quality_score: Minimum quality score threshold.
    """
    
    min_instruction_length: int = 10
    max_instruction_length: int = 1000
    min_output_length: int = 20
    max_output_length: int = 8000
    max_token_length: int = 2048
    min_samples: int = 100
    required_fields: list[str] = field(
        default_factory=lambda: ["instruction", "output"]
    )
    allow_empty_input: bool = True
    min_quality_score: float = 0.3


class DataValidator:
    """Validates and cleans training data for fine-tuning.
    
    This class performs comprehensive validation including:
    - Format and field checks
    - Length validation
    - Deduplication
    - Quality scoring
    - Character encoding validation
    
    Attributes:
        data: List of training samples.
        config: Validation configuration.
        
    Example:
        >>> data = [{"instruction": "...", "output": "..."}, ...]
        >>> validator = DataValidator(data)
        >>> results = validator.validate_all()
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        data: list[dict[str, Any]],
        config: Optional[ValidatorConfig] = None,
    ) -> None:
        """Initialize DataValidator.
        
        Args:
            data: List of training samples.
            config: Optional validation configuration.
        """
        self.data = data
        self.config = config or ValidatorConfig()
        self._valid_indices: set[int] = set(range(len(data)))
        self._duplicate_hashes: set[str] = set()
        
        logger.info(f"Initialized DataValidator with {len(data)} samples")
    
    def _compute_hash(self, sample: dict[str, Any]) -> str:
        """Compute hash for deduplication.
        
        Args:
            sample: Training sample.
            
        Returns:
            SHA256 hash of the sample content.
        """
        content = f"{sample.get('instruction', '')}{sample.get('output', '')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token).
        
        Args:
            text: Text to estimate tokens for.
            
        Returns:
            Estimated token count.
        """
        return len(text) // 4
    
    def check_required_fields(self) -> list[str]:
        """Check for required fields.
        
        Returns:
            List of error messages for missing fields.
        """
        errors: list[str] = []
        
        for idx, sample in enumerate(self.data):
            for field in self.config.required_fields:
                if field not in sample or sample[field] is None:
                    errors.append(f"Sample {idx}: Missing required field '{field}'")
                    self._valid_indices.discard(idx)
        
        return errors
    
    def check_lengths(self) -> tuple[list[str], list[str]]:
        """Check field lengths.
        
        Returns:
            Tuple of (errors, warnings).
        """
        errors: list[str] = []
        warnings: list[str] = []
        
        for idx, sample in enumerate(self.data):
            if idx not in self._valid_indices:
                continue
            
            instruction = sample.get("instruction", "")
            output = sample.get("output", "")
            input_text = sample.get("input", "")
            
            # Instruction length
            if len(instruction) < self.config.min_instruction_length:
                errors.append(f"Sample {idx}: Instruction too short ({len(instruction)} chars)")
                self._valid_indices.discard(idx)
            elif len(instruction) > self.config.max_instruction_length:
                warnings.append(f"Sample {idx}: Instruction very long ({len(instruction)} chars)")
            
            # Output length
            if len(output) < self.config.min_output_length:
                errors.append(f"Sample {idx}: Output too short ({len(output)} chars)")
                self._valid_indices.discard(idx)
            elif len(output) > self.config.max_output_length:
                warnings.append(f"Sample {idx}: Output very long ({len(output)} chars)")
            
            # Total tokens
            total_tokens = self._estimate_tokens(instruction + input_text + output)
            if total_tokens > self.config.max_token_length:
                warnings.append(f"Sample {idx}: May exceed token limit ({total_tokens} tokens)")
        
        return errors, warnings
    
    def deduplicate(self) -> int:
        """Remove duplicate samples.
        
        Returns:
            Number of duplicates removed.
        """
        seen_hashes: set[str] = set()
        duplicates_removed = 0
        
        for idx, sample in enumerate(self.data):
            if idx not in self._valid_indices:
                continue
            
            hash_val = self._compute_hash(sample)
            if hash_val in seen_hashes:
                self._valid_indices.discard(idx)
                self._duplicate_hashes.add(hash_val)
                duplicates_removed += 1
            else:
                seen_hashes.add(hash_val)
        
        logger.info(f"Removed {duplicates_removed} duplicates")
        return duplicates_removed
    
    def compute_quality_score(self, sample: dict[str, Any]) -> float:
        """Compute quality score for a sample.
        
        Scoring factors:
        - Output length (longer is generally better)
        - Instruction clarity (question words, etc.)
        - Code formatting (backticks, etc.)
        - No repetition
        
        Args:
            sample: Training sample to score.
            
        Returns:
            Quality score between 0 and 1.
        """
        score = 0.0
        
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        
        # Length factor (0.0 - 0.3)
        output_len = len(output)
        if output_len >= 200:
            score += 0.3
        elif output_len >= 100:
            score += 0.2
        elif output_len >= 50:
            score += 0.1
        
        # Instruction quality (0.0 - 0.3)
        question_words = ["what", "how", "why", "explain", "describe", "when", "where"]
        if any(word in instruction.lower() for word in question_words):
            score += 0.15
        if instruction.endswith("?"):
            score += 0.1
        if len(instruction.split()) >= 5:
            score += 0.05
        
        # Output quality (0.0 - 0.4)
        if "```" in output:  # Has code blocks
            score += 0.1
        if output.count(".") >= 2:  # Multiple sentences
            score += 0.1
        if not self._has_repetition(output):
            score += 0.2
        
        return min(score, 1.0)
    
    def _has_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """Check for excessive repetition.
        
        Args:
            text: Text to check.
            threshold: Repetition threshold.
            
        Returns:
            True if excessive repetition detected.
        """
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        
        return most_common_count / len(words) > threshold
    
    def filter_by_quality(self) -> int:
        """Filter samples by quality score.
        
        Returns:
            Number of samples filtered out.
        """
        filtered = 0
        
        for idx, sample in enumerate(self.data):
            if idx not in self._valid_indices:
                continue
            
            score = self.compute_quality_score(sample)
            if score < self.config.min_quality_score:
                self._valid_indices.discard(idx)
                filtered += 1
        
        logger.info(f"Filtered {filtered} low-quality samples")
        return filtered
    
    def validate_all(self) -> ValidationResult:
        """Run all validation checks.
        
        Returns:
            ValidationResult with all findings.
        """
        errors: list[str] = []
        warnings: list[str] = []
        
        # Run checks
        field_errors = self.check_required_fields()
        errors.extend(field_errors)
        
        length_errors, length_warnings = self.check_lengths()
        errors.extend(length_errors)
        warnings.extend(length_warnings)
        
        duplicates = self.deduplicate()
        if duplicates > 0:
            warnings.append(f"Removed {duplicates} duplicate samples")
        
        filtered = self.filter_by_quality()
        if filtered > 0:
            warnings.append(f"Filtered {filtered} low-quality samples")
        
        # Check minimum samples
        valid_count = len(self._valid_indices)
        if valid_count < self.config.min_samples:
            errors.append(
                f"Insufficient samples: {valid_count} < {self.config.min_samples} required"
            )
        
        # Compute stats
        stats = {
            "duplicates_removed": duplicates,
            "quality_filtered": filtered,
            "average_output_length": self._compute_average_output_length(),
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0 and valid_count >= self.config.min_samples,
            total_samples=len(self.data),
            valid_samples=valid_count,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
    
    def _compute_average_output_length(self) -> float:
        """Compute average output length for valid samples."""
        lengths = [
            len(self.data[idx].get("output", ""))
            for idx in self._valid_indices
        ]
        return sum(lengths) / len(lengths) if lengths else 0.0
    
    def get_cleaned_data(self) -> list[dict[str, Any]]:
        """Get validated and cleaned data.
        
        Returns:
            List of valid samples only.
        """
        return [self.data[idx] for idx in sorted(self._valid_indices)]
    
    def save_cleaned_data(self, output_path: str) -> None:
        """Save cleaned data to file.
        
        Args:
            output_path: Path to save JSON file.
        """
        import json
        
        cleaned = self.get_cleaned_data()
        with open(output_path, "w") as f:
            json.dump(cleaned, f, indent=2)
        
        logger.info(f"Saved {len(cleaned)} samples to {output_path}")
