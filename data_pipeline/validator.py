"""Data Validator - Comprehensive validation and quality scoring pipeline.

This module provides validation, quality scoring, and filtering for training data,
ensuring high-quality examples for LLM fine-tuning.

Key Features:
    - Code block validation (syntax, docstring, length)
    - RAFT example validation (oracle relevance, distractor irrelevance)
    - Multi-dimensional quality scoring
    - Duplicate detection
    - Quality report generation

Example:
    >>> from data_pipeline.validator import DataValidator, ValidationConfig
    >>> validator = DataValidator(ValidationConfig())
    >>> result = validator.validate_code_block(block)
    >>> metrics = validator.score_data_quality(dataset)
"""

from __future__ import annotations

import hashlib
import logging
import re
import statistics
from collections import Counter
from datetime import datetime
from typing import Any, Optional

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
    RAFTDataset,
    RAFTExample,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Text Similarity Utilities
# =============================================================================

def compute_text_hash(text: str) -> str:
    """Compute hash of text for duplicate detection."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between texts."""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def compute_relevance(query: str, document: str) -> float:
    """Compute relevance score between query and document.
    
    Uses keyword overlap and structure matching.
    """
    # Extract key terms from query
    query_terms = set(re.findall(r'\w+', query.lower()))
    doc_terms = set(re.findall(r'\w+', document.lower()))
    
    if not query_terms:
        return 0.5
    
    # Term overlap
    overlap = len(query_terms & doc_terms)
    term_score = overlap / len(query_terms) if query_terms else 0
    
    # Check if document contains function/class names from query
    name_pattern = r'\b([a-z_][a-z0-9_]*)\b'
    query_names = set(re.findall(name_pattern, query.lower()))
    doc_names = set(re.findall(name_pattern, document.lower()))
    
    name_overlap = len(query_names & doc_names)
    name_score = name_overlap / len(query_names) if query_names else 0
    
    # Weighted combination
    return 0.6 * term_score + 0.4 * name_score


# =============================================================================
# Code Block Validation
# =============================================================================

def validate_code_syntax(code: str, language: str) -> tuple[bool, Optional[str]]:
    """Validate code syntax.
    
    Args:
        code: Source code to validate.
        language: Programming language.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if language == "python":
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    # For other languages, perform basic structure checks
    if language in ("javascript", "typescript", "java", "go"):
        # Check for balanced braces
        open_count = code.count('{')
        close_count = code.count('}')
        if open_count != close_count:
            return False, f"Unbalanced braces: {open_count} open, {close_count} close"
        
        # Check for balanced parentheses
        open_paren = code.count('(')
        close_paren = code.count(')')
        if open_paren != close_paren:
            return False, f"Unbalanced parentheses: {open_paren} open, {close_paren} close"
    
    return True, None


def validate_code_block(
    block: CodeBlock,
    config: Optional[ValidationConfig] = None,
) -> tuple[bool, list[str]]:
    """Validate a code block.
    
    Checks:
    - Non-empty source code
    - Syntactically valid (for Python)
    - Has docstring (if required)
    - Reasonable token length
    
    Args:
        block: Code block to validate.
        config: Validation configuration.
        
    Returns:
        Tuple of (is_valid, error_messages).
    """
    config = config or ValidationConfig()
    errors: list[str] = []
    
    # Check non-empty
    if not block.source_code or not block.source_code.strip():
        errors.append("Empty source code")
        return False, errors
    
    # Check token length
    token_count = block.token_count
    if token_count < config.min_code_tokens:
        errors.append(f"Too few tokens: {token_count} < {config.min_code_tokens}")
    
    if token_count > config.max_code_tokens:
        errors.append(f"Too many tokens: {token_count} > {config.max_code_tokens}")
    
    # Check docstring
    if config.require_docstring and not block.has_docstring:
        errors.append("Missing docstring")
    
    # Check syntax
    if config.check_syntax:
        is_valid_syntax, syntax_error = validate_code_syntax(
            block.source_code, block.language
        )
        if not is_valid_syntax:
            errors.append(f"Invalid syntax: {syntax_error}")
    
    # Check name
    if not block.name or block.name.strip() == "":
        errors.append("Missing block name")
    
    return len(errors) == 0, errors


# =============================================================================
# RAFT Example Validation
# =============================================================================

def validate_raft_example(
    example: RAFTExample | dict[str, Any],
    config: Optional[ValidationConfig] = None,
) -> tuple[bool, list[str]]:
    """Validate a RAFT training example.
    
    Checks:
    - Oracle documents are relevant to query
    - Distractor documents are sufficiently irrelevant
    - Reasoning cites oracle documents
    - Answer is coherent
    
    Args:
        example: RAFT example to validate.
        config: Validation configuration.
        
    Returns:
        Tuple of (is_valid, error_messages).
    """
    config = config or ValidationConfig()
    errors: list[str] = []
    
    # Handle dict or RAFTExample
    if isinstance(example, dict):
        question = example.get("question", "")
        oracle_docs = example.get("oracle_documents", [])
        distractor_docs = example.get("distractor_documents", [])
        reasoning = example.get("reasoning", "")
        answer = example.get("final_answer", "")
    else:
        question = example.question
        oracle_docs = example.oracle_documents
        distractor_docs = example.distractor_documents
        reasoning = example.reasoning
        answer = example.final_answer
    
    # Check has question
    if not question or len(question.strip()) < 10:
        errors.append("Question too short or missing")
    
    # Check has oracle documents
    if not oracle_docs:
        errors.append("No oracle documents")
    
    # Check oracle relevance
    for i, doc in enumerate(oracle_docs):
        doc_code = doc.get("source_code", "") if isinstance(doc, dict) else getattr(doc, "source_code", "")
        relevance = compute_relevance(question, doc_code)
        
        if relevance < config.min_oracle_relevance * 0.5:  # Allow some slack
            errors.append(f"Oracle doc {i+1} has low relevance: {relevance:.2f}")
    
    # Check distractor irrelevance
    for i, doc in enumerate(distractor_docs):
        doc_code = doc.get("source_code", "") if isinstance(doc, dict) else getattr(doc, "source_code", "")
        relevance = compute_relevance(question, doc_code)
        
        if relevance > config.max_distractor_relevance * 2:  # Allow some slack
            errors.append(f"Distractor {i+1} too relevant: {relevance:.2f}")
    
    # Check reasoning cites documents
    if reasoning:
        # Check if reasoning mentions any document names
        has_citation = False
        for doc in oracle_docs:
            doc_name = doc.get("name", "") if isinstance(doc, dict) else getattr(doc, "name", "")
            if doc_name and doc_name.lower() in reasoning.lower():
                has_citation = True
                break
        
        # Also check for generic citation patterns
        if not has_citation:
            citation_patterns = [
                r"looking at",
                r"examining",
                r"from the code",
                r"in the \w+ function",
                r"the \w+ shows",
            ]
            for pattern in citation_patterns:
                if re.search(pattern, reasoning.lower()):
                    has_citation = True
                    break
        
        if not has_citation:
            errors.append("Reasoning lacks document citations")
    else:
        errors.append("Missing reasoning")
    
    # Check answer
    if not answer or len(answer.strip()) < 5:
        errors.append("Answer too short or missing")
    
    return len(errors) == 0, errors


# =============================================================================
# Quality Scoring
# =============================================================================

def score_example_quality(
    example: RAFTExample | dict[str, Any],
) -> QualityScore:
    """Compute quality score for a single example.
    
    Args:
        example: RAFT example to score.
        
    Returns:
        Multi-dimensional quality score.
    """
    # Handle dict or RAFTExample
    if isinstance(example, dict):
        question = example.get("question", "")
        oracle_docs = example.get("oracle_documents", [])
        distractor_docs = example.get("distractor_documents", [])
        reasoning = example.get("reasoning", "")
        answer = example.get("final_answer", "")
        difficulty = example.get("difficulty", "medium")
        question_type = example.get("question_type", "purpose")
    else:
        question = example.question
        oracle_docs = example.oracle_documents
        distractor_docs = example.distractor_documents
        reasoning = example.reasoning
        answer = example.final_answer
        difficulty = example.difficulty.value if hasattr(example.difficulty, 'value') else str(example.difficulty)
        question_type = example.question_type.value if hasattr(example.question_type, 'value') else str(example.question_type)
    
    # Relevance score (oracle docs should be relevant)
    relevance_scores = []
    for doc in oracle_docs:
        doc_code = doc.get("source_code", "") if isinstance(doc, dict) else getattr(doc, "source_code", "")
        rel = compute_relevance(question, doc_code)
        relevance_scores.append(rel)
    
    relevance = statistics.mean(relevance_scores) if relevance_scores else 0.5
    
    # Diversity score (based on question type variety in dataset context)
    # For single example, use difficulty as proxy
    diversity_map = {"easy": 0.6, "medium": 0.8, "hard": 0.9}
    diversity = diversity_map.get(difficulty, 0.7)
    
    # Clarity score (answer length and structure)
    clarity = 0.0
    if answer:
        # Good answer is 20-200 chars
        answer_len = len(answer)
        if 20 <= answer_len <= 200:
            clarity = 0.9
        elif 10 <= answer_len <= 500:
            clarity = 0.7
        elif answer_len > 5:
            clarity = 0.5
    
    # Grounding score (reasoning cites sources)
    grounding = 0.0
    if reasoning:
        # Check citation presence
        oracle_names = []
        for doc in oracle_docs:
            name = doc.get("name", "") if isinstance(doc, dict) else getattr(doc, "name", "")
            if name:
                oracle_names.append(name.lower())
        
        citations_found = sum(1 for name in oracle_names if name in reasoning.lower())
        if citations_found > 0:
            grounding = min(0.5 + 0.25 * citations_found, 1.0)
        
        # Check reasoning structure
        if "[REASONING]" in reasoning or "examining" in reasoning.lower() or "looking at" in reasoning.lower():
            grounding += 0.2
        
        grounding = min(grounding, 1.0)
    
    return QualityScore.compute(
        relevance=relevance,
        diversity=diversity,
        clarity=clarity,
        grounding=grounding,
    )


def score_data_quality(
    examples: list[RAFTExample | dict[str, Any]],
    config: Optional[ValidationConfig] = None,
) -> DataQualityMetrics:
    """Compute comprehensive quality metrics for a dataset.
    
    Args:
        examples: List of RAFT examples.
        config: Validation configuration.
        
    Returns:
        Complete quality metrics.
    """
    config = config or ValidationConfig()
    
    if not examples:
        return DataQualityMetrics()
    
    metrics = DataQualityMetrics(
        total_examples=len(examples),
        difficulty_distribution={d.value: 0 for d in Difficulty},
        question_type_distribution={q.value: 0 for q in QuestionType},
    )
    
    quality_scores: list[float] = []
    relevance_scores: list[float] = []
    irrelevance_scores: list[float] = []
    failure_reasons: Counter[str] = Counter()
    
    # Track duplicates
    seen_hashes: set[str] = set()
    duplicate_count = 0
    
    for example in examples:
        # Extract data
        if isinstance(example, dict):
            question = example.get("question", "")
            difficulty = example.get("difficulty", "medium")
            q_type = example.get("question_type", "purpose")
            oracle_docs = example.get("oracle_documents", [])
            distractor_docs = example.get("distractor_documents", [])
        else:
            question = example.question
            difficulty = example.difficulty.value if hasattr(example.difficulty, 'value') else str(example.difficulty)
            q_type = example.question_type.value if hasattr(example.question_type, 'value') else str(example.question_type)
            oracle_docs = example.oracle_documents
            distractor_docs = example.distractor_documents
        
        # Update distributions
        if difficulty in metrics.difficulty_distribution:
            metrics.difficulty_distribution[difficulty] += 1
        if q_type in metrics.question_type_distribution:
            metrics.question_type_distribution[q_type] += 1
        
        # Check duplicates
        q_hash = compute_text_hash(question)
        if q_hash in seen_hashes:
            duplicate_count += 1
        else:
            seen_hashes.add(q_hash)
        
        # Validate
        is_valid, errors = validate_raft_example(example, config)
        if is_valid:
            metrics.valid_examples += 1
        else:
            metrics.invalid_examples += 1
            for error in errors:
                failure_reasons[error] += 1
        
        # Score quality
        score = score_example_quality(example)
        quality_scores.append(score.overall)
        
        # Track relevance/irrelevance
        relevance_scores.append(score.relevance)
        
        # Compute distractor irrelevance
        for doc in distractor_docs:
            doc_code = doc.get("source_code", "") if isinstance(doc, dict) else getattr(doc, "source_code", "")
            rel = compute_relevance(question, doc_code)
            irrelevance_scores.append(1.0 - rel)  # Convert to irrelevance
    
    # Compute aggregate metrics
    metrics.quality_score_mean = statistics.mean(quality_scores)
    metrics.quality_score_std = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
    metrics.quality_score_min = min(quality_scores)
    metrics.quality_score_max = max(quality_scores)
    
    metrics.duplicate_count = duplicate_count
    metrics.duplicate_rate = duplicate_count / len(examples) if examples else 0
    
    metrics.avg_oracle_relevance = statistics.mean(relevance_scores) if relevance_scores else 0
    metrics.avg_distractor_irrelevance = statistics.mean(irrelevance_scores) if irrelevance_scores else 0
    
    # Top failure modes (up to 5)
    metrics.failure_modes = [f"{reason}: {count}" for reason, count in failure_reasons.most_common(5)]
    
    # By-dimension scores
    metrics.by_dimension = {
        "relevance": metrics.avg_oracle_relevance,
        "diversity": len(set(metrics.difficulty_distribution.keys())) / 3,
        "validity": metrics.valid_examples / max(metrics.total_examples, 1),
    }
    
    return metrics


# =============================================================================
# Filtering
# =============================================================================

def filter_by_quality(
    examples: list[RAFTExample | dict[str, Any]],
    config: Optional[ValidationConfig] = None,
) -> tuple[list[RAFTExample | dict[str, Any]], FilteringStats]:
    """Filter examples by quality thresholds.
    
    Args:
        examples: List of examples to filter.
        config: Filtering configuration.
        
    Returns:
        Tuple of (filtered_examples, filtering_stats).
    """
    config = config or ValidationConfig()
    
    filtered: list[RAFTExample | dict[str, Any]] = []
    removal_reasons: Counter[str] = Counter()
    
    for example in examples:
        # Validate
        is_valid, errors = validate_raft_example(example, config)
        
        if not is_valid:
            for error in errors:
                removal_reasons[error] += 1
            continue
        
        # Score quality
        score = score_example_quality(example)
        
        if score.overall < config.min_quality_score:
            removal_reasons["low_quality_score"] += 1
            continue
        
        filtered.append(example)
    
    stats = FilteringStats(
        before_count=len(examples),
        after_count=len(filtered),
        removed_count=len(examples) - len(filtered),
        removal_rate=(len(examples) - len(filtered)) / max(len(examples), 1),
        removal_reasons=dict(removal_reasons),
    )
    
    return filtered, stats


# =============================================================================
# Quality Report Generation
# =============================================================================

def generate_quality_report(
    examples: list[RAFTExample | dict[str, Any]],
    config: Optional[ValidationConfig] = None,
    include_worst_examples: int = 5,
) -> str:
    """Generate a Markdown quality report.
    
    Args:
        examples: Dataset to analyze.
        config: Validation configuration.
        include_worst_examples: Number of worst examples to include.
        
    Returns:
        Markdown report string.
    """
    config = config or ValidationConfig()
    metrics = score_data_quality(examples, config)
    
    # Score all examples
    scored_examples: list[tuple[float, dict | RAFTExample]] = []
    for ex in examples:
        score = score_example_quality(ex)
        scored_examples.append((score.overall, ex))
    
    # Sort by score
    scored_examples.sort(key=lambda x: x[0])
    
    # Build report
    report_parts = [
        "# Data Quality Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Total Examples**: {metrics.total_examples}",
        "",
        "---",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Valid Examples | {metrics.valid_examples} ({metrics.valid_examples/max(metrics.total_examples,1):.1%}) |",
        f"| Invalid Examples | {metrics.invalid_examples} |",
        f"| Quality Score (Mean) | {metrics.quality_score_mean:.3f} |",
        f"| Quality Score (Std) | {metrics.quality_score_std:.3f} |",
        f"| Quality Score (Range) | {metrics.quality_score_min:.3f} - {metrics.quality_score_max:.3f} |",
        f"| Duplicate Rate | {metrics.duplicate_rate:.1%} |",
        f"| Avg Oracle Relevance | {metrics.avg_oracle_relevance:.3f} |",
        "",
        "---",
        "",
        "## Difficulty Distribution",
        "",
        "| Difficulty | Count | Percentage |",
        "|------------|-------|------------|",
    ]
    
    for diff, count in metrics.difficulty_distribution.items():
        pct = count / max(metrics.total_examples, 1) * 100
        report_parts.append(f"| {diff} | {count} | {pct:.1f}% |")
    
    report_parts.extend([
        "",
        "---",
        "",
        "## Question Type Distribution",
        "",
        "| Type | Count | Percentage |",
        "|------|-------|------------|",
    ])
    
    for qtype, count in sorted(metrics.question_type_distribution.items(), key=lambda x: -x[1]):
        pct = count / max(metrics.total_examples, 1) * 100
        report_parts.append(f"| {qtype} | {count} | {pct:.1f}% |")
    
    # Failure modes
    if metrics.failure_modes:
        report_parts.extend([
            "",
            "---",
            "",
            "## Failure Modes",
            "",
        ])
        for mode in metrics.failure_modes:
            report_parts.append(f"- {mode}")
    
    # Quality score histogram (ASCII)
    report_parts.extend([
        "",
        "---",
        "",
        "## Quality Score Distribution",
        "",
        "```",
    ])
    
    # Build histogram
    buckets = [0] * 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    for score, _ in scored_examples:
        bucket = min(int(score * 10), 9)
        buckets[bucket] += 1
    
    max_count = max(buckets) if buckets else 1
    for i, count in enumerate(buckets):
        bar_len = int(count / max_count * 30)
        bar = "█" * bar_len
        label = f"{i/10:.1f}-{(i+1)/10:.1f}"
        report_parts.append(f"{label} |{bar} ({count})")
    
    report_parts.append("```")
    
    # Worst examples
    if include_worst_examples > 0 and scored_examples:
        report_parts.extend([
            "",
            "---",
            "",
            f"## Lowest Quality Examples (Bottom {include_worst_examples})",
            "",
        ])
        
        for i, (score, ex) in enumerate(scored_examples[:include_worst_examples], 1):
            if isinstance(ex, dict):
                question = ex.get("question", "N/A")[:80]
                difficulty = ex.get("difficulty", "N/A")
            else:
                question = ex.question[:80]
                difficulty = ex.difficulty.value if hasattr(ex.difficulty, 'value') else str(ex.difficulty)
            
            report_parts.extend([
                f"### Example {i} (Score: {score:.3f})",
                f"- **Difficulty**: {difficulty}",
                f"- **Question**: {question}...",
                "",
            ])
    
    # Recommendations
    report_parts.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
    ])
    
    recommendations = []
    
    if metrics.quality_score_mean < 0.7:
        recommendations.append("- ⚠️ Average quality score is low. Consider improving question/answer generation.")
    
    if metrics.duplicate_rate > 0.05:
        recommendations.append(f"- ⚠️ High duplicate rate ({metrics.duplicate_rate:.1%}). Enable deduplication.")
    
    if metrics.invalid_examples > metrics.total_examples * 0.1:
        recommendations.append("- ⚠️ Many invalid examples. Review validation errors above.")
    
    if metrics.avg_oracle_relevance < 0.6:
        recommendations.append("- ⚠️ Low oracle relevance. Improve document retrieval.")
    
    if not recommendations:
        recommendations.append("- ✅ Dataset quality looks good!")
    
    report_parts.extend(recommendations)
    
    return "\n".join(report_parts)


# =============================================================================
# DataValidator Class
# =============================================================================

class DataValidator:
    """Comprehensive data validation and quality scoring.
    
    Example:
        >>> validator = DataValidator(ValidationConfig())
        >>> result = validator.validate_code_block(block)
        >>> metrics = validator.score_dataset(examples)
        >>> report = validator.generate_report(examples)
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        """Initialize validator."""
        self.config = config or ValidationConfig()
    
    def validate_code_block(self, block: CodeBlock) -> ValidationResult:
        """Validate a code block."""
        is_valid, errors = validate_code_block(block, self.config)
        result = ValidationResult(is_valid=is_valid, errors=errors)
        result.checks_passed = 5 - len(errors)  # Approximate
        result.checks_failed = len(errors)
        return result
    
    def validate_raft_example(self, example: RAFTExample | dict) -> ValidationResult:
        """Validate a RAFT example."""
        is_valid, errors = validate_raft_example(example, self.config)
        result = ValidationResult(is_valid=is_valid, errors=errors)
        result.checks_passed = 5 - len(errors)
        result.checks_failed = len(errors)
        return result
    
    def score_example(self, example: RAFTExample | dict) -> QualityScore:
        """Score a single example."""
        return score_example_quality(example)
    
    def score_dataset(self, examples: list) -> DataQualityMetrics:
        """Score a complete dataset."""
        return score_data_quality(examples, self.config)
    
    def filter_dataset(self, examples: list) -> tuple[list, FilteringStats]:
        """Filter dataset by quality."""
        return filter_by_quality(examples, self.config)
    
    def generate_report(self, examples: list, worst_n: int = 5) -> str:
        """Generate quality report."""
        return generate_quality_report(examples, self.config, worst_n)
    
    def validate_all(self, examples: list) -> tuple[list[ValidationResult], DataQualityMetrics]:
        """Validate all examples and compute metrics."""
        results = []
        for ex in examples:
            result = self.validate_raft_example(ex)
            results.append(result)
        
        metrics = self.score_dataset(examples)
        return results, metrics
