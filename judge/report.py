"""Evaluation Report - Comprehensive model evaluation reporting.

This module provides the EvaluationReport class for generating
detailed evaluation reports with comparisons and visualizations.

Example:
    >>> report = EvaluationReport(model_name="my-model", base_model="llama-3.2")
    >>> report.add_metrics(perplexity=5.2, codebleu_score=0.75)
    >>> report.to_markdown("./report.md")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for fine-tuned models.
    
    Attributes:
        model_name: Name of the evaluated model.
        base_model: Base model used for fine-tuning.
        training_date: Date of training.
        num_training_examples: Number of training examples.
        num_eval_examples: Number of evaluation examples.
        perplexity: Model perplexity.
        codebleu_score: CodeBLEU score (0-1).
        humaneval_pass_rate: HumanEval pass@1 rate.
        reconstruction_score: Code reconstruction score.
        hallucination_rate: Hallucination rate (0-1).
        improvement_over_base: Percentage improvements vs base.
        loss_curve: Optional PNG bytes for loss curve.
        metric_distribution: Optional PNG bytes for metrics.
    """
    
    model_name: str
    base_model: str
    training_date: str = field(default_factory=lambda: datetime.now().isoformat())
    num_training_examples: int = 0
    num_eval_examples: int = 0
    
    # Core metrics
    perplexity: Optional[float] = None
    codebleu_score: Optional[float] = None
    humaneval_pass_rate: Optional[float] = None
    reconstruction_score: Optional[float] = None
    hallucination_rate: Optional[float] = None
    
    # Comparisons
    improvement_over_base: dict[str, float] = field(default_factory=dict)
    
    # Artifacts
    loss_curve: Optional[bytes] = None
    metric_distribution: Optional[bytes] = None
    
    # Additional metrics
    additional_metrics: dict[str, float] = field(default_factory=dict)
    
    def add_metrics(self, **kwargs: float) -> None:
        """Add metrics to the report.
        
        Args:
            **kwargs: Metric name-value pairs.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_metrics[key] = value
    
    def compute_improvement(
        self,
        base_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Compute improvement percentages over base model.
        
        Args:
            base_metrics: Base model metrics to compare against.
            
        Returns:
            Dictionary of improvement percentages.
        """
        improvements = {}
        
        # Perplexity (lower is better)
        if self.perplexity and "perplexity" in base_metrics:
            base_ppl = base_metrics["perplexity"]
            improvement = ((base_ppl - self.perplexity) / base_ppl) * 100
            improvements["perplexity"] = improvement
        
        # CodeBLEU (higher is better)
        if self.codebleu_score and "codebleu_score" in base_metrics:
            base_score = base_metrics["codebleu_score"]
            if base_score > 0:
                improvement = ((self.codebleu_score - base_score) / base_score) * 100
                improvements["codebleu_score"] = improvement
        
        # HumanEval (higher is better)
        if self.humaneval_pass_rate and "humaneval_pass_rate" in base_metrics:
            base_rate = base_metrics["humaneval_pass_rate"]
            if base_rate > 0:
                improvement = ((self.humaneval_pass_rate - base_rate) / base_rate) * 100
                improvements["humaneval_pass_rate"] = improvement
        
        # Hallucination (lower is better)
        if self.hallucination_rate is not None and "hallucination_rate" in base_metrics:
            base_rate = base_metrics["hallucination_rate"]
            if base_rate > 0:
                improvement = ((base_rate - self.hallucination_rate) / base_rate) * 100
                improvements["hallucination_rate"] = improvement
        
        self.improvement_over_base = improvements
        return improvements
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "model_name": self.model_name,
            "base_model": self.base_model,
            "training_date": self.training_date,
            "num_training_examples": self.num_training_examples,
            "num_eval_examples": self.num_eval_examples,
            "metrics": {
                "perplexity": self.perplexity,
                "codebleu_score": self.codebleu_score,
                "humaneval_pass_rate": self.humaneval_pass_rate,
                "reconstruction_score": self.reconstruction_score,
                "hallucination_rate": self.hallucination_rate,
                **self.additional_metrics,
            },
            "improvement_over_base": self.improvement_over_base,
        }
    
    def to_json(self, path: str | Path) -> None:
        """Save report as JSON.
        
        Args:
            path: Output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Report saved to {path}")
    
    def to_markdown(self, path: Optional[str | Path] = None) -> str:
        """Generate markdown report.
        
        Args:
            path: Optional path to save markdown.
            
        Returns:
            Markdown content.
        """
        lines = [
            f"# Evaluation Report: {self.model_name}",
            "",
            f"**Base Model:** {self.base_model}",
            f"**Training Date:** {self.training_date}",
            f"**Training Examples:** {self.num_training_examples:,}",
            f"**Evaluation Examples:** {self.num_eval_examples:,}",
            "",
            "## Metrics",
            "",
            "| Metric | Value | Improvement |",
            "|--------|-------|-------------|",
        ]
        
        # Add metric rows
        metrics = [
            ("Perplexity", self.perplexity, "perplexity"),
            ("CodeBLEU", self.codebleu_score, "codebleu_score"),
            ("HumanEval Pass@1", self.humaneval_pass_rate, "humaneval_pass_rate"),
            ("Reconstruction", self.reconstruction_score, "reconstruction_score"),
            ("Hallucination Rate", self.hallucination_rate, "hallucination_rate"),
        ]
        
        for name, value, key in metrics:
            if value is not None:
                improvement = self.improvement_over_base.get(key)
                imp_str = f"{improvement:+.1f}%" if improvement else "-"
                
                if key in ["perplexity", "hallucination_rate"]:
                    val_str = f"{value:.2f}"
                else:
                    val_str = f"{value:.2%}" if value <= 1 else f"{value:.2f}"
                
                lines.append(f"| {name} | {val_str} | {imp_str} |")
        
        # Additional metrics
        for name, value in self.additional_metrics.items():
            lines.append(f"| {name} | {value:.4f} | - |")
        
        lines.extend(["", "---", f"*Generated by AI Forge*"])
        
        content = "\n".join(lines)
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            logger.info(f"Markdown report saved to {path}")
        
        return content
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"📊 Evaluation Report: {self.model_name}",
            f"   Base: {self.base_model}",
        ]
        
        if self.perplexity:
            lines.append(f"   Perplexity: {self.perplexity:.2f}")
        if self.codebleu_score:
            lines.append(f"   CodeBLEU: {self.codebleu_score:.2%}")
        if self.humaneval_pass_rate:
            lines.append(f"   HumanEval: {self.humaneval_pass_rate:.2%}")
        if self.hallucination_rate is not None:
            lines.append(f"   Hallucination: {self.hallucination_rate:.2%}")
        
        return "\n".join(lines)
