"""Models for Dataset Versioning, Experiments, and Training Recipes.

This module defines the data models for:
- Dataset versioning and lifecycle management
- Experiment/Run tracking with metrics
- Training recipe definitions with pre-filled configurations
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Dataset Versioning Models
# =============================================================================


class DatasetVersionStatus(str, Enum):
    """Status of a dataset version."""

    DRAFT = "draft"
    READY = "ready"
    ARCHIVED = "archived"


class DatasetVersion(BaseModel):
    """A versioned snapshot of a dataset.

    Attributes:
        id: Unique version identifier.
        dataset_id: Parent dataset ID.
        version: Semantic version number.
        status: Version status.
        example_count: Number of training examples.
        filters_applied: Filters used to create this version.
        quality_stats: Quality statistics for this version.
        snapshot_hash: SHA-256 hash of the dataset content.
        created_at: Creation timestamp.
        parent_version_id: ID of the parent version (if derived).
    """

    id: str
    dataset_id: str
    version: int = Field(default=1, ge=1)
    status: DatasetVersionStatus = Field(default=DatasetVersionStatus.DRAFT)
    example_count: int = Field(default=0, ge=0)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    quality_stats: dict[str, float] = Field(default_factory=dict)
    snapshot_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    parent_version_id: Optional[str] = None

    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of dataset content.

        Args:
            content: The raw dataset content to hash.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# Experiment / Run Models
# =============================================================================


class ExperimentStatus(str, Enum):
    """Status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvalMetrics(BaseModel):
    """Evaluation metrics for an experiment run.

    Attributes:
        loss: Final training loss.
        eval_loss: Final evaluation loss.
        perplexity: Model perplexity.
        codebleu: CodeBLEU score (0-1) for code tasks.
        humaneval_pass_at_1: HumanEval pass@1 rate (0-1).
        rouge_l: ROUGE-L score (0-1) for text tasks.
        exact_match: Exact match rate (0-1).
        custom: Additional custom metrics.
    """

    loss: Optional[float] = None
    eval_loss: Optional[float] = None
    perplexity: Optional[float] = None
    codebleu: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    humaneval_pass_at_1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rouge_l: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    exact_match: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    custom: dict[str, float] = Field(default_factory=dict)


class Experiment(BaseModel):
    """An experiment run tracking training configuration and results.

    Attributes:
        id: Unique experiment identifier.
        name: Human-readable experiment name.
        description: Experiment description.
        status: Current status.
        base_model: Base model used.
        dataset_id: Dataset used for training.
        dataset_version_id: Specific dataset version used.
        recipe_id: Recipe used (if any).
        hyperparameters: Training hyperparameters.
        metrics: Evaluation metrics.
        tags: User-defined tags for organization.
        artifacts: Paths to output artifacts.
        job_id: Associated training job ID.
        created_at: Creation timestamp.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        duration_seconds: Total training duration.
        error: Error message if failed.
    """

    id: str
    name: str
    description: str = Field(default="")
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    base_model: str = Field(default="")
    dataset_id: Optional[str] = None
    dataset_version_id: Optional[str] = None
    recipe_id: Optional[str] = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: EvalMetrics = Field(default_factory=EvalMetrics)
    tags: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    job_id: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
# Training Recipe Models
# =============================================================================


class TaskType(str, Enum):
    """Type of training task."""

    INSTRUCTION_TUNING = "instruction_tuning"
    DOMAIN_ADAPTATION = "domain_adaptation"
    CODE_SPECIALIZATION = "code_specialization"
    QA_FINETUNING = "qa_finetuning"


class HardwareProfile(str, Enum):
    """Hardware profile for resource-aware defaults."""

    LOW = "low"  # 8GB VRAM (e.g., M1/M2 base)
    MEDIUM = "medium"  # 16GB VRAM (e.g., M1/M2 Pro)
    HIGH = "high"  # 32GB+ VRAM (e.g., M1/M2 Max/Ultra)


class RecipeDefaults(BaseModel):
    """Default hyperparameters for a recipe.

    Attributes:
        epochs: Default number of training epochs.
        learning_rate: Default learning rate.
        rank: Default LoRA/PiSSA rank.
        batch_size: Default batch size.
        use_pissa: Whether to use PiSSA initialization.
        gradient_accumulation_steps: Default gradient accumulation.
        warmup_ratio: Default warmup ratio.
        scheduler: Default LR scheduler.
        max_seq_length: Default max sequence length.
    """

    epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=2e-4, ge=1e-7, le=1.0)
    rank: int = Field(default=64, ge=1, le=512)
    batch_size: int = Field(default=2, ge=1, le=64)
    use_pissa: bool = Field(default=True)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    scheduler: str = Field(default="cosine")
    max_seq_length: int = Field(default=2048, ge=128, le=131072)


class HardwareOverrides(BaseModel):
    """Hardware-specific overrides for recipe defaults.

    Attributes:
        profile: Hardware profile name.
        batch_size: Override batch size.
        rank: Override rank.
        gradient_accumulation_steps: Override gradient accumulation.
        max_seq_length: Override max sequence length.
    """

    profile: HardwareProfile
    batch_size: Optional[int] = None
    rank: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    max_seq_length: Optional[int] = None


class EvalSuiteConfig(BaseModel):
    """Evaluation suite configuration for a recipe.

    Attributes:
        metrics: List of metrics to evaluate.
        pass_thresholds: Minimum thresholds for each metric.
    """

    metrics: list[str] = Field(default_factory=list)
    pass_thresholds: dict[str, float] = Field(default_factory=dict)


class Recipe(BaseModel):
    """A training recipe with opinionated defaults for a specific task.

    Attributes:
        id: Unique recipe identifier.
        name: Human-readable recipe name.
        description: Recipe description.
        task_type: Type of training task.
        supported_models: List of compatible base models.
        defaults: Default hyperparameters.
        hardware_overrides: Per-hardware-profile overrides.
        dataset_requirements: Expected dataset schema/format requirements.
        eval_suite: Recommended evaluation configuration.
        tags: Recipe tags.
        is_builtin: Whether this is a built-in recipe.
    """

    id: str
    name: str
    description: str = Field(default="")
    task_type: TaskType
    supported_models: list[str] = Field(default_factory=list)
    defaults: RecipeDefaults = Field(default_factory=RecipeDefaults)
    hardware_overrides: list[HardwareOverrides] = Field(default_factory=list)
    dataset_requirements: dict[str, Any] = Field(default_factory=dict)
    eval_suite: EvalSuiteConfig = Field(default_factory=EvalSuiteConfig)
    tags: list[str] = Field(default_factory=list)
    is_builtin: bool = Field(default=False)

    def get_defaults_for_hardware(self, profile: HardwareProfile) -> RecipeDefaults:
        """Get recipe defaults adjusted for the given hardware profile.

        Args:
            profile: The hardware profile to apply overrides for.

        Returns:
            A copy of the recipe defaults with hardware-specific overrides applied.
        """
        defaults = self.defaults.model_copy()
        for override in self.hardware_overrides:
            if override.profile == profile:
                if override.batch_size is not None:
                    defaults.batch_size = override.batch_size
                if override.rank is not None:
                    defaults.rank = override.rank
                if override.gradient_accumulation_steps is not None:
                    defaults.gradient_accumulation_steps = (
                        override.gradient_accumulation_steps
                    )
                if override.max_seq_length is not None:
                    defaults.max_seq_length = override.max_seq_length
                break
        return defaults


# =============================================================================
# Built-in Recipes
# =============================================================================

BUILTIN_RECIPES: list[Recipe] = [
    Recipe(
        id="instruction-tuning",
        name="Instruction Tuning",
        description=(
            "Fine-tune a model to follow instructions. Best for creating "
            "a general-purpose assistant from a base model using "
            "instruction-response pairs."
        ),
        task_type=TaskType.INSTRUCTION_TUNING,
        supported_models=[
            "unsloth/Llama-3.2-3B-Instruct",
            "unsloth/Llama-3.2-7B-Instruct",
            "unsloth/mistral-7b-instruct-v0.3",
        ],
        defaults=RecipeDefaults(
            epochs=3,
            learning_rate=2e-4,
            rank=64,
            batch_size=2,
            use_pissa=True,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            scheduler="cosine",
            max_seq_length=2048,
        ),
        hardware_overrides=[
            HardwareOverrides(
                profile=HardwareProfile.LOW,
                batch_size=1,
                rank=32,
                gradient_accumulation_steps=8,
                max_seq_length=1024,
            ),
            HardwareOverrides(
                profile=HardwareProfile.HIGH,
                batch_size=4,
                rank=128,
                gradient_accumulation_steps=2,
                max_seq_length=4096,
            ),
        ],
        dataset_requirements={
            "format": "alpaca",
            "min_examples": 100,
            "required_fields": ["instruction", "output"],
        },
        eval_suite=EvalSuiteConfig(
            metrics=["perplexity", "rouge_l", "exact_match"],
            pass_thresholds={"perplexity": 10.0},
        ),
        tags=["instruction", "general", "beginner-friendly"],
        is_builtin=True,
    ),
    Recipe(
        id="domain-adaptation",
        name="Domain Adaptation",
        description=(
            "Adapt a base model to a specific domain using internal "
            "documents, manuals, or knowledge bases. Ideal for creating "
            "a domain expert."
        ),
        task_type=TaskType.DOMAIN_ADAPTATION,
        supported_models=[
            "unsloth/Llama-3.2-3B-Instruct",
            "unsloth/Llama-3.2-7B-Instruct",
            "unsloth/Llama-3.1-13B-Instruct",
        ],
        defaults=RecipeDefaults(
            epochs=5,
            learning_rate=1e-4,
            rank=32,
            batch_size=2,
            use_pissa=True,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            scheduler="linear",
            max_seq_length=4096,
        ),
        hardware_overrides=[
            HardwareOverrides(
                profile=HardwareProfile.LOW,
                batch_size=1,
                rank=16,
                gradient_accumulation_steps=8,
                max_seq_length=2048,
            ),
            HardwareOverrides(
                profile=HardwareProfile.HIGH,
                batch_size=4,
                rank=64,
                gradient_accumulation_steps=2,
            ),
        ],
        dataset_requirements={
            "format": "alpaca",
            "min_examples": 200,
            "required_fields": ["instruction", "context", "output"],
        },
        eval_suite=EvalSuiteConfig(
            metrics=["perplexity", "rouge_l"],
            pass_thresholds={"perplexity": 8.0, "rouge_l": 0.3},
        ),
        tags=["domain", "knowledge", "documents"],
        is_builtin=True,
    ),
    Recipe(
        id="code-specialization",
        name="Code Specialization",
        description=(
            "Specialize a model for code generation and understanding "
            "using code + test pairs. Uses CodeBLEU and HumanEval "
            "for evaluation."
        ),
        task_type=TaskType.CODE_SPECIALIZATION,
        supported_models=[
            "unsloth/codellama-13b-instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "unsloth/Llama-3.2-3B-Instruct",
        ],
        defaults=RecipeDefaults(
            epochs=3,
            learning_rate=2e-4,
            rank=64,
            batch_size=2,
            use_pissa=True,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            scheduler="cosine",
            max_seq_length=4096,
        ),
        hardware_overrides=[
            HardwareOverrides(
                profile=HardwareProfile.LOW,
                batch_size=1,
                rank=32,
                gradient_accumulation_steps=8,
                max_seq_length=2048,
            ),
            HardwareOverrides(
                profile=HardwareProfile.HIGH,
                batch_size=4,
                rank=128,
                gradient_accumulation_steps=2,
                max_seq_length=8192,
            ),
        ],
        dataset_requirements={
            "format": "alpaca",
            "min_examples": 500,
            "required_fields": ["instruction", "output"],
            "recommended_fields": ["tests", "language"],
        },
        eval_suite=EvalSuiteConfig(
            metrics=["codebleu", "humaneval_pass_at_1", "perplexity"],
            pass_thresholds={
                "codebleu": 0.3,
                "humaneval_pass_at_1": 0.1,
            },
        ),
        tags=["code", "programming", "generation"],
        is_builtin=True,
    ),
    Recipe(
        id="qa-finetuning",
        name="QA Fine-Tuning",
        description=(
            "Fine-tune for question-answering tasks with context-grounded "
            "responses. Uses RAFT-style distractor documents for robust QA."
        ),
        task_type=TaskType.QA_FINETUNING,
        supported_models=[
            "unsloth/Llama-3.2-3B-Instruct",
            "unsloth/Llama-3.2-7B-Instruct",
            "unsloth/mistral-7b-instruct-v0.3",
        ],
        defaults=RecipeDefaults(
            epochs=3,
            learning_rate=2e-4,
            rank=64,
            batch_size=2,
            use_pissa=True,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            scheduler="cosine",
            max_seq_length=2048,
        ),
        hardware_overrides=[
            HardwareOverrides(
                profile=HardwareProfile.LOW,
                batch_size=1,
                rank=32,
                gradient_accumulation_steps=8,
                max_seq_length=1024,
            ),
            HardwareOverrides(
                profile=HardwareProfile.HIGH,
                batch_size=4,
                rank=128,
                gradient_accumulation_steps=2,
                max_seq_length=4096,
            ),
        ],
        dataset_requirements={
            "format": "alpaca",
            "min_examples": 200,
            "required_fields": ["instruction", "context", "output"],
        },
        eval_suite=EvalSuiteConfig(
            metrics=["perplexity", "rouge_l", "exact_match"],
            pass_thresholds={
                "perplexity": 8.0,
                "rouge_l": 0.4,
                "exact_match": 0.3,
            },
        ),
        tags=["qa", "raft", "context-grounded"],
        is_builtin=True,
    ),
]
