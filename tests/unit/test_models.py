"""Unit tests for Dataset Versioning, Experiments, and Recipe models.

Tests cover:
- DatasetVersion model and status transitions
- Experiment model with metrics
- Recipe model with hardware-aware defaults
- Built-in recipes validation

Run with: pytest tests/unit/test_models.py -v
"""

import pytest
from pathlib import Path

# Import path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conductor.models import (
    DatasetVersion,
    DatasetVersionStatus,
    EvalMetrics,
    Experiment,
    ExperimentStatus,
    HardwareOverrides,
    HardwareProfile,
    Recipe,
    RecipeDefaults,
    TaskType,
    EvalSuiteConfig,
    BUILTIN_RECIPES,
)


# =============================================================================
# DatasetVersion Tests
# =============================================================================


class TestDatasetVersion:
    """Tests for DatasetVersion model."""

    def test_default_values(self):
        """Test DatasetVersion with minimal fields."""
        version = DatasetVersion(id="v1", dataset_id="ds1")
        assert version.id == "v1"
        assert version.dataset_id == "ds1"
        assert version.version == 1
        assert version.status == DatasetVersionStatus.DRAFT
        assert version.example_count == 0
        assert version.filters_applied == {}
        assert version.quality_stats == {}
        assert version.snapshot_hash == ""
        assert version.parent_version_id is None

    def test_full_version(self):
        """Test DatasetVersion with all fields."""
        version = DatasetVersion(
            id="v2",
            dataset_id="ds1",
            version=3,
            status=DatasetVersionStatus.READY,
            example_count=1500,
            filters_applied={"min_quality": 0.7},
            quality_stats={"avg_score": 0.85, "diversity": 0.92},
            snapshot_hash="abc123",
            parent_version_id="v1",
        )
        assert version.version == 3
        assert version.status == DatasetVersionStatus.READY
        assert version.example_count == 1500
        assert version.filters_applied["min_quality"] == 0.7

    def test_compute_hash(self):
        """Test hash computation."""
        version = DatasetVersion(id="v1", dataset_id="ds1")
        hash1 = version.compute_hash("test content")
        hash2 = version.compute_hash("test content")
        hash3 = version.compute_hash("different content")
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_status_enum(self):
        """Test all dataset version statuses."""
        assert DatasetVersionStatus.DRAFT == "draft"
        assert DatasetVersionStatus.READY == "ready"
        assert DatasetVersionStatus.ARCHIVED == "archived"

    def test_serialization(self):
        """Test model serialization/deserialization."""
        version = DatasetVersion(
            id="v1",
            dataset_id="ds1",
            version=2,
            status=DatasetVersionStatus.READY,
        )
        data = version.model_dump()
        restored = DatasetVersion(**data)
        assert restored.id == version.id
        assert restored.version == version.version
        assert restored.status == version.status


# =============================================================================
# EvalMetrics Tests
# =============================================================================


class TestEvalMetrics:
    """Tests for EvalMetrics model."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = EvalMetrics()
        assert metrics.loss is None
        assert metrics.eval_loss is None
        assert metrics.perplexity is None
        assert metrics.codebleu is None
        assert metrics.humaneval_pass_at_1 is None
        assert metrics.rouge_l is None
        assert metrics.exact_match is None
        assert metrics.custom == {}

    def test_full_metrics(self):
        """Test metrics with all values."""
        metrics = EvalMetrics(
            loss=0.5,
            eval_loss=0.6,
            perplexity=8.5,
            codebleu=0.45,
            humaneval_pass_at_1=0.15,
            rouge_l=0.7,
            exact_match=0.5,
            custom={"my_metric": 0.9},
        )
        assert metrics.loss == 0.5
        assert metrics.codebleu == 0.45
        assert metrics.custom["my_metric"] == 0.9

    def test_codebleu_range_validation(self):
        """Test CodeBLEU score range validation."""
        metrics = EvalMetrics(codebleu=0.0)
        assert metrics.codebleu == 0.0

        metrics = EvalMetrics(codebleu=1.0)
        assert metrics.codebleu == 1.0

        with pytest.raises(Exception):
            EvalMetrics(codebleu=1.5)

        with pytest.raises(Exception):
            EvalMetrics(codebleu=-0.1)


# =============================================================================
# Experiment Tests
# =============================================================================


class TestExperiment:
    """Tests for Experiment model."""

    def test_minimal_experiment(self):
        """Test experiment with minimal fields."""
        exp = Experiment(id="exp1", name="Test Run")
        assert exp.id == "exp1"
        assert exp.name == "Test Run"
        assert exp.status == ExperimentStatus.PENDING
        assert exp.description == ""
        assert exp.tags == []
        assert exp.artifacts == {}
        assert exp.job_id is None

    def test_full_experiment(self):
        """Test experiment with all fields."""
        exp = Experiment(
            id="exp2",
            name="Code Fine-Tune v2",
            description="Fine-tuning on internal codebase",
            status=ExperimentStatus.COMPLETED,
            base_model="unsloth/Llama-3.2-3B-Instruct",
            dataset_id="ds1",
            dataset_version_id="v2",
            recipe_id="code-specialization",
            hyperparameters={"epochs": 5, "lr": 2e-4, "rank": 128},
            metrics=EvalMetrics(loss=0.3, codebleu=0.55),
            tags=["code", "v2"],
            artifacts={"checkpoint": "/output/exp2/best"},
            job_id="job123",
            duration_seconds=3600.5,
        )
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.metrics.loss == 0.3
        assert exp.metrics.codebleu == 0.55
        assert "code" in exp.tags
        assert exp.duration_seconds == 3600.5

    def test_experiment_status_enum(self):
        """Test all experiment statuses."""
        assert ExperimentStatus.PENDING == "pending"
        assert ExperimentStatus.RUNNING == "running"
        assert ExperimentStatus.COMPLETED == "completed"
        assert ExperimentStatus.FAILED == "failed"
        assert ExperimentStatus.CANCELLED == "cancelled"

    def test_experiment_serialization(self):
        """Test experiment round-trip serialization."""
        exp = Experiment(
            id="exp1",
            name="Test",
            metrics=EvalMetrics(loss=0.5, codebleu=0.4),
            tags=["test"],
        )
        data = exp.model_dump()
        restored = Experiment(**data)
        assert restored.name == exp.name
        assert restored.metrics.loss == exp.metrics.loss
        assert restored.metrics.codebleu == exp.metrics.codebleu
        assert restored.tags == exp.tags


# =============================================================================
# RecipeDefaults Tests
# =============================================================================


class TestRecipeDefaults:
    """Tests for RecipeDefaults model."""

    def test_default_values(self):
        """Test default recipe hyperparameters."""
        defaults = RecipeDefaults()
        assert defaults.epochs == 3
        assert defaults.learning_rate == 2e-4
        assert defaults.rank == 64
        assert defaults.batch_size == 2
        assert defaults.use_pissa is True
        assert defaults.gradient_accumulation_steps == 4
        assert defaults.warmup_ratio == 0.03
        assert defaults.scheduler == "cosine"
        assert defaults.max_seq_length == 2048

    def test_custom_defaults(self):
        """Test custom recipe defaults."""
        defaults = RecipeDefaults(
            epochs=5,
            learning_rate=1e-4,
            rank=128,
            batch_size=4,
        )
        assert defaults.epochs == 5
        assert defaults.learning_rate == 1e-4
        assert defaults.rank == 128
        assert defaults.batch_size == 4


# =============================================================================
# Recipe Tests
# =============================================================================


class TestRecipe:
    """Tests for Recipe model."""

    def test_minimal_recipe(self):
        """Test recipe with minimal fields."""
        recipe = Recipe(
            id="test",
            name="Test Recipe",
            task_type=TaskType.INSTRUCTION_TUNING,
        )
        assert recipe.id == "test"
        assert recipe.task_type == TaskType.INSTRUCTION_TUNING
        assert recipe.defaults.epochs == 3
        assert recipe.is_builtin is False
        assert recipe.supported_models == []

    def test_hardware_overrides(self):
        """Test hardware-aware default adjustment."""
        recipe = Recipe(
            id="test",
            name="Test",
            task_type=TaskType.CODE_SPECIALIZATION,
            defaults=RecipeDefaults(
                batch_size=2,
                rank=64,
                gradient_accumulation_steps=4,
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
                ),
            ],
        )

        # Default (medium) - should match base defaults
        medium = recipe.get_defaults_for_hardware(HardwareProfile.MEDIUM)
        assert medium.batch_size == 2
        assert medium.rank == 64

        # Low hardware - should apply overrides
        low = recipe.get_defaults_for_hardware(HardwareProfile.LOW)
        assert low.batch_size == 1
        assert low.rank == 32
        assert low.gradient_accumulation_steps == 8
        assert low.max_seq_length == 2048

        # High hardware - should apply overrides
        high = recipe.get_defaults_for_hardware(HardwareProfile.HIGH)
        assert high.batch_size == 4
        assert high.rank == 128
        assert high.gradient_accumulation_steps == 2
        # max_seq_length not overridden for HIGH, should keep default
        assert high.max_seq_length == 4096

    def test_task_type_enum(self):
        """Test all task types."""
        assert TaskType.INSTRUCTION_TUNING == "instruction_tuning"
        assert TaskType.DOMAIN_ADAPTATION == "domain_adaptation"
        assert TaskType.CODE_SPECIALIZATION == "code_specialization"
        assert TaskType.QA_FINETUNING == "qa_finetuning"


# =============================================================================
# Built-in Recipes Tests
# =============================================================================


class TestBuiltinRecipes:
    """Tests for built-in recipe definitions."""

    def test_builtin_count(self):
        """Test that all built-in recipes are defined."""
        assert len(BUILTIN_RECIPES) == 4

    def test_all_builtins_are_builtin(self):
        """Test that built-in flag is set."""
        for recipe in BUILTIN_RECIPES:
            assert recipe.is_builtin is True

    def test_unique_ids(self):
        """Test that all built-in recipes have unique IDs."""
        ids = [r.id for r in BUILTIN_RECIPES]
        assert len(ids) == len(set(ids))

    def test_instruction_tuning_recipe(self):
        """Test instruction tuning recipe."""
        recipe = next(r for r in BUILTIN_RECIPES if r.id == "instruction-tuning")
        assert recipe.task_type == TaskType.INSTRUCTION_TUNING
        assert len(recipe.supported_models) > 0
        assert "instruction" in recipe.tags
        assert recipe.defaults.use_pissa is True

    def test_code_specialization_recipe(self):
        """Test code specialization recipe."""
        recipe = next(r for r in BUILTIN_RECIPES if r.id == "code-specialization")
        assert recipe.task_type == TaskType.CODE_SPECIALIZATION
        assert "codebleu" in recipe.eval_suite.metrics
        assert "humaneval_pass_at_1" in recipe.eval_suite.metrics

    def test_domain_adaptation_recipe(self):
        """Test domain adaptation recipe."""
        recipe = next(r for r in BUILTIN_RECIPES if r.id == "domain-adaptation")
        assert recipe.task_type == TaskType.DOMAIN_ADAPTATION
        assert recipe.defaults.epochs == 5
        assert recipe.defaults.scheduler == "linear"

    def test_qa_finetuning_recipe(self):
        """Test QA fine-tuning recipe."""
        recipe = next(r for r in BUILTIN_RECIPES if r.id == "qa-finetuning")
        assert recipe.task_type == TaskType.QA_FINETUNING
        assert "rouge_l" in recipe.eval_suite.metrics
        assert "exact_match" in recipe.eval_suite.metrics

    def test_hardware_overrides_exist(self):
        """Test that built-in recipes have hardware overrides."""
        for recipe in BUILTIN_RECIPES:
            assert len(recipe.hardware_overrides) >= 2, (
                f"Recipe {recipe.id} should have at least LOW and HIGH overrides"
            )

    def test_eval_suites_defined(self):
        """Test that built-in recipes have evaluation suites."""
        for recipe in BUILTIN_RECIPES:
            assert len(recipe.eval_suite.metrics) > 0, (
                f"Recipe {recipe.id} should have eval metrics"
            )

    def test_dataset_requirements(self):
        """Test that built-in recipes specify dataset requirements."""
        for recipe in BUILTIN_RECIPES:
            assert "format" in recipe.dataset_requirements, (
                f"Recipe {recipe.id} should specify dataset format"
            )
            assert "min_examples" in recipe.dataset_requirements, (
                f"Recipe {recipe.id} should specify minimum examples"
            )
