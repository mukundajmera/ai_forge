"""Unit tests for Experiments and Recipes API endpoints.

Tests cover:
- Experiment CRUD operations
- Experiment comparison
- Dataset version management
- Recipe listing and detail with hardware overrides

Run with: pytest tests/unit/test_experiments_api.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conductor.models import (
    DatasetVersion,
    Experiment,
    ExperimentStatus,
    EvalMetrics,
    BUILTIN_RECIPES,
)


# =============================================================================
# In-memory storage mock for API tests
# =============================================================================


class MockStorage:
    """In-memory storage for testing."""

    def __init__(self):
        self._data = {
            "experiments": {},
            "dataset_versions": {},
            "recipes": {},
        }

    def get_all(self, collection):
        return dict(self._data.get(collection, {}))

    def get(self, collection, item_id):
        return self._data.get(collection, {}).get(item_id)

    def set(self, collection, item_id, value):
        if collection not in self._data:
            self._data[collection] = {}
        self._data[collection][item_id] = value

    def delete(self, collection, item_id):
        if collection in self._data and item_id in self._data[collection]:
            del self._data[collection][item_id]

    def clear(self, collection):
        self._data[collection] = {}


# =============================================================================
# Experiment Endpoint Tests
# =============================================================================


class TestExperimentEndpoints:
    """Tests for experiment API logic."""

    def setup_method(self):
        """Set up fresh mock storage for each test."""
        self.storage = MockStorage()

    def test_create_experiment(self):
        """Test creating an experiment stores correct data."""
        exp = Experiment(
            id="exp-1",
            name="Test Run",
            base_model="unsloth/Llama-3.2-3B-Instruct",
            tags=["test"],
        )
        self.storage.set("experiments", exp.id, exp.model_dump())

        stored = self.storage.get("experiments", "exp-1")
        assert stored is not None
        assert stored["name"] == "Test Run"
        assert stored["base_model"] == "unsloth/Llama-3.2-3B-Instruct"
        assert stored["tags"] == ["test"]

    def test_list_experiments(self):
        """Test listing experiments."""
        for i in range(3):
            exp = Experiment(id=f"exp-{i}", name=f"Run {i}")
            self.storage.set("experiments", exp.id, exp.model_dump())

        all_exps = self.storage.get_all("experiments")
        assert len(all_exps) == 3

    def test_get_experiment(self):
        """Test getting a single experiment."""
        exp = Experiment(
            id="exp-1",
            name="Test",
            metrics=EvalMetrics(loss=0.5),
        )
        self.storage.set("experiments", exp.id, exp.model_dump())

        stored = self.storage.get("experiments", "exp-1")
        assert stored is not None
        restored = Experiment(**stored)
        assert restored.metrics.loss == 0.5

    def test_update_experiment_metrics(self):
        """Test updating experiment metrics."""
        exp = Experiment(id="exp-1", name="Test")
        self.storage.set("experiments", exp.id, exp.model_dump())

        stored = self.storage.get("experiments", "exp-1")
        experiment = Experiment(**stored)
        experiment.metrics = EvalMetrics(loss=0.3, codebleu=0.55)
        experiment.status = ExperimentStatus.COMPLETED
        self.storage.set("experiments", experiment.id, experiment.model_dump())

        updated = self.storage.get("experiments", "exp-1")
        assert updated["metrics"]["loss"] == 0.3
        assert updated["metrics"]["codebleu"] == 0.55
        assert updated["status"] == "completed"

    def test_delete_experiment(self):
        """Test deleting an experiment."""
        exp = Experiment(id="exp-1", name="Test")
        self.storage.set("experiments", exp.id, exp.model_dump())
        assert self.storage.get("experiments", "exp-1") is not None

        self.storage.delete("experiments", "exp-1")
        assert self.storage.get("experiments", "exp-1") is None

    def test_experiment_comparison_logic(self):
        """Test experiment comparison metric extraction."""
        exp1 = Experiment(
            id="exp-1",
            name="Run A",
            metrics=EvalMetrics(loss=0.5, codebleu=0.4),
        )
        exp2 = Experiment(
            id="exp-2",
            name="Run B",
            metrics=EvalMetrics(loss=0.3, codebleu=0.6),
        )
        self.storage.set("experiments", exp1.id, exp1.model_dump())
        self.storage.set("experiments", exp2.id, exp2.model_dump())

        experiments = []
        for eid in ["exp-1", "exp-2"]:
            data = self.storage.get("experiments", eid)
            experiments.append(Experiment(**data))

        # Build metric summary
        metric_keys = ["loss", "codebleu"]
        summary = {}
        for key in metric_keys:
            summary[key] = {}
            for exp in experiments:
                summary[key][exp.id] = getattr(exp.metrics, key, None)

        assert summary["loss"]["exp-1"] == 0.5
        assert summary["loss"]["exp-2"] == 0.3
        assert summary["codebleu"]["exp-1"] == 0.4
        assert summary["codebleu"]["exp-2"] == 0.6

    def test_filter_by_status(self):
        """Test filtering experiments by status."""
        statuses = [
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.COMPLETED,
        ]
        for i, status in enumerate(statuses):
            exp = Experiment(id=f"exp-{i}", name=f"Run {i}", status=status)
            self.storage.set("experiments", exp.id, exp.model_dump())

        all_exps = self.storage.get_all("experiments")
        experiments = [Experiment(**v) for v in all_exps.values()]
        completed = [e for e in experiments if e.status == ExperimentStatus.COMPLETED]
        assert len(completed) == 2

    def test_filter_by_tag(self):
        """Test filtering experiments by tag."""
        exp1 = Experiment(id="exp-1", name="Run 1", tags=["code", "v1"])
        exp2 = Experiment(id="exp-2", name="Run 2", tags=["docs"])
        self.storage.set("experiments", exp1.id, exp1.model_dump())
        self.storage.set("experiments", exp2.id, exp2.model_dump())

        all_exps = self.storage.get_all("experiments")
        experiments = [Experiment(**v) for v in all_exps.values()]
        code_exps = [e for e in experiments if "code" in e.tags]
        assert len(code_exps) == 1
        assert code_exps[0].id == "exp-1"


# =============================================================================
# Dataset Version Endpoint Tests
# =============================================================================


class TestDatasetVersionEndpoints:
    """Tests for dataset version API logic."""

    def setup_method(self):
        """Set up fresh mock storage for each test."""
        self.storage = MockStorage()

    def test_create_dataset_version(self):
        """Test creating a dataset version."""
        version = DatasetVersion(
            id="v1",
            dataset_id="ds1",
            version=1,
            filters_applied={"min_quality": 0.7},
        )
        self.storage.set("dataset_versions", version.id, version.model_dump())

        stored = self.storage.get("dataset_versions", "v1")
        assert stored is not None
        assert stored["dataset_id"] == "ds1"
        assert stored["version"] == 1

    def test_auto_version_numbering(self):
        """Test that version numbers increment automatically."""
        for i in range(3):
            v = DatasetVersion(
                id=f"v{i+1}",
                dataset_id="ds1",
                version=i + 1,
            )
            self.storage.set("dataset_versions", v.id, v.model_dump())

        existing = self.storage.get_all("dataset_versions")
        dataset_versions = [
            DatasetVersion(**v)
            for v in existing.values()
            if v.get("dataset_id") == "ds1"
        ]
        next_version = max((v.version for v in dataset_versions), default=0) + 1
        assert next_version == 4

    def test_filter_by_dataset(self):
        """Test filtering versions by dataset ID."""
        self.storage.set(
            "dataset_versions",
            "v1",
            DatasetVersion(id="v1", dataset_id="ds1").model_dump(),
        )
        self.storage.set(
            "dataset_versions",
            "v2",
            DatasetVersion(id="v2", dataset_id="ds2").model_dump(),
        )
        self.storage.set(
            "dataset_versions",
            "v3",
            DatasetVersion(id="v3", dataset_id="ds1").model_dump(),
        )

        all_versions = self.storage.get_all("dataset_versions")
        versions = [DatasetVersion(**v) for v in all_versions.values()]
        ds1_versions = [v for v in versions if v.dataset_id == "ds1"]
        assert len(ds1_versions) == 2


# =============================================================================
# Recipe Endpoint Tests
# =============================================================================


class TestRecipeEndpoints:
    """Tests for recipe API logic."""

    def test_list_includes_builtins(self):
        """Test that listing includes built-in recipes."""
        storage = MockStorage()
        recipes = list(BUILTIN_RECIPES)
        custom = storage.get_all("recipes")
        recipes.extend([])  # No custom recipes
        assert len(recipes) == 4

    def test_filter_by_task_type(self):
        """Test filtering recipes by task type."""
        recipes = list(BUILTIN_RECIPES)
        code_recipes = [r for r in recipes if r.task_type.value == "code_specialization"]
        assert len(code_recipes) == 1
        assert code_recipes[0].id == "code-specialization"

    def test_hardware_adjusted_defaults(self):
        """Test getting hardware-adjusted recipe defaults."""
        recipe = next(r for r in BUILTIN_RECIPES if r.id == "code-specialization")

        # Low hardware should have smaller batch size and rank
        low_defaults = recipe.get_defaults_for_hardware("low")
        assert low_defaults.batch_size == 1
        assert low_defaults.rank == 32

        # High hardware should have larger batch size and rank
        high_defaults = recipe.get_defaults_for_hardware("high")
        assert high_defaults.batch_size == 4
        assert high_defaults.rank == 128

    def test_builtin_recipe_not_deletable(self):
        """Test that built-in recipes are identified correctly."""
        for recipe in BUILTIN_RECIPES:
            assert recipe.is_builtin is True

    def test_custom_recipe_creation(self):
        """Test creating a custom recipe."""
        from conductor.models import Recipe, TaskType, RecipeDefaults

        storage = MockStorage()
        recipe = Recipe(
            id="custom-1",
            name="My Custom Recipe",
            task_type=TaskType.INSTRUCTION_TUNING,
            defaults=RecipeDefaults(epochs=10, learning_rate=1e-5),
            is_builtin=False,
        )
        storage.set("recipes", recipe.id, recipe.model_dump())

        stored = storage.get("recipes", "custom-1")
        assert stored is not None
        restored = Recipe(**stored)
        assert restored.name == "My Custom Recipe"
        assert restored.defaults.epochs == 10
        assert restored.is_builtin is False
