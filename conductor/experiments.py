"""Experiments API Router - Endpoints for experiment tracking and comparison.

This module provides REST API endpoints for:
- Creating and managing experiments/runs
- Tracking experiment metrics and status
- Comparing experiments side-by-side
- Managing dataset versions

Example:
    >>> from conductor.experiments import router as experiments_router
    >>> app.include_router(experiments_router, prefix="/api")
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from conductor.models import (
    DatasetVersion,
    DatasetVersionStatus,
    EvalMetrics,
    Experiment,
    ExperimentStatus,
)
from conductor.persistence import storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Experiments"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""

    name: str
    description: str = ""
    base_model: str
    dataset_id: Optional[str] = None
    dataset_version_id: Optional[str] = None
    recipe_id: Optional[str] = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class UpdateExperimentRequest(BaseModel):
    """Request to update an experiment."""

    status: Optional[ExperimentStatus] = None
    metrics: Optional[EvalMetrics] = None
    tags: Optional[list[str]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    artifacts: Optional[dict[str, str]] = None


class CreateDatasetVersionRequest(BaseModel):
    """Request to create a new dataset version."""

    dataset_id: str
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    parent_version_id: Optional[str] = None


class CompareExperimentsRequest(BaseModel):
    """Request to compare multiple experiments."""

    experiment_ids: list[str] = Field(min_length=2, max_length=10)


class ExperimentComparison(BaseModel):
    """Comparison result for multiple experiments."""

    experiments: list[Experiment]
    metric_summary: dict[str, dict[str, Optional[float]]]


# =============================================================================
# Experiment Endpoints
# =============================================================================


@router.get("/experiments")
async def list_experiments(
    status: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
) -> list[Experiment]:
    """List all experiments with optional filtering."""
    all_experiments = storage.get_all("experiments")
    experiments = [Experiment(**v) for v in all_experiments.values()]

    if status:
        experiments = [e for e in experiments if e.status.value == status]
    if tag:
        experiments = [e for e in experiments if tag in e.tags]

    # Sort by created_at descending
    experiments.sort(key=lambda e: e.created_at, reverse=True)
    return experiments[:limit]


@router.post("/experiments")
async def create_experiment(request: CreateExperimentRequest) -> Experiment:
    """Create a new experiment."""
    experiment = Experiment(
        id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        base_model=request.base_model,
        dataset_id=request.dataset_id,
        dataset_version_id=request.dataset_version_id,
        recipe_id=request.recipe_id,
        hyperparameters=request.hyperparameters,
        tags=request.tags,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    storage.set("experiments", experiment.id, experiment.model_dump())
    logger.info(f"Created experiment: {experiment.id} ({experiment.name})")
    return experiment


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> Experiment:
    """Get a specific experiment by ID."""
    data = storage.get("experiments", experiment_id)
    if not data:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return Experiment(**data)


@router.patch("/experiments/{experiment_id}")
async def update_experiment(
    experiment_id: str, request: UpdateExperimentRequest
) -> Experiment:
    """Update an experiment's status, metrics, or metadata."""
    data = storage.get("experiments", experiment_id)
    if not data:
        raise HTTPException(status_code=404, detail="Experiment not found")

    experiment = Experiment(**data)
    update_data = request.model_dump(exclude_none=True)

    if "metrics" in update_data:
        existing_metrics = experiment.metrics.model_dump()
        existing_metrics.update(update_data["metrics"])
        update_data["metrics"] = EvalMetrics(**existing_metrics)

    for key, value in update_data.items():
        setattr(experiment, key, value)

    storage.set("experiments", experiment.id, experiment.model_dump())
    logger.info(f"Updated experiment: {experiment.id}")
    return experiment


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str) -> dict[str, str]:
    """Delete an experiment."""
    data = storage.get("experiments", experiment_id)
    if not data:
        raise HTTPException(status_code=404, detail="Experiment not found")

    storage.delete("experiments", experiment_id)
    logger.info(f"Deleted experiment: {experiment_id}")
    return {"status": "deleted", "id": experiment_id}


@router.post("/experiments/compare")
async def compare_experiments(
    request: CompareExperimentsRequest,
) -> ExperimentComparison:
    """Compare multiple experiments side-by-side."""
    experiments = []
    for exp_id in request.experiment_ids:
        data = storage.get("experiments", exp_id)
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment not found: {exp_id}",
            )
        experiments.append(Experiment(**data))

    # Build metric summary (dynamically from EvalMetrics model fields)
    metric_keys = [
        k for k, v in EvalMetrics.model_fields.items()
        if k != "custom"
    ]
    metric_summary: dict[str, dict[str, Optional[float]]] = {}
    for key in metric_keys:
        metric_summary[key] = {}
        for exp in experiments:
            metric_summary[key][exp.id] = getattr(exp.metrics, key, None)

    return ExperimentComparison(
        experiments=experiments,
        metric_summary=metric_summary,
    )


# =============================================================================
# Dataset Version Endpoints
# =============================================================================


@router.get("/dataset-versions")
async def list_dataset_versions(
    dataset_id: Optional[str] = None,
) -> list[DatasetVersion]:
    """List dataset versions, optionally filtered by dataset ID."""
    all_versions = storage.get_all("dataset_versions")
    versions = [DatasetVersion(**v) for v in all_versions.values()]

    if dataset_id:
        versions = [v for v in versions if v.dataset_id == dataset_id]

    versions.sort(key=lambda v: v.version, reverse=True)
    return versions


@router.post("/dataset-versions")
async def create_dataset_version(
    request: CreateDatasetVersionRequest,
) -> DatasetVersion:
    """Create a new dataset version."""
    # Determine version number
    existing = storage.get_all("dataset_versions")
    dataset_versions = [
        DatasetVersion(**v)
        for v in existing.values()
        if v.get("dataset_id") == request.dataset_id
    ]
    next_version = max((v.version for v in dataset_versions), default=0) + 1

    version = DatasetVersion(
        id=str(uuid.uuid4()),
        dataset_id=request.dataset_id,
        version=next_version,
        filters_applied=request.filters_applied,
        parent_version_id=request.parent_version_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    storage.set("dataset_versions", version.id, version.model_dump())
    logger.info(
        f"Created dataset version: {version.id} "
        f"(dataset={request.dataset_id}, v{next_version})"
    )
    return version


@router.get("/dataset-versions/{version_id}")
async def get_dataset_version(version_id: str) -> DatasetVersion:
    """Get a specific dataset version."""
    data = storage.get("dataset_versions", version_id)
    if not data:
        raise HTTPException(
            status_code=404, detail="Dataset version not found"
        )
    return DatasetVersion(**data)


@router.patch("/dataset-versions/{version_id}")
async def update_dataset_version_status(
    version_id: str,
    status: DatasetVersionStatus,
) -> DatasetVersion:
    """Update a dataset version's status."""
    data = storage.get("dataset_versions", version_id)
    if not data:
        raise HTTPException(
            status_code=404, detail="Dataset version not found"
        )

    version = DatasetVersion(**data)
    version.status = status
    storage.set("dataset_versions", version.id, version.model_dump())
    return version
