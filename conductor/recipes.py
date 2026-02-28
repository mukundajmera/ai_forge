"""Recipes API Router - Endpoints for training recipe management.

This module provides REST API endpoints for:
- Listing built-in and custom training recipes
- Getting recipe details with hardware-aware defaults
- Creating custom recipes
- Deleting custom recipes

Example:
    >>> from conductor.recipes import router as recipes_router
    >>> app.include_router(recipes_router, prefix="/api")
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from conductor.models import (
    BUILTIN_RECIPES,
    HardwareProfile,
    Recipe,
    RecipeDefaults,
    TaskType,
)
from conductor.persistence import storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Recipes"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateRecipeRequest(BaseModel):
    """Request to create a custom recipe."""

    name: str
    description: str = ""
    task_type: TaskType
    supported_models: list[str] = Field(default_factory=list)
    defaults: RecipeDefaults = Field(default_factory=RecipeDefaults)
    tags: list[str] = Field(default_factory=list)


class RecipeWithDefaults(BaseModel):
    """Recipe with hardware-adjusted defaults."""

    recipe: Recipe
    adjusted_defaults: RecipeDefaults


# =============================================================================
# Recipe Endpoints
# =============================================================================


@router.get("/recipes")
async def list_recipes(
    task_type: Optional[str] = None,
    tag: Optional[str] = None,
) -> list[Recipe]:
    """List all available recipes (built-in + custom)."""
    # Start with built-in recipes
    recipes = list(BUILTIN_RECIPES)

    # Add custom recipes from storage
    custom = storage.get_all("recipes")
    recipes.extend(Recipe(**v) for v in custom.values())

    if task_type:
        recipes = [r for r in recipes if r.task_type.value == task_type]
    if tag:
        recipes = [r for r in recipes if tag in r.tags]

    return recipes


@router.get("/recipes/{recipe_id}")
async def get_recipe(
    recipe_id: str,
    hardware: Optional[HardwareProfile] = None,
) -> RecipeWithDefaults:
    """Get a recipe with optional hardware-adjusted defaults."""
    # Check built-in recipes first
    recipe = next((r for r in BUILTIN_RECIPES if r.id == recipe_id), None)

    # Check custom recipes
    if not recipe:
        data = storage.get("recipes", recipe_id)
        if data:
            recipe = Recipe(**data)

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    # Apply hardware overrides if specified
    profile = hardware or HardwareProfile.MEDIUM
    adjusted = recipe.get_defaults_for_hardware(profile)

    return RecipeWithDefaults(recipe=recipe, adjusted_defaults=adjusted)


@router.post("/recipes")
async def create_recipe(request: CreateRecipeRequest) -> Recipe:
    """Create a custom training recipe."""
    recipe = Recipe(
        id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        task_type=request.task_type,
        supported_models=request.supported_models,
        defaults=request.defaults,
        tags=request.tags,
        is_builtin=False,
    )

    storage.set("recipes", recipe.id, recipe.model_dump())
    logger.info(f"Created custom recipe: {recipe.id} ({recipe.name})")
    return recipe


@router.delete("/recipes/{recipe_id}")
async def delete_recipe(recipe_id: str) -> dict[str, str]:
    """Delete a custom recipe (built-in recipes cannot be deleted)."""
    # Check if it's a built-in recipe
    if any(r.id == recipe_id for r in BUILTIN_RECIPES):
        raise HTTPException(
            status_code=400, detail="Cannot delete built-in recipes"
        )

    data = storage.get("recipes", recipe_id)
    if not data:
        raise HTTPException(status_code=404, detail="Recipe not found")

    storage.delete("recipes", recipe_id)
    logger.info(f"Deleted custom recipe: {recipe_id}")
    return {"status": "deleted", "id": recipe_id}
