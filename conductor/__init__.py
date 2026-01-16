"""AI Forge Conductor Module.

This module provides the FastAPI service, Ollama integration,
and job queue management for the fine-tuning service.
"""

from ai_forge.conductor.service import app, create_app
from ai_forge.conductor.ollama_manager import OllamaManager
from ai_forge.conductor.job_queue import JobQueue, Job

__all__ = ["app", "create_app", "OllamaManager", "JobQueue", "Job"]
