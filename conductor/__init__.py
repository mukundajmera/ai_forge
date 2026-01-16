"""AI Forge Conductor Module.

This module provides the FastAPI service, Ollama integration,
and job queue management for the fine-tuning service.
"""

from conductor.service import app, create_app
from conductor.ollama_manager import OllamaManager
from conductor.job_queue import JobQueue, Job

__all__ = ["app", "create_app", "OllamaManager", "JobQueue", "Job"]

