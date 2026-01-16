"""AI Forge - Local LLM Fine-Tuning Service.

A production-grade fine-tuning service for Mac Apple Silicon,
using PiSSA + QLoRA for optimal performance.
"""

__version__ = "1.0.0"
__author__ = "AI Forge Team"

# Re-export main components for convenience
from ai_forge.data_pipeline import CodeMiner, RAFTGenerator, DataValidator
from ai_forge.training import TrainingForge, ForgeConfig
from ai_forge.judge import ModelEvaluator, GGUFExporter
from ai_forge.antigravity_agent import RepoGuardian

__all__ = [
    "CodeMiner",
    "RAFTGenerator", 
    "DataValidator",
    "TrainingForge",
    "ForgeConfig",
    "ModelEvaluator",
    "GGUFExporter",
    "RepoGuardian",
]
