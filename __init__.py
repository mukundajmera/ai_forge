"""AI Forge - Local LLM Fine-Tuning Service.

A production-grade fine-tuning service for Mac Apple Silicon,
using PiSSA + QLoRA for optimal performance.
"""

__version__ = "1.0.0"
__author__ = "AI Forge Team"

# NOTE: These imports are temporarily disabled due to module export mismatches.
# The submodule __init__.py files need to be updated to export these classes.
# For now, import directly from the submodules:
#   from data_pipeline.miner import parse_repository
#   from data_pipeline.raft_generator import RAFTGenerator
#   from training.forge import TrainingForge
# 
# from ai_forge.data_pipeline import CodeMiner, RAFTGenerator, DataValidator
# from ai_forge.training import TrainingForge, ForgeConfig
# from ai_forge.judge import ModelEvaluator, GGUFExporter
# from ai_forge.antigravity_agent import RepoGuardian

__all__ = [
    "__version__",
    "__author__",
]

