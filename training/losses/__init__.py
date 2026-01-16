"""Training losses for AI Forge.

This module provides custom loss functions for fine-tuning,
including DPO (Direct Preference Optimization) and RAFT losses.
"""

from ai_forge.training.losses.dpo_loss import DPOLoss
from ai_forge.training.losses.raft_loss import RAFTLoss

__all__ = ["DPOLoss", "RAFTLoss"]
