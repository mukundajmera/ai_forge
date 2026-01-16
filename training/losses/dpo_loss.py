"""DPO Loss - Direct Preference Optimization.

This module implements the DPO objective for training models
to prefer better responses over worse ones.

Reference: https://arxiv.org/abs/2305.18290

Example:
    >>> dpo_loss = DPOLoss(beta=0.1)
    >>> loss = dpo_loss(
    ...     policy_chosen_logps=chosen_logps,
    ...     policy_rejected_logps=rejected_logps,
    ...     reference_chosen_logps=ref_chosen_logps,
    ...     reference_rejected_logps=ref_rejected_logps,
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO loss.
    
    Attributes:
        beta: Temperature parameter controlling deviation from reference.
        label_smoothing: Label smoothing factor.
        loss_type: Type of DPO loss (sigmoid, hinge, ipo).
        reference_free: Whether to use reference-free DPO.
    """
    
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    reference_free: bool = False


class DPOLoss:
    """Direct Preference Optimization loss function.
    
    DPO directly optimizes language models to align with human preferences
    without needing a separate reward model. It uses the implicit reward
    from the policy and reference model log-probabilities.
    
    The loss encourages the policy to increase the probability of
    chosen responses relative to rejected ones, while staying close
    to the reference model.
    
    Attributes:
        config: DPO configuration.
        
    Example:
        >>> dpo = DPOLoss(DPOConfig(beta=0.1))
        >>> loss, metrics = dpo.compute_loss(
        ...     policy_chosen_logps, policy_rejected_logps,
        ...     ref_chosen_logps, ref_rejected_logps
        ... )
    """
    
    def __init__(self, config: Optional[DPOConfig] = None) -> None:
        """Initialize DPOLoss.
        
        Args:
            config: DPO configuration.
        """
        self.config = config or DPOConfig()
    
    def compute_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"] = None,
        reference_rejected_logps: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", dict[str, float]]:
        """Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log-probs of chosen responses under policy.
            policy_rejected_logps: Log-probs of rejected responses under policy.
            reference_chosen_logps: Log-probs of chosen under reference model.
            reference_rejected_logps: Log-probs of rejected under reference model.
            
        Returns:
            Tuple of (loss tensor, metrics dictionary).
        """
        import torch
        import torch.nn.functional as F
        
        # Compute log ratios
        if self.config.reference_free:
            # Reference-free: just use policy log-probs
            chosen_rewards = policy_chosen_logps
            rejected_rewards = policy_rejected_logps
        else:
            # Standard DPO: compute implicit rewards
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError(
                    "Reference log-probs required for non-reference-free DPO"
                )
            
            chosen_rewards = self.config.beta * (
                policy_chosen_logps - reference_chosen_logps
            )
            rejected_rewards = self.config.beta * (
                policy_rejected_logps - reference_rejected_logps
            )
        
        # Compute loss based on type
        logits = chosen_rewards - rejected_rewards
        
        if self.config.loss_type == "sigmoid":
            # Standard DPO: cross-entropy loss
            labels = torch.ones_like(logits)
            if self.config.label_smoothing > 0:
                labels = labels * (1 - self.config.label_smoothing) + 0.5 * self.config.label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
        elif self.config.loss_type == "hinge":
            # Hinge loss variant
            loss = torch.relu(1 - logits).mean()
            
        elif self.config.loss_type == "ipo":
            # IPO (Identity Preference Optimization)
            loss = ((logits - 1 / (2 * self.config.beta)) ** 2).mean()
            
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Compute metrics
        with torch.no_grad():
            rewards_chosen = chosen_rewards.mean().item()
            rewards_rejected = rejected_rewards.mean().item()
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        
        metrics = {
            "rewards_chosen": rewards_chosen,
            "rewards_rejected": rewards_rejected,
            "reward_accuracy": reward_accuracy,
            "reward_margin": reward_margin,
        }
        
        return loss, metrics
    
    def __call__(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"] = None,
        reference_rejected_logps: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute DPO loss (callable interface).
        
        Args:
            policy_chosen_logps: Log-probs of chosen responses.
            policy_rejected_logps: Log-probs of rejected responses.
            reference_chosen_logps: Reference log-probs for chosen.
            reference_rejected_logps: Reference log-probs for rejected.
            
        Returns:
            Loss tensor.
        """
        loss, _ = self.compute_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        return loss
    
    def get_batch_log_probs(
        self,
        model: "torch.nn.Module",
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute log-probabilities for a batch.
        
        Args:
            model: Language model.
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Target labels (-100 for ignored tokens).
            
        Returns:
            Per-sequence log-probabilities.
        """
        import torch
        import torch.nn.functional as F
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits[:, :-1, :]  # Exclude last position
        labels = labels[:, 1:]  # Shift labels
        
        # Compute per-token log-probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log-probs for target tokens
        selected_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)
        
        # Mask padding tokens (label = -100)
        mask = labels != -100
        selected_log_probs = selected_log_probs * mask
        
        # Sum log-probs per sequence
        return selected_log_probs.sum(dim=-1)
