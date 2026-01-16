"""RAFT Loss - Retrieval-Augmented Fine-Tuning Loss.

This module implements a specialized loss function for RAFT training,
combining retrieval awareness with generation objectives.

Reference: https://arxiv.org/abs/2403.10131

Example:
    >>> raft_loss = RAFTLoss(alpha=0.5)
    >>> loss = raft_loss(logits, labels, retrieval_scores)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class RAFTLossConfig:
    """Configuration for RAFT loss.
    
    Attributes:
        alpha: Weight for retrieval-aware component (0-1).
        temperature: Temperature for retrieval score softmax.
        use_chain_of_thought: Whether to weight CoT tokens higher.
        cot_weight: Weight multiplier for chain-of-thought tokens.
        ignore_index: Label index to ignore in loss.
    """
    
    alpha: float = 0.5  # Balance between standard LM loss and retrieval-aware loss
    temperature: float = 1.0
    use_chain_of_thought: bool = True
    cot_weight: float = 1.5  # Weight CoT reasoning higher
    ignore_index: int = -100


class RAFTLoss:
    """RAFT-specific loss combining generation and retrieval objectives.
    
    RAFT (Retrieval-Augmented Fine-Tuning) trains models to use retrieved
    documents effectively. This loss function:
    
    1. Standard language modeling loss for generation quality
    2. Retrieval-aware weighting based on document relevance
    3. Optional chain-of-thought emphasis
    
    The combined objective teaches the model to extract relevant
    information from context while generating accurate responses.
    
    Attributes:
        config: RAFT loss configuration.
        
    Example:
        >>> config = RAFTLossConfig(alpha=0.5, use_chain_of_thought=True)
        >>> raft_loss = RAFTLoss(config)
        >>> loss, metrics = raft_loss.compute_loss(
        ...     logits, labels,
        ...     oracle_mask=oracle_mask,
        ...     cot_mask=cot_mask
        ... )
    """
    
    def __init__(self, config: Optional[RAFTLossConfig] = None) -> None:
        """Initialize RAFTLoss.
        
        Args:
            config: RAFT loss configuration.
        """
        self.config = config or RAFTLossConfig()
    
    def compute_loss(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        oracle_mask: Optional["torch.Tensor"] = None,
        cot_mask: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", dict[str, float]]:
        """Compute RAFT loss.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab].
            labels: Target labels [batch, seq_len].
            oracle_mask: Mask for tokens from oracle document [batch, seq_len].
            cot_mask: Mask for chain-of-thought tokens [batch, seq_len].
            
        Returns:
            Tuple of (loss tensor, metrics dictionary).
        """
        import torch
        import torch.nn.functional as F
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute standard cross-entropy loss
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        # Per-token loss
        per_token_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.config.ignore_index,
            reduction="none",
        )
        per_token_loss = per_token_loss.view(shift_labels.size())
        
        # Create weight tensor
        weights = torch.ones_like(per_token_loss)
        
        # Apply oracle mask weighting if provided
        if oracle_mask is not None:
            # Shift oracle mask to align with labels
            shift_oracle_mask = oracle_mask[..., 1:]
            # Reduce loss on oracle tokens (model should learn from them)
            weights = weights * (1 + self.config.alpha * shift_oracle_mask)
        
        # Apply chain-of-thought weighting if configured
        if self.config.use_chain_of_thought and cot_mask is not None:
            # Shift CoT mask to align with labels
            shift_cot_mask = cot_mask[..., 1:]
            # Increase weight on CoT tokens
            weights = weights * (1 + (self.config.cot_weight - 1) * shift_cot_mask)
        
        # Compute weighted loss
        weighted_loss = per_token_loss * weights
        
        # Mask for valid tokens (not ignored)
        valid_mask = shift_labels != self.config.ignore_index
        
        # Mean loss over valid tokens
        loss = weighted_loss.sum() / valid_mask.sum().clamp(min=1)
        
        # Compute metrics
        with torch.no_grad():
            standard_loss = per_token_loss[valid_mask].mean().item()
            
            metrics = {
                "raft_loss": loss.item(),
                "standard_lm_loss": standard_loss,
            }
            
            if oracle_mask is not None:
                shift_oracle = oracle_mask[..., 1:]
                oracle_loss = per_token_loss[shift_oracle.bool() & valid_mask].mean()
                metrics["oracle_loss"] = oracle_loss.item() if not oracle_loss.isnan() else 0.0
            
            if cot_mask is not None:
                shift_cot = cot_mask[..., 1:]
                cot_loss = per_token_loss[shift_cot.bool() & valid_mask].mean()
                metrics["cot_loss"] = cot_loss.item() if not cot_loss.isnan() else 0.0
        
        return loss, metrics
    
    def __call__(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        oracle_mask: Optional["torch.Tensor"] = None,
        cot_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute RAFT loss (callable interface).
        
        Args:
            logits: Model output logits.
            labels: Target labels.
            oracle_mask: Optional oracle document mask.
            cot_mask: Optional chain-of-thought mask.
            
        Returns:
            Loss tensor.
        """
        loss, _ = self.compute_loss(logits, labels, oracle_mask, cot_mask)
        return loss
    
    def create_masks(
        self,
        input_ids: "torch.Tensor",
        oracle_start_token: int,
        oracle_end_token: int,
        cot_start_token: Optional[int] = None,
        cot_end_token: Optional[int] = None,
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Create oracle and CoT masks from input IDs.
        
        This helper creates masks based on special tokens that
        delimit oracle documents and chain-of-thought sections.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            oracle_start_token: Token ID marking oracle start.
            oracle_end_token: Token ID marking oracle end.
            cot_start_token: Optional token ID marking CoT start.
            cot_end_token: Optional token ID marking CoT end.
            
        Returns:
            Tuple of (oracle_mask, cot_mask) tensors.
        """
        import torch
        
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create oracle mask
        oracle_mask = torch.zeros(batch_size, seq_len, device=device)
        
        for b in range(batch_size):
            in_oracle = False
            for i, token in enumerate(input_ids[b]):
                if token == oracle_start_token:
                    in_oracle = True
                elif token == oracle_end_token:
                    in_oracle = False
                elif in_oracle:
                    oracle_mask[b, i] = 1.0
        
        # Create CoT mask if tokens provided
        cot_mask = None
        if cot_start_token is not None and cot_end_token is not None:
            cot_mask = torch.zeros(batch_size, seq_len, device=device)
            
            for b in range(batch_size):
                in_cot = False
                for i, token in enumerate(input_ids[b]):
                    if token == cot_start_token:
                        in_cot = True
                    elif token == cot_end_token:
                        in_cot = False
                    elif in_cot:
                        cot_mask[b, i] = 1.0
        
        return oracle_mask, cot_mask
