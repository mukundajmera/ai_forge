"""Early Stopping Callback - Stop training when validation loss plateaus.

This module provides early stopping functionality to prevent
overfitting and save training time.

Example:
    >>> early_stop = EarlyStopping(patience=3, min_delta=0.01)
    >>> trainer.add_callback(early_stop)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for EarlyStopping.
    
    Attributes:
        patience: Number of checks with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        metric: Metric to monitor (loss, eval_loss, etc.).
        mode: Whether lower or higher is better.
        baseline: Baseline value for the metric.
        restore_best_weights: Whether to restore best weights on stop.
    """
    
    patience: int = 3
    min_delta: float = 0.0
    metric: str = "eval_loss"
    mode: str = "min"  # "min" or "max"
    baseline: Optional[float] = None
    restore_best_weights: bool = True


class EarlyStopping:
    """Early stopping callback to halt training on plateau.
    
    Monitors a metric and stops training if it doesn't improve
    for a specified number of evaluation rounds.
    
    Attributes:
        config: Early stopping configuration.
        best_value: Best observed value of the monitored metric.
        best_step: Step at which best value was observed.
        wait_count: Number of checks without improvement.
        stopped: Whether training was stopped early.
        
    Example:
        >>> early_stop = EarlyStopping(EarlyStoppingConfig(patience=5))
        >>> # During training loop:
        >>> if early_stop.check(eval_loss, step=100):
        ...     print("Stopping early!")
        ...     break
    """
    
    def __init__(self, config: Optional[EarlyStoppingConfig] = None) -> None:
        """Initialize EarlyStopping.
        
        Args:
            config: Early stopping configuration.
        """
        self.config = config or EarlyStoppingConfig()
        self.best_value: Optional[float] = self.config.baseline
        self.best_step: int = 0
        self.wait_count: int = 0
        self.stopped: bool = False
        self._best_weights: Optional[dict] = None
        
        # Set comparison function based on mode
        if self.config.mode == "min":
            self._is_better = lambda current, best: current < best - self.config.min_delta
        else:
            self._is_better = lambda current, best: current > best + self.config.min_delta
    
    def check(
        self,
        current_value: float,
        step: int,
        model: Optional[object] = None,
    ) -> bool:
        """Check if training should stop.
        
        Args:
            current_value: Current value of monitored metric.
            step: Current training step.
            model: Optional model to save best weights from.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.stopped:
            return True
        
        # First check or improvement
        if self.best_value is None or self._is_better(current_value, self.best_value):
            self.best_value = current_value
            self.best_step = step
            self.wait_count = 0
            
            # Save best weights
            if self.config.restore_best_weights and model is not None:
                self._save_best_weights(model)
            
            logger.info(
                f"EarlyStopping: Improved {self.config.metric} to {current_value:.4f} at step {step}"
            )
            return False
        
        # No improvement
        self.wait_count += 1
        logger.info(
            f"EarlyStopping: No improvement for {self.wait_count}/{self.config.patience} checks"
        )
        
        if self.wait_count >= self.config.patience:
            self.stopped = True
            logger.warning(
                f"EarlyStopping: Stopping at step {step}. "
                f"Best {self.config.metric}: {self.best_value:.4f} at step {self.best_step}"
            )
            return True
        
        return False
    
    def _save_best_weights(self, model: object) -> None:
        """Save model weights.
        
        Args:
            model: Model to save weights from.
        """
        try:
            import copy
            self._best_weights = copy.deepcopy(model.state_dict())
            logger.debug("Saved best model weights")
        except Exception as e:
            logger.warning(f"Could not save model weights: {e}")
    
    def restore_best_weights(self, model: object) -> bool:
        """Restore best weights to model.
        
        Args:
            model: Model to restore weights to.
            
        Returns:
            True if weights were restored, False otherwise.
        """
        if self._best_weights is None:
            logger.warning("No best weights to restore")
            return False
        
        try:
            model.load_state_dict(self._best_weights)
            logger.info(f"Restored best weights from step {self.best_step}")
            return True
        except Exception as e:
            logger.error(f"Could not restore weights: {e}")
            return False
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_value = self.config.baseline
        self.best_step = 0
        self.wait_count = 0
        self.stopped = False
        self._best_weights = None
    
    def on_evaluate(
        self,
        metrics: dict[str, float],
        step: int,
        model: Optional[object] = None,
    ) -> bool:
        """Called after evaluation.
        
        Args:
            metrics: Evaluation metrics dictionary.
            step: Current training step.
            model: Optional model for weight saving.
            
        Returns:
            True if training should stop.
        """
        if self.config.metric not in metrics:
            logger.warning(f"Metric '{self.config.metric}' not in evaluation metrics")
            return False
        
        return self.check(metrics[self.config.metric], step, model)
