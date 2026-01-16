"""Loss Plotter Callback - Real-time loss curve visualization.

This module provides a callback for generating real-time loss curves
during training, with support for saving plots as artifacts.

Example:
    >>> plotter = LossPlotter(output_dir="./plots")
    >>> trainer.add_callback(plotter)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class LossPlotterConfig:
    """Configuration for LossPlotter.
    
    Attributes:
        output_dir: Directory to save plots.
        plot_interval: Save plot every N steps.
        figsize: Figure size (width, height) in inches.
        dpi: Resolution of saved plots.
        style: Plot style (dark, light).
        metrics: Which metrics to plot.
        show_lr: Whether to show learning rate on secondary axis.
        smooth_window: Window size for loss smoothing (0 = no smoothing).
    """
    
    output_dir: str = "./plots"
    plot_interval: int = 50
    figsize: tuple[float, float] = (12, 6)
    dpi: int = 100
    style: Literal["dark", "light"] = "dark"
    metrics: list[str] = field(
        default_factory=lambda: ["train_loss", "eval_loss"]
    )
    show_lr: bool = True
    smooth_window: int = 5


@dataclass
class LossEntry:
    """A single loss log entry.
    
    Attributes:
        step: Training step.
        epoch: Training epoch.
        metrics: Dictionary of metric values.
    """
    
    step: int
    epoch: float
    metrics: dict[str, float]


class LossPlotter:
    """Callback for generating real-time loss curves.
    
    Creates and updates loss visualization plots during training,
    saving them as PNG files for use in artifacts and monitoring.
    
    Attributes:
        config: Plotter configuration.
        history: List of logged loss entries.
        
    Example:
        >>> plotter = LossPlotter(LossPlotterConfig(output_dir="./plots"))
        >>> plotter.on_log({"train_loss": 0.5, "eval_loss": 0.6}, step=100, epoch=1.5)
        >>> plotter.save_plot("loss_curve.png")
    """
    
    def __init__(self, config: Optional[LossPlotterConfig] = None) -> None:
        """Initialize LossPlotter.
        
        Args:
            config: Plotter configuration.
        """
        self.config = config or LossPlotterConfig()
        self.history: list[LossEntry] = []
        self._step_counter = 0
        self._last_plot_step = 0
        
        # Setup output directory
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib for headless operation
        self._setup_matplotlib()
    
    def _setup_matplotlib(self) -> None:
        """Configure matplotlib backend and style."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Headless backend
            import matplotlib.pyplot as plt
            
            # Set style based on config
            if self.config.style == "dark":
                plt.style.use('dark_background')
            else:
                plt.style.use('default')
            
            self._plt = plt
            self._matplotlib_available = True
            logger.debug("matplotlib configured successfully")
        except ImportError:
            self._matplotlib_available = False
            logger.warning("matplotlib not available - loss plotting disabled")
    
    def on_log(
        self,
        metrics: dict[str, Any],
        step: int,
        epoch: float,
    ) -> None:
        """Called when metrics are logged.
        
        Args:
            metrics: Dictionary of metric values.
            step: Current training step.
            epoch: Current training epoch.
        """
        # Filter to numeric metrics we care about
        filtered_metrics = {}
        for key in list(self.config.metrics) + (["learning_rate"] if self.config.show_lr else []):
            if key in metrics and isinstance(metrics[key], (int, float)):
                filtered_metrics[key] = float(metrics[key])
        
        if not filtered_metrics:
            return
        
        # Store entry
        entry = LossEntry(step=step, epoch=epoch, metrics=filtered_metrics)
        self.history.append(entry)
        self._step_counter = step
        
        # Check if we should save a plot
        if step - self._last_plot_step >= self.config.plot_interval:
            self.save_plot()
            self._last_plot_step = step
    
    def _smooth_values(self, values: list[float]) -> list[float]:
        """Apply moving average smoothing.
        
        Args:
            values: Raw metric values.
            
        Returns:
            Smoothed values.
        """
        if self.config.smooth_window <= 1 or len(values) < self.config.smooth_window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - self.config.smooth_window + 1)
            window = values[start:i + 1]
            smoothed.append(sum(window) / len(window))
        
        return smoothed
    
    def save_plot(self, filename: Optional[str] = None) -> Optional[Path]:
        """Save current loss curve as PNG.
        
        Args:
            filename: Optional custom filename.
            
        Returns:
            Path to saved plot or None if plotting unavailable.
        """
        if not self._matplotlib_available:
            return None
        
        if not self.history:
            logger.debug("No history to plot")
            return None
        
        plt = self._plt
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Extract data
        steps = [e.step for e in self.history]
        
        # Plot each loss metric
        colors = plt.cm.viridis([0.2, 0.5, 0.8, 0.95])
        color_idx = 0
        
        for metric in self.config.metrics:
            values = [e.metrics.get(metric) for e in self.history]
            
            # Filter out None values
            valid_steps = [s for s, v in zip(steps, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if not valid_values:
                continue
            
            # Apply smoothing
            smoothed = self._smooth_values(valid_values)
            
            # Plot raw values with low alpha
            ax1.plot(
                valid_steps, valid_values,
                alpha=0.3,
                color=colors[color_idx % len(colors)],
                linewidth=0.5,
            )
            
            # Plot smoothed values
            ax1.plot(
                valid_steps, smoothed,
                label=metric.replace("_", " ").title(),
                color=colors[color_idx % len(colors)],
                linewidth=2,
            )
            
            color_idx += 1
        
        ax1.set_xlabel("Step", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        
        # Plot learning rate on secondary axis if available
        if self.config.show_lr:
            lr_values = [e.metrics.get("learning_rate") for e in self.history]
            valid_lr_steps = [s for s, v in zip(steps, lr_values) if v is not None]
            valid_lr_values = [v for v in lr_values if v is not None]
            
            if valid_lr_values:
                ax2 = ax1.twinx()
                ax2.plot(
                    valid_lr_steps, valid_lr_values,
                    color='red',
                    alpha=0.6,
                    linewidth=1.5,
                    linestyle='--',
                    label='Learning Rate',
                )
                ax2.set_ylabel("Learning Rate", color='red', fontsize=12)
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.legend(loc="upper center")
        
        # Title with step info
        current_step = self.history[-1].step if self.history else 0
        current_epoch = self.history[-1].epoch if self.history else 0
        plt.title(
            f"Training Loss Curve (Step: {current_step}, Epoch: {current_epoch:.2f})",
            fontsize=14,
            fontweight='bold',
        )
        
        plt.tight_layout()
        
        # Save
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loss_curve_{timestamp}.png"
        
        output_path = self._output_dir / filename
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        logger.debug(f"Saved loss plot to {output_path}")
        return output_path
    
    def on_train_begin(self) -> None:
        """Called when training begins."""
        logger.info("LossPlotter: Training started - loss visualization enabled")
        self.history.clear()
        self._step_counter = 0
        self._last_plot_step = 0
    
    def on_train_end(self) -> None:
        """Called when training ends."""
        # Save final plot
        final_path = self.save_plot("loss_curve_final.png")
        if final_path:
            logger.info(f"LossPlotter: Final loss curve saved to {final_path}")
    
    def get_history(self) -> list[dict[str, Any]]:
        """Get all logged loss values as a list of dictionaries.
        
        Returns:
            List of loss dictionaries.
        """
        return [
            {"step": e.step, "epoch": e.epoch, **e.metrics}
            for e in self.history
        ]
    
    def get_loss_series(self, metric_name: str) -> list[tuple[int, float]]:
        """Get a specific loss metric as a time series.
        
        Args:
            metric_name: Name of the metric (e.g., "train_loss").
            
        Returns:
            List of (step, value) tuples.
        """
        return [
            (e.step, e.metrics[metric_name])
            for e in self.history
            if metric_name in e.metrics
        ]
    
    def get_latest_loss(self) -> Optional[float]:
        """Get the most recent training loss value.
        
        Returns:
            Latest train_loss or None if not available.
        """
        if not self.history:
            return None
        
        return self.history[-1].metrics.get("train_loss")
