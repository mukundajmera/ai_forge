"""Metrics Logger Callback - Real-time training metrics logging.

This module provides a callback for logging training metrics
during fine-tuning, with support for various backends.

Example:
    >>> logger = MetricsLogger(log_dir="./logs")
    >>> trainer.add_callback(logger)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricsLoggerConfig:
    """Configuration for MetricsLogger.
    
    Attributes:
        log_dir: Directory to save logs.
        log_to_file: Whether to log to file.
        log_to_console: Whether to log to console.
        log_format: Format for log entries (json, csv, text).
        metrics_to_log: Which metrics to log.
        log_interval: Log every N steps.
    """
    
    log_dir: str = "./logs"
    log_to_file: bool = True
    log_to_console: bool = True
    log_format: Literal["json", "csv", "text"] = "json"
    metrics_to_log: list[str] = field(
        default_factory=lambda: ["loss", "learning_rate", "epoch", "step"]
    )
    log_interval: int = 1  # Log every step by default


@dataclass
class MetricEntry:
    """A single metric log entry.
    
    Attributes:
        timestamp: When the metric was logged.
        step: Training step.
        epoch: Training epoch.
        metrics: Dictionary of metric values.
    """
    
    timestamp: str
    step: int
    epoch: float
    metrics: dict[str, float]
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "timestamp": self.timestamp,
            "step": self.step,
            "epoch": self.epoch,
            **self.metrics,
        })
    
    def to_csv_row(self) -> str:
        """Convert to CSV row."""
        values = [self.timestamp, str(self.step), str(self.epoch)]
        values.extend(str(v) for v in self.metrics.values())
        return ",".join(values)


class MetricsLogger:
    """Callback for logging training metrics.
    
    This callback logs training metrics to file and/or console,
    supporting JSON, CSV, and text formats.
    
    Attributes:
        config: Logger configuration.
        entries: List of logged metric entries.
        
    Example:
        >>> logger = MetricsLogger(MetricsLoggerConfig(log_dir="./logs"))
        >>> logger.on_log({"loss": 0.5, "learning_rate": 1e-4}, step=100, epoch=1.5)
    """
    
    def __init__(self, config: Optional[MetricsLoggerConfig] = None) -> None:
        """Initialize MetricsLogger.
        
        Args:
            config: Logger configuration.
        """
        self.config = config or MetricsLoggerConfig()
        self.entries: list[MetricEntry] = []
        self._log_file: Optional[Path] = None
        self._step_counter = 0
        
        # Setup log directory and file
        if self.config.log_to_file:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self.config.log_format if self.config.log_format != "text" else "log"
            self._log_file = log_dir / f"training_{timestamp}.{ext}"
            
            # Write header for CSV
            if self.config.log_format == "csv":
                header = ["timestamp", "step", "epoch"] + self.config.metrics_to_log
                self._log_file.write_text(",".join(header) + "\n")
    
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
        self._step_counter += 1
        
        # Check log interval
        if self._step_counter % self.config.log_interval != 0:
            return
        
        # Filter metrics
        filtered_metrics = {
            k: float(v) for k, v in metrics.items()
            if k in self.config.metrics_to_log and isinstance(v, (int, float))
        }
        
        # Create entry
        entry = MetricEntry(
            timestamp=datetime.now().isoformat(),
            step=step,
            epoch=epoch,
            metrics=filtered_metrics,
        )
        
        self.entries.append(entry)
        
        # Log to console
        if self.config.log_to_console:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in filtered_metrics.items())
            logger.info(f"Step {step} (epoch {epoch:.2f}): {metrics_str}")
        
        # Log to file
        if self.config.log_to_file and self._log_file:
            self._write_entry(entry)
    
    def _write_entry(self, entry: MetricEntry) -> None:
        """Write entry to log file.
        
        Args:
            entry: Metric entry to write.
        """
        if self._log_file is None:
            return
        
        if self.config.log_format == "json":
            line = entry.to_json()
        elif self.config.log_format == "csv":
            line = entry.to_csv_row()
        else:  # text
            metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in entry.metrics.items())
            line = f"[{entry.timestamp}] Step {entry.step}: {metrics_str}"
        
        with open(self._log_file, "a") as f:
            f.write(line + "\n")
    
    def on_train_begin(self) -> None:
        """Called when training begins."""
        logger.info("Training started - metrics logging enabled")
    
    def on_train_end(self, final_metrics: Optional[dict[str, Any]] = None) -> None:
        """Called when training ends.
        
        Args:
            final_metrics: Optional final metrics to log.
        """
        if final_metrics:
            logger.info(f"Training complete: {final_metrics}")
        
        if self._log_file:
            logger.info(f"Metrics saved to {self._log_file}")
    
    def get_history(self) -> list[dict[str, Any]]:
        """Get all logged metrics as a list of dictionaries.
        
        Returns:
            List of metric dictionaries.
        """
        return [
            {"step": e.step, "epoch": e.epoch, **e.metrics}
            for e in self.entries
        ]
    
    def get_metric_series(self, metric_name: str) -> list[tuple[int, float]]:
        """Get a specific metric as a time series.
        
        Args:
            metric_name: Name of the metric.
            
        Returns:
            List of (step, value) tuples.
        """
        return [
            (e.step, e.metrics[metric_name])
            for e in self.entries
            if metric_name in e.metrics
        ]
