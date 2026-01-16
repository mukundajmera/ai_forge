"""Memory Monitor Callback - Apple Silicon unified memory monitoring.

This module provides memory monitoring with OOM prevention for
training on Mac Apple Silicon devices.

Example:
    >>> monitor = MemoryMonitor(threshold_gb=14.0)
    >>> trainer.add_callback(monitor)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryMonitorConfig:
    """Configuration for MemoryMonitor.
    
    Attributes:
        check_interval_steps: Check memory every N steps.
        warning_threshold_percent: Warn when memory exceeds this %.
        critical_threshold_percent: Stop training above this %.
        log_memory_usage: Whether to log memory usage.
        enable_gc_on_warning: Run garbage collection on warning.
    """
    
    check_interval_steps: int = 10
    warning_threshold_percent: float = 80.0
    critical_threshold_percent: float = 95.0
    log_memory_usage: bool = True
    enable_gc_on_warning: bool = True


class MemoryMonitor:
    """Memory monitoring callback for Apple Silicon.
    
    Monitors unified memory usage during training and can
    trigger garbage collection or stop training to prevent
    out-of-memory errors.
    
    Attributes:
        config: Monitor configuration.
        total_memory_gb: Total system memory in GB.
        peak_memory_gb: Peak observed memory usage.
        
    Example:
        >>> monitor = MemoryMonitor(MemoryMonitorConfig(critical_threshold_percent=90))
        >>> if monitor.check_memory(step=100):
        ...     print("Memory critical!")
    """
    
    def __init__(self, config: Optional[MemoryMonitorConfig] = None) -> None:
        """Initialize MemoryMonitor.
        
        Args:
            config: Monitor configuration.
        """
        self.config = config or MemoryMonitorConfig()
        self.total_memory_gb: Optional[float] = None
        self.peak_memory_gb: float = 0.0
        self._step_counter = 0
        self._warning_count = 0
        
        # Get total memory
        self.total_memory_gb = self._get_total_memory()
        if self.total_memory_gb:
            logger.info(f"MemoryMonitor: Total memory {self.total_memory_gb:.1f} GB")
    
    def _get_total_memory(self) -> Optional[float]:
        """Get total system memory.
        
        Returns:
            Total memory in GB or None if unavailable.
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            memory_bytes = int(result.stdout.strip())
            return memory_bytes / (1024**3)
        except Exception as e:
            logger.warning(f"Could not get total memory: {e}")
            return None
    
    def _get_current_memory(self) -> Optional[float]:
        """Get current memory usage.
        
        Returns:
            Current memory usage in GB or None if unavailable.
        """
        try:
            # Use vm_stat for memory info
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
            )
            
            # Parse vm_stat output
            lines = result.stdout.strip().split("\n")
            page_size = 16384  # Default for Apple Silicon
            
            # Parse page size from first line
            if "page size of" in lines[0]:
                page_size = int(lines[0].split("page size of")[1].split()[0])
            
            # Get active and wired pages
            active_pages = 0
            wired_pages = 0
            
            for line in lines[1:]:
                if "Pages active:" in line:
                    active_pages = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages wired down:" in line:
                    wired_pages = int(line.split(":")[1].strip().rstrip("."))
            
            used_bytes = (active_pages + wired_pages) * page_size
            return used_bytes / (1024**3)
            
        except Exception as e:
            logger.debug(f"Could not get current memory: {e}")
            return None
    
    def get_memory_percent(self) -> Optional[float]:
        """Get current memory usage as percentage.
        
        Returns:
            Memory usage percentage or None if unavailable.
        """
        if self.total_memory_gb is None:
            return None
        
        current = self._get_current_memory()
        if current is None:
            return None
        
        return (current / self.total_memory_gb) * 100
    
    def check_memory(self, step: int) -> bool:
        """Check memory status.
        
        Args:
            step: Current training step.
            
        Returns:
            True if training should stop due to critical memory.
        """
        self._step_counter += 1
        
        if self._step_counter % self.config.check_interval_steps != 0:
            return False
        
        current_gb = self._get_current_memory()
        if current_gb is None:
            return False
        
        # Update peak
        if current_gb > self.peak_memory_gb:
            self.peak_memory_gb = current_gb
        
        percent = self.get_memory_percent()
        if percent is None:
            return False
        
        # Log if configured
        if self.config.log_memory_usage:
            logger.debug(f"Step {step}: Memory {current_gb:.1f} GB ({percent:.1f}%)")
        
        # Check warning threshold
        if percent >= self.config.warning_threshold_percent:
            self._warning_count += 1
            logger.warning(
                f"MemoryMonitor: High memory usage {current_gb:.1f} GB ({percent:.1f}%)"
            )
            
            # Trigger garbage collection
            if self.config.enable_gc_on_warning:
                self._run_gc()
        
        # Check critical threshold
        if percent >= self.config.critical_threshold_percent:
            logger.error(
                f"MemoryMonitor: Critical memory usage {current_gb:.1f} GB ({percent:.1f}%). "
                f"Stopping training to prevent OOM."
            )
            return True
        
        return False
    
    def _run_gc(self) -> None:
        """Run garbage collection."""
        import gc
        
        gc.collect()
        
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
        except Exception:
            pass
    
    def on_step_end(self, step: int) -> bool:
        """Called at the end of each training step.
        
        Args:
            step: Current training step.
            
        Returns:
            True if training should stop.
        """
        return self.check_memory(step)
    
    def get_summary(self) -> dict[str, float]:
        """Get memory usage summary.
        
        Returns:
            Dictionary with memory statistics.
        """
        return {
            "total_memory_gb": self.total_memory_gb or 0.0,
            "peak_memory_gb": self.peak_memory_gb,
            "current_memory_gb": self._get_current_memory() or 0.0,
            "warning_count": self._warning_count,
        }
