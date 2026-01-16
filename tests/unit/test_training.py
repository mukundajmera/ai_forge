"""Unit tests for training module."""

import pytest
from pathlib import Path

from ai_forge.training.forge import TrainingForge, ForgeConfig, TrainingMetrics
from ai_forge.training.callbacks.early_stopping import EarlyStopping, EarlyStoppingConfig
from ai_forge.training.callbacks.metrics_logger import MetricsLogger, MetricsLoggerConfig


class TestForgeConfig:
    """Tests for ForgeConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ForgeConfig()
        
        assert config.model_name == "unsloth/Llama-3.2-3B-Instruct"
        assert config.use_pissa is True
        assert config.load_in_4bit is True
        assert config.num_epochs == 3
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ForgeConfig(
            model_name="custom/model",
            num_epochs=5,
            pissa_rank=128,
        )
        
        assert config.model_name == "custom/model"
        assert config.num_epochs == 5
        assert config.pissa_rank == 128


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""
    
    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = TrainingMetrics(
            epoch=1.5,
            step=100,
            loss=0.5,
            learning_rate=1e-4,
            eval_loss=0.6,
        )
        
        result = metrics.to_dict()
        
        assert result["epoch"] == 1.5
        assert result["step"] == 100
        assert result["loss"] == 0.5
        assert result["eval_loss"] == 0.6
    
    def test_optional_fields_excluded(self) -> None:
        """Test optional fields are excluded when None."""
        metrics = TrainingMetrics(
            epoch=1.0,
            step=50,
            loss=0.3,
            learning_rate=1e-4,
        )
        
        result = metrics.to_dict()
        
        assert "eval_loss" not in result
        assert "perplexity" not in result


class TestTrainingForge:
    """Tests for TrainingForge class."""
    
    def test_init_creates_output_dir(self, tmp_path: Path) -> None:
        """Test initialization creates output directory."""
        config = ForgeConfig(output_dir=str(tmp_path / "output"))
        forge = TrainingForge(config)
        
        assert Path(config.output_dir).exists()
    
    def test_detect_hardware(self, tmp_path: Path) -> None:
        """Test hardware detection."""
        config = ForgeConfig(output_dir=str(tmp_path / "output"))
        forge = TrainingForge(config)
        
        hardware = forge._detect_hardware()
        
        assert "platform" in hardware
        assert "processor" in hardware
    
    def test_format_data(self, tmp_path: Path) -> None:
        """Test data formatting."""
        config = ForgeConfig(output_dir=str(tmp_path / "output"))
        forge = TrainingForge(config)
        
        example = {
            "instruction": "Explain this",
            "input": "code here",
            "output": "explanation",
        }
        
        result = forge._format_data(example)
        
        assert "text" in result
        assert "[INST]" in result["text"]
        assert "Explain this" in result["text"]


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""
    
    def test_improvement_resets_counter(self) -> None:
        """Test that improvement resets wait counter."""
        early_stop = EarlyStopping(EarlyStoppingConfig(patience=3))
        
        assert not early_stop.check(1.0, step=1)  # First value
        assert early_stop.wait_count == 0
        
        assert not early_stop.check(0.8, step=2)  # Improvement
        assert early_stop.wait_count == 0
        assert early_stop.best_value == 0.8
    
    def test_no_improvement_increments_counter(self) -> None:
        """Test that no improvement increments wait counter."""
        early_stop = EarlyStopping(EarlyStoppingConfig(patience=3))
        
        early_stop.check(1.0, step=1)
        early_stop.check(1.1, step=2)  # Worse
        
        assert early_stop.wait_count == 1
        assert early_stop.best_value == 1.0
    
    def test_stops_after_patience_exceeded(self) -> None:
        """Test stopping after patience exceeded."""
        early_stop = EarlyStopping(EarlyStoppingConfig(patience=2))
        
        early_stop.check(1.0, step=1)
        early_stop.check(1.1, step=2)  # Wait = 1
        should_stop = early_stop.check(1.2, step=3)  # Wait = 2
        
        assert should_stop is True
        assert early_stop.stopped is True
    
    def test_reset_clears_state(self) -> None:
        """Test reset clears all state."""
        early_stop = EarlyStopping()
        
        early_stop.check(1.0, step=1)
        early_stop.check(1.1, step=2)
        early_stop.reset()
        
        assert early_stop.best_value is None
        assert early_stop.wait_count == 0
        assert early_stop.stopped is False


class TestMetricsLogger:
    """Tests for MetricsLogger callback."""
    
    def test_log_entry(self, tmp_path: Path) -> None:
        """Test logging a metric entry."""
        config = MetricsLoggerConfig(
            log_dir=str(tmp_path),
            log_to_console=False,
        )
        logger = MetricsLogger(config)
        
        logger.on_log({"loss": 0.5, "learning_rate": 1e-4}, step=100, epoch=1.5)
        
        assert len(logger.entries) == 1
        assert logger.entries[0].step == 100
    
    def test_get_history(self, tmp_path: Path) -> None:
        """Test getting metric history."""
        config = MetricsLoggerConfig(log_dir=str(tmp_path), log_to_console=False)
        logger = MetricsLogger(config)
        
        logger.on_log({"loss": 0.5}, step=1, epoch=0.1)
        logger.on_log({"loss": 0.3}, step=2, epoch=0.2)
        
        history = logger.get_history()
        
        assert len(history) == 2
        assert history[0]["loss"] == 0.5
        assert history[1]["loss"] == 0.3
    
    def test_log_interval(self, tmp_path: Path) -> None:
        """Test log interval filtering."""
        config = MetricsLoggerConfig(
            log_dir=str(tmp_path),
            log_to_console=False,
            log_interval=2,
        )
        logger = MetricsLogger(config)
        
        logger.on_log({"loss": 0.5}, step=1, epoch=0.1)  # Skipped
        logger.on_log({"loss": 0.4}, step=2, epoch=0.2)  # Logged
        logger.on_log({"loss": 0.3}, step=3, epoch=0.3)  # Skipped
        
        assert len(logger.entries) == 1
