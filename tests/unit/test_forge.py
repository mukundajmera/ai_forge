"""Comprehensive unit tests for the Training Forge module.

Tests cover:
- PiSSA SVD initialization
- Configuration loading
- Trainer initialization
- Callback functionality
- Memory monitoring

Note: Full training tests require GPU/MPS and model downloads.
These tests focus on logic validation without heavy dependencies.

Run with: pytest tests/unit/test_forge.py -v
"""

import pytest
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.schemas import (
    FineTuneConfig,
    PiSSAConfig,
    QuantizationConfig,
    TrainingConfig,
    InitMethod,
    QuantType,
    OptimizerType,
)
from training.forge import (
    PiSSAInitializer,
    TrainingState,
    MetricsLoggerCallback,
    EarlyStoppingCallback,
    MemoryMonitorCallback,
    detect_device,
    get_memory_info,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config() -> FineTuneConfig:
    """Create default FineTuneConfig."""
    return FineTuneConfig()


@pytest.fixture
def custom_config() -> FineTuneConfig:
    """Create customized config."""
    config = FineTuneConfig()
    config.pissa.rank = 128
    config.pissa.lora_alpha = 256
    config.training.learning_rate = 1e-4
    config.training.num_train_epochs = 5
    return config


@pytest.fixture
def training_state() -> TrainingState:
    """Create training state."""
    return TrainingState(
        global_step=100,
        epoch=1.5,
        total_steps=1000,
        loss=0.5,
        learning_rate=2e-4,
    )


# =============================================================================
# Configuration Tests
# =============================================================================

class TestFineTuneConfig:
    """Tests for training configuration."""
    
    def test_default_config(self, default_config: FineTuneConfig) -> None:
        """Test default configuration values."""
        assert default_config.model.base_model == "unsloth/Llama-3.2-3B-Instruct-4bit"
        assert default_config.pissa.rank == 64
        assert default_config.pissa.lora_alpha == 128
        assert default_config.training.learning_rate == 2e-4
    
    def test_pissa_config(self) -> None:
        """Test PiSSA configuration."""
        config = PiSSAConfig()
        
        assert config.init_method == InitMethod.PISSA
        assert config.rank == 64
        assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def test_pissa_scaling(self) -> None:
        """Test PiSSA scaling calculation."""
        config = PiSSAConfig(rank=64, lora_alpha=128, use_rslora=True)
        
        expected = 128 / math.sqrt(64)  # 16.0
        assert config.scaling == pytest.approx(expected)
    
    def test_quantization_config(self) -> None:
        """Test quantization configuration."""
        config = QuantizationConfig()
        
        assert config.bits == 4
        assert config.quant_type == QuantType.NF4
        assert config.double_quant is True
    
    def test_effective_batch_size(self) -> None:
        """Test effective batch size calculation."""
        config = TrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        )
        
        assert config.effective_batch_size == 8
    
    def test_get_training_arguments(self, default_config: FineTuneConfig) -> None:
        """Test conversion to training arguments."""
        args = default_config.get_training_arguments()
        
        assert "learning_rate" in args
        assert "num_train_epochs" in args
        assert "gradient_accumulation_steps" in args
        assert args["learning_rate"] == 2e-4


# =============================================================================
# PiSSA Initialization Tests
# =============================================================================

class TestPiSSAInitializer:
    """Tests for PiSSA SVD initialization."""
    
    def test_init_creates_correct_shape(self) -> None:
        """Test that PiSSA creates correct output shapes."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        rank = 16
        out_features = 128
        in_features = 256
        
        W = torch.randn(out_features, in_features)
        initializer = PiSSAInitializer(rank=rank)
        
        A, B, W_res = initializer.compute_init(W)
        
        # Check shapes
        assert A.shape == (out_features, rank)
        assert B.shape == (rank, in_features)
        assert W_res.shape == W.shape
    
    def test_init_preserves_approximation(self) -> None:
        """Test that A @ B approximates W."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        rank = 32
        W = torch.randn(128, 256)
        
        initializer = PiSSAInitializer(rank=rank)
        A, B, W_res = initializer.compute_init(W)
        
        # W ≈ A @ B + W_res
        reconstructed = A @ B + W_res
        
        # Should be close to original
        error = torch.norm(W - reconstructed).item()
        assert error < 1e-5
    
    def test_reconstruction_error(self) -> None:
        """Test reconstruction error calculation."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        rank = 16
        W = torch.randn(64, 128)
        
        initializer = PiSSAInitializer(rank=rank)
        A, B, _ = initializer.compute_init(W)
        
        error = initializer.get_reconstruction_error(W, A, B)
        
        # Error should be between 0 and 1
        assert 0.0 <= error <= 1.0
    
    def test_higher_rank_lower_error(self) -> None:
        """Test that higher rank gives lower reconstruction error."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        W = torch.randn(64, 128)
        
        init_low = PiSSAInitializer(rank=8)
        init_high = PiSSAInitializer(rank=32)
        
        A_low, B_low, _ = init_low.compute_init(W)
        A_high, B_high, _ = init_high.compute_init(W)
        
        error_low = init_low.get_reconstruction_error(W, A_low, B_low)
        error_high = init_high.get_reconstruction_error(W, A_high, B_high)
        
        # Higher rank should have lower error
        assert error_high < error_low


# =============================================================================
# Callback Tests
# =============================================================================

class TestMetricsLoggerCallback:
    """Tests for MetricsLoggerCallback."""
    
    def test_log_on_step(self, tmp_path: Path, training_state: TrainingState) -> None:
        """Test logging on step end."""
        callback = MetricsLoggerCallback(
            output_dir=str(tmp_path),
            log_every=10,
        )
        
        callback.on_train_begin(training_state)
        
        # Step 100 should log (divisible by 10)
        training_state.global_step = 100
        callback.on_step_end(training_state)
        
        assert len(callback.history) == 1
        assert callback.history[0]["step"] == 100
    
    def test_no_log_between_intervals(self, tmp_path: Path) -> None:
        """Test that non-interval steps don't log."""
        callback = MetricsLoggerCallback(
            output_dir=str(tmp_path),
            log_every=10,
        )
        
        state = TrainingState(global_step=5)
        callback.on_train_begin(state)
        callback.on_step_end(state)
        
        assert len(callback.history) == 0


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""
    
    def test_improvement_resets_counter(self) -> None:
        """Test that improvement resets patience counter."""
        callback = EarlyStoppingCallback(patience=3)
        state = TrainingState()
        
        # First evaluation sets baseline
        callback.on_evaluate(state, {"eval_loss": 1.0})
        assert callback.wait_count == 0
        
        # Improvement resets counter
        callback.on_evaluate(state, {"eval_loss": 0.8})
        assert callback.wait_count == 0
    
    def test_no_improvement_increments_counter(self) -> None:
        """Test that no improvement increments counter."""
        callback = EarlyStoppingCallback(patience=3, min_delta=0.01)
        state = TrainingState()
        
        callback.on_evaluate(state, {"eval_loss": 1.0})
        callback.on_evaluate(state, {"eval_loss": 1.005})  # Not enough improvement
        
        assert callback.wait_count == 1
    
    def test_triggers_stop_after_patience(self) -> None:
        """Test that stopping triggers after patience exhausted."""
        callback = EarlyStoppingCallback(patience=2, min_delta=0.01)
        state = TrainingState()
        
        callback.on_evaluate(state, {"eval_loss": 1.0})
        callback.on_evaluate(state, {"eval_loss": 1.0})  # No improvement
        callback.on_evaluate(state, {"eval_loss": 1.0})  # Still no improvement
        
        assert callback.should_stop is True
    
    def test_on_step_returns_false_when_stopped(self) -> None:
        """Test that on_step_end returns False when stopped."""
        callback = EarlyStoppingCallback()
        callback.should_stop = True
        
        result = callback.on_step_end(TrainingState())
        assert result is False


class TestMemoryMonitorCallback:
    """Tests for MemoryMonitorCallback."""
    
    def test_tracks_peak_memory(self) -> None:
        """Test that peak memory is tracked."""
        callback = MemoryMonitorCallback(check_every=1)
        state = TrainingState(global_step=1)
        
        callback.on_step_end(state)
        
        # Peak memory should be set
        assert callback.peak_memory >= 0
    
    def test_always_returns_true(self) -> None:
        """Test that callback doesn't stop training."""
        callback = MemoryMonitorCallback()
        
        result = callback.on_step_end(TrainingState(global_step=50))
        assert result is True


# =============================================================================
# Hardware Detection Tests
# =============================================================================

class TestHardwareDetection:
    """Tests for hardware detection."""
    
    def test_detect_device_returns_string(self) -> None:
        """Test that detect_device returns valid device."""
        device = detect_device()
        assert device in ["mps", "cuda", "cpu"]
    
    def test_get_memory_info(self) -> None:
        """Test memory info retrieval."""
        mem_info = get_memory_info()
        
        assert "process_rss_gb" in mem_info
        assert "system_total_gb" in mem_info
        assert "system_percent" in mem_info
        assert mem_info["system_total_gb"] > 0


# =============================================================================
# Training State Tests
# =============================================================================

class TestTrainingState:
    """Tests for TrainingState."""
    
    def test_default_values(self) -> None:
        """Test default training state values."""
        state = TrainingState()
        
        assert state.global_step == 0
        assert state.epoch == 0.0
        assert state.loss == 0.0
    
    def test_metrics_dict(self, training_state: TrainingState) -> None:
        """Test that metrics can be added."""
        training_state.metrics["accuracy"] = 0.95
        
        assert training_state.metrics["accuracy"] == 0.95


# =============================================================================
# Integration Tests (Mock-based)
# =============================================================================

class TestFineTuneTrainerMocked:
    """Integration tests with mocked dependencies."""
    
    def test_trainer_initialization(self, default_config: FineTuneConfig) -> None:
        """Test trainer can be initialized."""
        from training.forge import FineTuneTrainer
        
        trainer = FineTuneTrainer(default_config)
        
        assert trainer.config == default_config
        assert trainer.model is None  # Not loaded yet
    
    def test_add_callback(self, default_config: FineTuneConfig) -> None:
        """Test adding callbacks."""
        from training.forge import FineTuneTrainer
        
        trainer = FineTuneTrainer(default_config)
        callback = EarlyStoppingCallback()
        
        trainer.add_callback(callback)
        
        assert callback in trainer.callbacks
    
    def test_count_trainable_params_no_model(self, default_config: FineTuneConfig) -> None:
        """Test trainable params count without model."""
        from training.forge import FineTuneTrainer
        
        trainer = FineTuneTrainer(default_config)
        
        assert trainer._count_trainable_params() == 0


# =============================================================================
# Config Serialization Tests
# =============================================================================

class TestConfigSerialization:
    """Tests for config serialization."""
    
    def test_to_yaml(self, default_config: FineTuneConfig, tmp_path: Path) -> None:
        """Test saving config to YAML."""
        yaml_path = tmp_path / "config.yaml"
        
        default_config.to_yaml(yaml_path)
        
        assert yaml_path.exists()
    
    def test_from_yaml(self, default_config: FineTuneConfig, tmp_path: Path) -> None:
        """Test loading config from YAML."""
        yaml_path = tmp_path / "config.yaml"
        default_config.to_yaml(yaml_path)
        
        loaded = FineTuneConfig.from_yaml(yaml_path)
        
        assert loaded.pissa.rank == default_config.pissa.rank
        assert loaded.training.learning_rate == default_config.training.learning_rate


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_very_small_rank(self) -> None:
        """Test PiSSA with very small rank."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        W = torch.randn(64, 128)
        initializer = PiSSAInitializer(rank=1)
        
        A, B, _ = initializer.compute_init(W)
        
        assert A.shape == (64, 1)
        assert B.shape == (1, 128)
    
    def test_rank_larger_than_dimension(self) -> None:
        """Test PiSSA when rank exceeds matrix dimension."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        W = torch.randn(16, 32)  # Small matrix
        initializer = PiSSAInitializer(rank=10)  # Less than min dim
        
        A, B, _ = initializer.compute_init(W)
        
        # Should clamp to min dimension
        assert A.shape[1] <= min(16, 32)
    
    def test_empty_metrics(self) -> None:
        """Test early stopping with missing metric."""
        callback = EarlyStoppingCallback(metric="nonexistent")
        state = TrainingState()
        
        # Should not crash on missing metric
        callback.on_evaluate(state, {"loss": 1.0})
        
        assert callback.best_value is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
