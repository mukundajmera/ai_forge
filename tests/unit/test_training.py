"""Unit tests for training module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_forge.training.forge import FineTuneTrainer, TrainingState, TrainerCallback
from ai_forge.training.schemas import FineTuneConfig, InitMethod


class TestFineTuneConfig:
    """Tests for FineTuneConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FineTuneConfig()
        
        # Check defaults from schemas.py
        assert config.model.base_model == "unsloth/Llama-3.2-3B-Instruct-4bit"
        assert config.pissa.init_method == InitMethod.PISSA
        assert config.quantization.bits == 4
        assert config.training.num_train_epochs == 3
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = FineTuneConfig()
        config.model.base_model = "custom/model"
        config.training.num_train_epochs = 5
        config.pissa.rank = 128
        
        assert config.model.base_model == "custom/model"
        assert config.training.num_train_epochs == 5
        assert config.pissa.rank == 128


class TestFineTuneTrainer:
    """Tests for FineTuneTrainer class."""
    
    def test_init(self, tmp_path: Path) -> None:
        """Test initialization."""
        config = FineTuneConfig()
        config.logging.output_dir = str(tmp_path / "output")
        
        trainer = FineTuneTrainer(config)
        
        assert trainer.config.logging.output_dir == str(tmp_path / "output")
        assert trainer.state.global_step == 0
    
    @patch("ai_forge.training.forge.detect_device")
    def test_device_auto_detection(self, mock_detect, tmp_path: Path):
        """Test hardware detection."""
        mock_detect.return_value = "cpu"
        
        config = FineTuneConfig()
        config.hardware.device = "auto"
        
        trainer = FineTuneTrainer(config)
        
        assert trainer.device == "cpu"
        mock_detect.assert_called_once()
    
    def test_callback_registration(self, tmp_path: Path):
        """Test that callbacks can be registered and called."""
        config = FineTuneConfig()
        trainer = FineTuneTrainer(config)
        
        mock_callback = MagicMock(spec=TrainerCallback)
        trainer.add_callback(mock_callback)
        
        # Manually trigger a callback point to verify
        state = TrainingState()
        for cb in trainer.callbacks:
            cb.on_train_begin(state)
            
        mock_callback.on_train_begin.assert_called_once_with(state)


class TestTrainingState:
    """Tests for TrainingState."""
    
    def test_default_state(self):
        state = TrainingState()
        assert state.global_step == 0
        assert state.epoch == 0.0
