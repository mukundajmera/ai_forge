"""Training Schemas - Pydantic models for training configuration.

This module defines configuration models for the PiSSA + QLoRA training pipeline,
including model settings, training hyperparameters, and callback configurations.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class InitMethod(str, Enum):
    """Adapter initialization method."""
    
    PISSA = "pissa"       # Principal Singular components (SVD-based)
    GAUSSIAN = "gaussian"  # Random Gaussian (standard LoRA)
    ZEROS = "zeros"       # Zero initialization


class QuantType(str, Enum):
    """Quantization type."""
    
    NF4 = "nf4"  # Normal Float 4-bit
    FP4 = "fp4"  # Float 4-bit


class OptimizerType(str, Enum):
    """Optimizer type."""
    
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    SGDM = "sgdm"
    ADAFACTOR = "adafactor"


class SchedulerType(str, Enum):
    """Learning rate scheduler type."""
    
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class DPOLossType(str, Enum):
    """DPO loss function type."""
    
    SIGMOID = "sigmoid"
    HINGE = "hinge"
    IPO = "ipo"


class ModelConfig(BaseModel):
    """Base model configuration.
    
    Attributes:
        base_model: HuggingFace model identifier.
        max_seq_length: Maximum sequence length.
        dtype: Data type for model weights.
        load_in_4bit: Whether to load in 4-bit quantized format.
        trust_remote_code: Whether to trust remote code.
    """
    
    base_model: str = Field(default="unsloth/Llama-3.2-3B-Instruct-4bit")
    max_seq_length: int = Field(default=2048, ge=128, le=131072)
    dtype: str = Field(default="auto")
    load_in_4bit: bool = Field(default=True)
    trust_remote_code: bool = Field(default=True)


class PiSSAConfig(BaseModel):
    """PiSSA adapter configuration.
    
    PiSSA initializes adapter matrices using principal singular components
    from SVD decomposition, providing faster convergence than random init.
    
    Attributes:
        init_method: Initialization method (pissa or gaussian).
        rank: Rank of adapter matrices.
        lora_alpha: Scaling factor (typically 2x rank).
        lora_dropout: Dropout probability.
        target_modules: List of modules to adapt.
        use_rslora: Use RSLoRA scaling.
        pissa_niter: SVD refinement iterations.
    """
    
    init_method: InitMethod = Field(default=InitMethod.PISSA)
    rank: int = Field(default=64, ge=1, le=512)
    lora_alpha: int = Field(default=128, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    use_rslora: bool = Field(default=True)
    pissa_niter: int = Field(default=4, ge=1, le=10)
    
    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor."""
        if self.use_rslora:
            import math
            return self.lora_alpha / math.sqrt(self.rank)
        return self.lora_alpha / self.rank


class QuantizationConfig(BaseModel):
    """Quantization configuration.
    
    QLoRA freezes the base model in 4-bit precision and trains only adapters,
    reducing memory by ~75%.
    
    Attributes:
        bits: Quantization bits (4 or 8).
        quant_type: Quantization format (nf4 or fp4).
        double_quant: Double quantization for further compression.
        compute_dtype: Compute dtype for matrix operations.
    """
    
    bits: int = Field(default=4, ge=1, le=8)
    quant_type: QuantType = Field(default=QuantType.NF4)
    double_quant: bool = Field(default=True)
    compute_dtype: str = Field(default="bfloat16")


class TrainingConfig(BaseModel):
    """Training hyperparameters.
    
    Attributes:
        learning_rate: Learning rate.
        lr_scheduler_type: Scheduler type.
        warmup_steps: Number of warmup steps.
        warmup_ratio: Fraction of steps for warmup.
        weight_decay: L2 regularization.
        max_grad_norm: Gradient clipping threshold.
        optimizer: Optimizer type.
        per_device_train_batch_size: Batch size per device.
        per_device_eval_batch_size: Eval batch size.
        gradient_accumulation_steps: Gradient accumulation.
        num_train_epochs: Number of epochs.
        max_steps: Maximum training steps (-1 for epochs).
        seed: Random seed.
    """
    
    learning_rate: float = Field(default=2e-4, ge=1e-7, le=1.0)
    lr_scheduler_type: SchedulerType = Field(default=SchedulerType.COSINE)
    warmup_steps: int = Field(default=100, ge=0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW_8BIT)
    per_device_train_batch_size: int = Field(default=2, ge=1, le=64)
    per_device_eval_batch_size: int = Field(default=2, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    num_train_epochs: int = Field(default=3, ge=1, le=100)
    max_steps: int = Field(default=-1)
    seed: int = Field(default=42)
    
    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


class EvaluationConfig(BaseModel):
    """Evaluation configuration.
    
    Attributes:
        strategy: Evaluation strategy (steps, epoch, no).
        eval_steps: Evaluate every N steps.
        eval_accumulation_steps: Accumulation for eval predictions.
    """
    
    strategy: str = Field(default="steps")
    eval_steps: int = Field(default=50, ge=1)
    eval_accumulation_steps: int = Field(default=4, ge=1)


class CheckpointConfig(BaseModel):
    """Checkpoint configuration.
    
    Attributes:
        save_strategy: Save strategy.
        save_steps: Save every N steps.
        save_total_limit: Maximum checkpoints to keep.
        load_best_model_at_end: Load best model after training.
        metric_for_best_model: Metric for best model selection.
        greater_is_better: Whether higher metric is better.
        resume_from_checkpoint: Path to resume from.
    """
    
    save_strategy: str = Field(default="steps")
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    load_best_model_at_end: bool = Field(default=True)
    metric_for_best_model: str = Field(default="eval_loss")
    greater_is_better: bool = Field(default=False)
    resume_from_checkpoint: Optional[str] = Field(default=None)


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration.
    
    Attributes:
        enabled: Whether to enable early stopping.
        patience: Evaluations without improvement before stopping.
        min_delta: Minimum improvement threshold.
        metric: Metric to monitor.
    """
    
    enabled: bool = Field(default=True)
    patience: int = Field(default=5, ge=1, le=50)
    min_delta: float = Field(default=0.001, ge=0.0)
    metric: str = Field(default="eval_loss")


class MemoryConfig(BaseModel):
    """Memory management configuration.
    
    Attributes:
        alert_threshold: Memory usage alert threshold.
        gradient_checkpointing: Enable gradient checkpointing.
        cpu_offload: Offload optimizer states to CPU.
        empty_cache_steps: Empty cache every N steps.
        max_memory_fraction: Maximum memory fraction to use.
    """
    
    alert_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    gradient_checkpointing: bool = Field(default=True)
    cpu_offload: bool = Field(default=False)
    empty_cache_steps: int = Field(default=10, ge=1)
    max_memory_fraction: float = Field(default=0.90, ge=0.1, le=1.0)


class LoggingConfig(BaseModel):
    """Logging configuration.
    
    Attributes:
        logging_steps: Log every N steps.
        log_level: Logging level.
        output_dir: Output directory.
        run_name: Run name for tracking.
        report_to: Reporting destination.
    """
    
    logging_steps: int = Field(default=10, ge=1)
    log_level: str = Field(default="info")
    output_dir: str = Field(default="./output")
    run_name: Optional[str] = Field(default=None)
    report_to: str = Field(default="tensorboard")


class DPOConfig(BaseModel):
    """DPO (Direct Preference Optimization) configuration.
    
    Optional second phase after SFT to align model preferences.
    
    Attributes:
        enabled: Whether to enable DPO.
        beta: KL penalty strength.
        learning_rate: DPO learning rate.
        num_epochs: DPO training epochs.
        loss_type: DPO loss function type.
    """
    
    enabled: bool = Field(default=False)
    beta: float = Field(default=0.1, ge=0.01, le=1.0)
    learning_rate: float = Field(default=5e-5, ge=1e-7)
    num_epochs: int = Field(default=1, ge=1)
    loss_type: DPOLossType = Field(default=DPOLossType.SIGMOID)


class HardwareConfig(BaseModel):
    """Hardware configuration.
    
    Attributes:
        device: Device (auto, cuda, mps, cpu).
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        dataloader_num_workers: Number of data loading workers.
    """
    
    device: str = Field(default="auto")
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=True)
    dataloader_num_workers: int = Field(default=0, ge=0)


class FineTuneConfig(BaseModel):
    """Complete fine-tuning configuration.
    
    Aggregates all configuration components into a single config object.
    
    Example:
        >>> config = FineTuneConfig()
        >>> config.pissa.rank = 128
        >>> trainer = FineTuneTrainer(config)
    """
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    pissa: PiSSAConfig = Field(default_factory=PiSSAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "FineTuneConfig":
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def get_training_arguments(self) -> dict[str, Any]:
        """Convert to HuggingFace TrainingArguments format."""
        return {
            "output_dir": self.logging.output_dir,
            "run_name": self.logging.run_name,
            "learning_rate": self.training.learning_rate,
            "lr_scheduler_type": self.training.lr_scheduler_type.value,
            "warmup_steps": self.training.warmup_steps,
            "warmup_ratio": self.training.warmup_ratio,
            "weight_decay": self.training.weight_decay,
            "max_grad_norm": self.training.max_grad_norm,
            "per_device_train_batch_size": self.training.per_device_train_batch_size,
            "per_device_eval_batch_size": self.training.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "num_train_epochs": self.training.num_train_epochs,
            "max_steps": self.training.max_steps,
            "seed": self.training.seed,
            "eval_strategy": self.evaluation.strategy,
            "eval_steps": self.evaluation.eval_steps,
            "save_strategy": self.checkpoints.save_strategy,
            "save_steps": self.checkpoints.save_steps,
            "save_total_limit": self.checkpoints.save_total_limit,
            "load_best_model_at_end": self.checkpoints.load_best_model_at_end,
            "metric_for_best_model": self.checkpoints.metric_for_best_model,
            "greater_is_better": self.checkpoints.greater_is_better,
            "logging_steps": self.logging.logging_steps,
            "report_to": self.logging.report_to,
            "fp16": self.hardware.fp16,
            "bf16": self.hardware.bf16,
            "dataloader_num_workers": self.hardware.dataloader_num_workers,
            "gradient_checkpointing": self.memory.gradient_checkpointing,
        }
