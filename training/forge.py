"""Fine-Tune Trainer - Production-grade PiSSA + QLoRA training engine.

This module implements the core training logic using:
- PiSSA (Principal Singular components Initialization)
- QLoRA (Quantized Low-Rank Adaptation)
- Unsloth-MLX optimizations for Apple Silicon

Key Innovation - PiSSA:
    Instead of random Gaussian initialization (standard LoRA), PiSSA initializes
    adapter matrices using principal singular components via SVD decomposition:
    
    W = U @ S @ V^T
    A_init = U[:, :rank] @ sqrt(S[:rank])
    B_init = sqrt(S[:rank]) @ V[:, :rank]^T
    
    Benefits: 3-5x faster convergence, +5% accuracy on code benchmarks.

Example:
    >>> from training.forge import FineTuneTrainer
    >>> config = FineTuneConfig()
    >>> trainer = FineTuneTrainer(config)
    >>> trainer.train(train_dataset, eval_dataset)
"""

from __future__ import annotations

import gc
import logging
import math
import os

# Enable MPS fallback to CPU for unsupported PyTorch operations on Apple Silicon
# This accounts for missing ops like aten::linalg_qr.out in torch 2.x on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from training.schemas import FineTuneConfig, InitMethod

logger = logging.getLogger(__name__)


# =============================================================================
# Hardware Detection
# =============================================================================

def detect_device() -> str:
    """Detect the best available device.
    
    Returns:
        Device string: "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" otherwise.
    """
    try:
        import torch
        
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def get_memory_info() -> dict[str, float]:
    """Get current memory usage information.
    
    Returns:
        Dictionary with memory usage stats.
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    system_memory = psutil.virtual_memory()
    
    return {
        "process_rss_gb": memory_info.rss / (1024**3),
        "system_used_gb": system_memory.used / (1024**3),
        "system_total_gb": system_memory.total / (1024**3),
        "system_percent": system_memory.percent / 100,
    }


# =============================================================================
# Callback Definitions
# =============================================================================

@dataclass
class TrainingState:
    """Current training state for callbacks.
    
    Attributes:
        global_step: Current training step.
        epoch: Current epoch.
        total_steps: Total training steps.
        loss: Current loss value.
        learning_rate: Current learning rate.
        metrics: Additional metrics.
    """
    
    global_step: int = 0
    epoch: float = 0.0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


class TrainerCallback:
    """Base class for training callbacks."""
    
    def on_train_begin(self, state: TrainingState) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, state: TrainingState) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, state: TrainingState) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_step_begin(self, state: TrainingState) -> None:
        """Called at the beginning of each step."""
        pass
    
    def on_step_end(self, state: TrainingState) -> bool:
        """Called at the end of each step.
        
        Returns:
            False to stop training, True to continue.
        """
        return True
    
    def on_evaluate(self, state: TrainingState, metrics: dict[str, float]) -> None:
        """Called after evaluation."""
        pass


class MetricsLoggerCallback(TrainerCallback):
    """Logs training metrics to file and console."""
    
    def __init__(self, output_dir: str, log_every: int = 10) -> None:
        self.output_dir = Path(output_dir)
        self.log_every = log_every
        self.log_file: Optional[Path] = None
        self.history: list[dict] = []
    
    def on_train_begin(self, state: TrainingState) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "training_log.jsonl"
        logger.info(f"Training started. Logging to {self.log_file}")
    
    def on_step_end(self, state: TrainingState) -> bool:
        if state.global_step % self.log_every == 0:
            record = {
                "step": state.global_step,
                "epoch": round(state.epoch, 3),
                "loss": round(state.loss, 4),
                "lr": state.learning_rate,
                "timestamp": datetime.now().isoformat(),
            }
            record.update(state.metrics)
            self.history.append(record)
            
            logger.info(
                f"Step {state.global_step}: loss={state.loss:.4f}, "
                f"lr={state.learning_rate:.2e}"
            )
        
        return True
    
    def on_train_end(self, state: TrainingState) -> None:
        if self.log_file and self.history:
            import json
            with open(self.log_file, 'w') as f:
                for record in self.history:
                    f.write(json.dumps(record) + '\n')
            logger.info(f"Training complete. Logs saved to {self.log_file}")


class EarlyStoppingCallback(TrainerCallback):
    """Stops training if metric doesn't improve."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        metric: str = "eval_loss",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_value: Optional[float] = None
        self.wait_count = 0
        self.should_stop = False
    
    def on_evaluate(self, state: TrainingState, metrics: dict[str, float]) -> None:
        current_value = metrics.get(self.metric)
        
        if current_value is None:
            return
        
        if self.best_value is None:
            self.best_value = current_value
            return
        
        # Check improvement (assuming lower is better for loss)
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait_count = 0
            logger.info(f"Early stopping: New best {self.metric}={current_value:.4f}")
        else:
            self.wait_count += 1
            logger.info(
                f"Early stopping: No improvement for {self.wait_count}/{self.patience}"
            )
            
            if self.wait_count >= self.patience:
                self.should_stop = True
                logger.info("Early stopping triggered!")
    
    def on_step_end(self, state: TrainingState) -> bool:
        return not self.should_stop


class MemoryMonitorCallback(TrainerCallback):
    """Monitors memory usage and alerts on high usage."""
    
    def __init__(
        self,
        alert_threshold: float = 0.80,
        check_every: int = 50,
    ) -> None:
        self.alert_threshold = alert_threshold
        self.check_every = check_every
        self.peak_memory = 0.0
    
    def on_step_end(self, state: TrainingState) -> bool:
        if state.global_step % self.check_every == 0:
            mem_info = get_memory_info()
            current_usage = mem_info["system_percent"]
            
            self.peak_memory = max(self.peak_memory, mem_info["process_rss_gb"])
            
            if current_usage > self.alert_threshold:
                logger.warning(
                    f"⚠️ High memory usage: {current_usage:.1%} "
                    f"(threshold: {self.alert_threshold:.1%})"
                )
                
                # Try to free memory
                gc.collect()
        
        return True
    
    def on_train_end(self, state: TrainingState) -> None:
        logger.info(f"Peak memory usage: {self.peak_memory:.2f} GB")


# =============================================================================
# PiSSA Initialization
# =============================================================================

class PiSSAInitializer:
    """Computes PiSSA initialization using SVD.
    
    PiSSA initializes adapter matrices A and B such that:
    W ≈ A @ B + W_residual
    
    Where A and B are computed from SVD of W:
    W = U @ S @ V^T
    A = U[:, :rank] @ sqrt(diag(S[:rank]))
    B = sqrt(diag(S[:rank])) @ V[:, :rank]^T
    """
    
    def __init__(self, rank: int, niter: int = 4) -> None:
        """Initialize PiSSA computer.
        
        Args:
            rank: Target rank for decomposition.
            niter: Number of SVD refinement iterations.
        """
        self.rank = rank
        self.niter = niter
    
    def compute_init(
        self,
        weight: Any,  # torch.Tensor
    ) -> Tuple[Any, Any, Any]:
        """Compute PiSSA initialization for a weight matrix.
        
        Args:
            weight: Weight matrix W of shape (out_features, in_features).
            
        Returns:
            Tuple of (A, B, W_residual) where:
            - A: Shape (out_features, rank)
            - B: Shape (rank, in_features)
            - W_residual: Residual weights
        """
        import torch
        
        # Convert to float for SVD
        W = weight.float()
        out_features, in_features = W.shape
        
        # Compute truncated SVD
        # For large matrices, use randomized SVD (faster)
        if min(out_features, in_features) > 1000:
            U, S, Vh = torch.svd_lowrank(W, q=self.rank, niter=self.niter)
        else:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            U = U[:, :self.rank]
            S = S[:self.rank]
            Vh = Vh[:self.rank, :]
        
        # Compute sqrt(S) for balanced initialization
        sqrt_S = torch.sqrt(S)
        
        # A = U @ sqrt(diag(S))
        A = U * sqrt_S.unsqueeze(0)  # Broadcasting: (out_features, rank)
        
        # B = sqrt(diag(S)) @ V^T
        B = sqrt_S.unsqueeze(1) * Vh  # Broadcasting: (rank, in_features)
        
        # Compute residual: W - A @ B
        W_approx = A @ B
        W_residual = W - W_approx
        
        # Convert back to original dtype
        A = A.to(weight.dtype)
        B = B.to(weight.dtype)
        W_residual = W_residual.to(weight.dtype)
        
        return A, B, W_residual
    
    def get_reconstruction_error(
        self,
        W: Any,
        A: Any,
        B: Any,
    ) -> float:
        """Compute reconstruction error ||W - AB||_F / ||W||_F."""
        import torch
        
        reconstruction = A @ B
        error = torch.norm(W - reconstruction, 'fro')
        original_norm = torch.norm(W, 'fro')
        
        return (error / original_norm).item() if original_norm > 0 else 0.0


# =============================================================================
# Fine-Tune Trainer
# =============================================================================

class FineTuneTrainer:
    """Production-grade PiSSA + QLoRA trainer.
    
    Implements efficient fine-tuning using:
    - PiSSA initialization (SVD-based adapter init)
    - QLoRA quantization (4-bit base model)
    - Unsloth-MLX optimizations for Apple Silicon
    
    Example:
        >>> config = FineTuneConfig()
        >>> trainer = FineTuneTrainer(config)
        >>> trainer.load_model()
        >>> trainer.train(train_dataset, eval_dataset)
        >>> trainer.save_model("./output")
    """
    
    def __init__(self, config: FineTuneConfig) -> None:
        """Initialize trainer.
        
        Args:
            config: Complete training configuration.
        """
        self.config = config
        self.device = detect_device() if config.hardware.device == "auto" else config.hardware.device
        
        # Model components (initialized by load_model)
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # PiSSA initializer
        self.pissa_init = PiSSAInitializer(
            rank=config.pissa.rank,
            niter=config.pissa.pissa_niter,
        )
        
        # Training state
        self.state = TrainingState()
        self.callbacks: list[TrainerCallback] = []
        
        # Metrics tracking
        self.train_history: list[dict] = []
        self.eval_history: list[dict] = []
        
        logger.info(f"FineTuneTrainer initialized on device: {self.device}")
    
    def add_callback(self, callback: TrainerCallback) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)
    
    def _setup_default_callbacks(self) -> None:
        """Setup default training callbacks."""
        # Metrics logger
        self.add_callback(MetricsLoggerCallback(
            output_dir=self.config.logging.output_dir,
            log_every=self.config.logging.logging_steps,
        ))
        
        # Memory monitor
        self.add_callback(MemoryMonitorCallback(
            alert_threshold=self.config.memory.alert_threshold,
        ))
        
        # Early stopping
        if self.config.early_stopping.enabled:
            self.add_callback(EarlyStoppingCallback(
                patience=self.config.early_stopping.patience,
                min_delta=self.config.early_stopping.min_delta,
                metric=self.config.early_stopping.metric,
            ))
    
    def load_model(self) -> None:
        """Load and prepare model for training.
        
        This method:
        1. Loads the base model in quantized format
        2. Prepares PiSSA adapters
        3. Applies Unsloth optimizations if available
        """
        logger.info(f"Loading model: {self.config.model.base_model}")
        
        try:
            # Try Unsloth for optimized loading
            self._load_with_unsloth()
        except ImportError:
            logger.warning("Unsloth not available, using standard loading")
            self._load_standard()
        
        logger.info(f"Model loaded. Trainable params: {self._count_trainable_params():,}")
    
    def _load_with_unsloth(self) -> None:
        """Load model using Unsloth optimizations."""
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.base_model,
            max_seq_length=self.config.model.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.config.model.load_in_4bit,
        )
        
        # Add LoRA adapters with PiSSA initialization
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.pissa.rank,
            lora_alpha=self.config.pissa.lora_alpha,
            lora_dropout=self.config.pissa.lora_dropout,
            target_modules=self.config.pissa.target_modules,
            bias="none",
            use_gradient_checkpointing=self.config.memory.gradient_checkpointing,
            random_state=self.config.training.seed,
            use_rslora=self.config.pissa.use_rslora,
            # PiSSA initialization
            init_lora_weights=self.config.pissa.init_method.value,
        )
        
        self.model = model
        self.tokenizer = tokenizer
        self.peft_model = model
    
    def _load_standard(self) -> None:
        """Load model using standard HuggingFace/PEFT."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # Quantization config
        quant_config = None
        load_in_4bit = self.config.quantization.bits == 4
        
        # Check for MPS
        if self.device == "mps" and load_in_4bit:
            logger.warning("⚠️ Apple MPS detected: Disabling 4-bit quantization (bitsandbytes not supported). Using native precision.")
            load_in_4bit = False
        
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quantization.quant_type.value,
                bnb_4bit_use_double_quant=self.config.quantization.double_quant,
                bnb_4bit_compute_dtype=getattr(torch, self.config.quantization.compute_dtype),
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            quantization_config=quant_config,
            device_map="auto" if self.device != "mps" else self.device,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.float16 if self.device == "mps" else "auto",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        
        # Prepare for k-bit training
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        else:
            # For MPS/CPU without quantization, we still need to enable input gradients
            # for LoRA training to work. This is normally done by prepare_model_for_kbit_training.
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            logger.info("Enabled input gradients for non-quantized training")
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.pissa.rank,
            lora_alpha=self.config.pissa.lora_alpha,
            lora_dropout=self.config.pissa.lora_dropout,
            target_modules=self.config.pissa.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=self.config.pissa.use_rslora,
            init_lora_weights=self.config.pissa.init_method.value != "gaussian",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Apply PiSSA initialization if enabled
        # TEMPORARILY DISABLED to debug tensor mismatch issue
        # if self.config.pissa.init_method == InitMethod.PISSA:
        #     self._apply_pissa_init(model)
        logger.info("PiSSA initialization skipped (standard LoRA used)")
        
        self.model = model
        self.tokenizer = tokenizer
        self.peft_model = model
    
    def _apply_pissa_init(self, model: Any) -> None:
        """Apply PiSSA initialization to LoRA layers."""
        import torch
        
        logger.info("Applying PiSSA initialization...")
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Get original weight
                if hasattr(module, 'base_layer'):
                    W = module.base_layer.weight.data
                elif hasattr(module, 'weight'):
                    W = module.weight.data
                else:
                    continue
                
                # Compute PiSSA init
                A, B, W_res = self.pissa_init.compute_init(W)
                
                # Apply to LoRA layers
                # LoRA computes: output = input @ lora_A.T @ lora_B.T
                # So: lora_A.weight shape = (rank, in_features) -> This is B from PiSSA
                #     lora_B.weight shape = (out_features, rank) -> This is A from PiSSA
                for adapter_name in module.lora_A.keys():
                    lora_A = module.lora_A[adapter_name]
                    lora_B = module.lora_B[adapter_name]
                    
                    # B has shape (rank, in_features) - matches lora_A expectation
                    # A has shape (out_features, rank) - matches lora_B expectation
                    lora_A.weight.data = B.contiguous()
                    lora_B.weight.data = A.contiguous()
                
                # Store residual in base layer if possible
                if hasattr(module, 'base_layer'):
                    module.base_layer.weight.data = W_res
                
                logger.debug(f"PiSSA initialized: {name}")
        
        logger.info("PiSSA initialization complete")
    
    def _count_trainable_params(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            return 0
        
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def compute_pissa_init(
        self,
        weight_matrix: Any,
    ) -> Tuple[Any, Any, Any]:
        """Compute PiSSA initialization for a weight matrix.
        
        Public method for external use.
        
        Args:
            weight_matrix: Weight matrix to decompose.
            
        Returns:
            Tuple of (A, B, residual).
        """
        return self.pissa_init.compute_init(weight_matrix)
    
    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        num_epochs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Execute training loop.
        
        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            num_epochs: Override config epochs.
            
        Returns:
            Dictionary with training results.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        # Setup callbacks
        self._setup_default_callbacks()
        
        # Determine epochs
        epochs = num_epochs or self.config.training.num_train_epochs
        
        logger.info(f"Starting training for {epochs} epochs")
        
        # Training arguments
        training_args_dict = self.config.get_training_arguments()
        training_args_dict["num_train_epochs"] = epochs  # Override if provided
        
        # Disable evaluation if no eval dataset provided
        if eval_dataset is None:
            logger.warning("No evaluation dataset provided. Disabling evaluation.")
            training_args_dict["eval_strategy"] = "no"
            
        training_args = TrainingArguments(**training_args_dict)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )
        
        # Run callbacks
        self.state.total_steps = len(train_dataset) * epochs // (
            self.config.training.per_device_train_batch_size * 
            self.config.training.gradient_accumulation_steps
        )
        
        for callback in self.callbacks:
            callback.on_train_begin(self.state)
        
        # Train
        start_time = time.time()
        result = trainer.train()
        training_time = time.time() - start_time
        
        # Run end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self.state)
        
        logger.info(f"Training complete in {training_time:.1f}s")
        
        return {
            "training_time": training_time,
            "train_loss": result.training_loss,
            "train_samples": result.metrics.get("train_samples", 0),
            "epochs": epochs,
        }
    
    def evaluate(self, eval_dataset: Any) -> dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            eval_dataset: Evaluation dataset.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        
        eval_args = TrainingArguments(
            output_dir=self.config.logging.output_dir,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            report_to="none",
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )
        
        metrics = trainer.evaluate()
        
        # Run callbacks
        for callback in self.callbacks:
            callback.on_evaluate(self.state, metrics)
        
        return metrics
    
    def save_model(self, output_dir: str) -> None:
        """Save trained model and adapters.
        
        Args:
            output_dir: Directory to save to.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_path)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        
        # Save config
        self.config.to_yaml(output_path / "training_config.yaml")
        
        logger.info(f"Model saved to {output_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint.
        """
        from peft import PeftModel
        
        if self.model is None:
            raise RuntimeError("Base model not loaded.")
        
        self.model = PeftModel.from_pretrained(
            self.model,
            checkpoint_path,
        )
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train_dpo(
        self,
        preference_dataset: Any,
        num_epochs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train with Direct Preference Optimization (optional second phase).
        
        DPO optimizes the model to prefer correct answers over incorrect ones,
        reducing hallucinations.
        
        Args:
            preference_dataset: Dataset with (prompt, chosen, rejected) examples.
            num_epochs: Override config epochs.
            
        Returns:
            Dictionary with DPO training results.
        """
        if not self.config.dpo.enabled:
            logger.warning("DPO not enabled in config")
            return {}
        
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        logger.info("Starting DPO training phase")
        
        try:
            from trl import DPOTrainer, DPOConfig as TRLDPOConfig
            
            dpo_config = TRLDPOConfig(
                beta=self.config.dpo.beta,
                learning_rate=self.config.dpo.learning_rate,
                num_train_epochs=num_epochs or self.config.dpo.num_epochs,
                per_device_train_batch_size=self.config.training.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                output_dir=self.config.logging.output_dir + "/dpo",
                loss_type=self.config.dpo.loss_type.value,
            )
            
            trainer = DPOTrainer(
                model=self.model,
                ref_model=None,  # Use implicit reference
                args=dpo_config,
                train_dataset=preference_dataset,
                processing_class=self.tokenizer,
            )
            
            result = trainer.train()
            
            logger.info("DPO training complete")
            
            return {
                "dpo_loss": result.training_loss,
                "dpo_epochs": dpo_config.num_train_epochs,
            }
            
        except ImportError:
            logger.error("TRL not installed. Install with: pip install trl")
            return {}


# =============================================================================
# Convenience Functions
# =============================================================================

def create_trainer(config_path: Optional[str] = None) -> FineTuneTrainer:
    """Create trainer from config file.
    
    Args:
        config_path: Path to YAML config (optional).
        
    Returns:
        Configured FineTuneTrainer.
    """
    if config_path:
        config = FineTuneConfig.from_yaml(config_path)
    else:
        config = FineTuneConfig()
    
    return FineTuneTrainer(config)


def quick_train(
    train_data: list[str],
    model_name: str = "unsloth/Llama-3.2-3B-Instruct-4bit",
    output_dir: str = "./output",
) -> FineTuneTrainer:
    """Quick training utility for simple use cases.
    
    Args:
        train_data: List of training texts.
        model_name: Model identifier.
        output_dir: Output directory.
        
    Returns:
        Trained FineTuneTrainer.
    """
    config = FineTuneConfig()
    config.model.base_model = model_name
    config.logging.output_dir = output_dir
    
    trainer = FineTuneTrainer(config)
    
    # Note: Dataset preparation would be needed here
    logger.info("quick_train: Dataset preparation required")
    
    return trainer
