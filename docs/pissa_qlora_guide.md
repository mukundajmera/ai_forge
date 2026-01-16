# PiSSA + QLoRA Technical Guide

This document explains the core algorithms and design decisions for the AI Forge fine-tuning engine.

## PiSSA (Principal Singular components Initialization)

### Mathematical Foundation

PiSSA represents a theoretical and practical improvement over standard LoRA initialization. Instead of random Gaussian initialization, PiSSA initializes adapter matrices using **principal singular components** from SVD decomposition.

#### Standard LoRA Initialization
```
A ~ N(0, σ²)  (random Gaussian)
B = 0         (zero matrix)
```

#### PiSSA Initialization
Given weight matrix W ∈ ℝ^(m×n), compute SVD:

```
W = U @ S @ V^T
```

Where:
- U ∈ ℝ^(m×m): Left singular vectors
- S ∈ ℝ^(min(m,n)): Singular values (diagonal)
- V ∈ ℝ^(n×n): Right singular vectors

Then initialize adapters with rank r:

```
A = U[:, :r] @ sqrt(diag(S[:r]))   ∈ ℝ^(m×r)
B = sqrt(diag(S[:r])) @ V^T[:r, :] ∈ ℝ^(r×n)
```

The residual is stored in the base model:
```
W_residual = W - A @ B
```

### Why This Works

1. **Captures Principal Components**: The top-r singular values capture the most important directions in the weight space
2. **Balanced Initialization**: Using sqrt(S) distributes the singular values evenly between A and B
3. **Faster Convergence**: Training starts closer to the optimal solution (3-5x faster)
4. **Higher Accuracy**: +5.16% on code and reasoning benchmarks

### Implementation

```python
class PiSSAInitializer:
    def compute_init(self, weight: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        W = weight.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        sqrt_S = torch.sqrt(S[:self.rank])
        A = U[:, :self.rank] * sqrt_S.unsqueeze(0)
        B = sqrt_S.unsqueeze(1) * Vh[:self.rank, :]
        
        W_residual = W - A @ B
        return A, B, W_residual
```

---

## QLoRA (Quantized Low-Rank Adaptation)

### Quantization Theory

QLoRA freezes the base model in 4-bit precision while training only the adapter weights in full precision.

#### NF4 (Normal Float 4-bit)
The NF4 format is specifically designed for normally-distributed neural network weights:

```
Quantization levels: {-1.0, -0.6962, -0.5251, ..., 0.5251, 0.6962, 1.0}
```

These levels are optimized to minimize quantization error for Gaussian-distributed values.

#### Block-wise Quantization
Weights are quantized in blocks of 64 parameters:
- Each block has its own scale factor (fp16)
- Optional double quantization: quantize the quantization constants

### Memory Reduction

| Configuration | Bytes/Param | Reduction |
|---------------|-------------|-----------|
| Float32 | 4.0 | 0% |
| Float16 | 2.0 | 50% |
| NF4 (basic) | 0.5 | 87.5% |
| NF4 + scale | ~0.53 | ~87% |
| NF4 + double quant | ~0.56 | **86%** |

In practice, **~75% reduction** is achieved when accounting for all overhead.

---

## Apple Silicon Optimizations

### MPS Backend
- Uses Metal Performance Shaders for GPU acceleration
- Unified memory architecture eliminates CPU-GPU transfers

### Recommended Settings
```yaml
hardware:
  device: "mps"
  bf16: true              # Native bfloat16 support
  dataloader_num_workers: 0  # No multiprocessing overhead

memory:
  gradient_checkpointing: true
  alert_threshold: 0.80
```

### Memory Budget (16GB Unified Memory)
| Model Size | Max Batch | Memory Usage |
|------------|-----------|--------------|
| 3B params | 2 | ~8-10 GB |
| 7B params | 1 | ~12-14 GB |

---

## Training Configuration Reference

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 64 | Adapter rank (PiSSA supports higher values) |
| `lora_alpha` | 128 | Scaling factor (typically 2× rank) |
| `learning_rate` | 2e-4 | Higher than standard LoRA due to better init |
| `gradient_accumulation_steps` | 4 | Effective batch = batch_size × this |

### Why These Defaults

1. **Rank 64**: PiSSA's principal component initialization allows stable training at higher ranks than random LoRA (typically 16-32)

2. **Alpha 2× Rank**: Standard scaling ratio for LoRA adapters

3. **Learning Rate 2e-4**: PiSSA's better initialization allows higher learning rates without instability

4. **Target Modules**: Attention projections capture most task-specific knowledge
   - q_proj, k_proj, v_proj, o_proj

---

## DPO (Direct Preference Optimization)

Optional second phase after SFT to reduce hallucinations:

```yaml
dpo:
  enabled: true
  beta: 0.1             # KL penalty strength
  learning_rate: 5e-5   # Lower than SFT
  loss_type: "sigmoid"
```

DPO trains the model to prefer correct answers over incorrect ones using preference pairs, significantly reducing hallucinations in the final model.
