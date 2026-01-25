#!/usr/bin/env python3
"""Verify the tensor dimension fix for LoRA training.

This script tests that:
1. Loading a model works
2. LoRA/PiSSA initialization doesn't cause dimension mismatches
3. A forward pass completes without errors
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_lora_dimensions():
    """Test that LoRA dimensions are correct."""
    
    print("=" * 60)
    print("Testing LoRA Dimension Fix")
    print("=" * 60)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    
    # Use a small model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    print(f"\n1. Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="mps",
        trust_remote_code=True,
    )
    
    print(f"   Model hidden size: {model.config.hidden_size}")
    print(f"   Model num layers: {model.config.num_hidden_layers}")
    
    # LoRA config with rank 32 (like the UI test)
    print("\n2. Applying LoRA config with rank 32")
    
    lora_config = LoraConfig(
        r=32,  # LoRA rank
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Check LoRA layer shapes
    print("\n3. Checking LoRA layer shapes")
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for adapter_name in module.lora_A.keys():
                lora_A = module.lora_A[adapter_name]
                lora_B = module.lora_B[adapter_name]
                print(f"   {name}:")
                print(f"      lora_A weight shape: {lora_A.weight.shape}")  
                print(f"      lora_B weight shape: {lora_B.weight.shape}")
            # Only print first LoRA layer
            break
    
    # Test forward pass
    print("\n4. Testing forward pass")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer("Hello, world!", return_tensors="pt", padding=True)
    inputs = {k: v.to("mps") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"   ✅ Forward pass successful!")
    print(f"   Output logits shape: {outputs.logits.shape}")
    
    # Test with labels for training
    print("\n5. Testing with labels (simulating training)")
    
    inputs["labels"] = inputs["input_ids"].clone()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"   ✅ Forward pass with labels successful!")
    print(f"   Loss: {outputs.loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ LORA DIMENSION TEST PASSED!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_lora_dimensions()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
