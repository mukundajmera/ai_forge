import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())

from training.forge import FineTuneTrainer
from training.schemas import FineTuneConfig, ModelConfig

def test_inference():
    print("Testing inference with trained model...")
    model_path = "./output/dryrun"
    
    if not Path(model_path).exists():
        print("FAIL: Model path not found")
        sys.exit(1)
        
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading from {model_path}...")
        # Load directly
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        print("Function generated...")
        outputs = model.generate(**inputs, max_new_tokens=20)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Result: {result}")
        
        if len(result) > len(input_text):
            print("PASS: Inference successful")
        else:
            print("FAIL: Model did not generate new tokens?")
            sys.exit(1)
            
    except Exception as e:
        print(f"FAIL: Inference exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_inference()
