import sys
import os
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from training.schemas import FineTuneConfig, TrainingConfig, ModelConfig, CheckpointConfig, LoggingConfig, PiSSAConfig, EvaluationConfig, MemoryConfig
from training.forge import FineTuneTrainer

def test_training_dryrun():
    print("Starting training dry-run...")
    
    # 1. Setup Config
    # Use a very small model or mock. GPT2 is small enough for a test.
    # We disable 4-bit loading to avoid bitsandbytes dependency issues if not present on Mac standard load
    config = FineTuneConfig(
        model=ModelConfig(
            base_model="gpt2",
            load_in_4bit=False, 
        ),
        training=TrainingConfig(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            use_cpu=False # Allow auto-detect (should use MPS)
        ),
        pissa=PiSSAConfig(
            target_modules=["c_attn"] 
        ),
        checkpoints=CheckpointConfig(
            save_strategy="no",
            load_best_model_at_end=False
        ),
        evaluation=EvaluationConfig(
            strategy="no"
        ),
        logging=LoggingConfig(
            output_dir="./output/dryrun",
            report_to="none"
        ),
        memory=MemoryConfig(
             gradient_checkpointing=False
        )
    )
    
    # 2. Setup Dummy Dataset
    from datasets import Dataset
    data = [
        {"text": "Hello world, this is a test."},
    ] + [{"text": "Another sample line for training."}] * 10
    dataset = Dataset.from_list(data)
    
    # Tokenization function (simplified)
    def tokenize(example):
        return {"input_ids": [1, 2, 3], "labels": [1, 2, 3]} # Mock tokenization output for speed?
        # Actually gpt2 tokenizer is fast. Let's rely on real one to test flow.
    
    # 3. Initialize Trainer
    trainer = FineTuneTrainer(config)
    
    try:
        # Load Model
        print("Loading model (gpt2)...")
        trainer.load_model()
        
        # Real tokenization using loaded tokenizer
        print("Tokenizing data...")
        trainer.tokenizer.pad_token = trainer.tokenizer.eos_token
        
        def tokenize_calc(examples):
            return trainer.tokenizer(examples["text"], padding="max_length", max_length=16, truncation=True)
            
        tokenized_ds = dataset.map(tokenize_calc, batched=True)
        # Add labels
        def add_labels(examples):
            examples["labels"] = examples["input_ids"]
            return examples
        tokenized_ds = tokenized_ds.map(add_labels, batched=True)
        
        # 4. Train
        print("Starting train loop...")
        trainer.train(train_dataset=tokenized_ds)
        
        print("Training completed.")
        print("Saving model...")
        trainer.save_model("./output/dryrun")
        print("Model saved.")
        
        print("PASS: Dry run successful")
        
    except Exception as e:
        print(f"FAIL: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        # if Path("./output/dryrun").exists():
        #     shutil.rmtree("./output/dryrun")
        pass

if __name__ == "__main__":
    test_training_dryrun()
