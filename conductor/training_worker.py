
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import traceback

# Setup logging for worker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

def run_training_job(job_id: str, config_dict: dict, data_path: str, output_base_dir: str = "./output") -> dict:
    """
    Run training job in a separate process.
    Returns a dict with verification results/metrics or raises Exception.
    """
    progress_file = None
    try:
        # Setup progress tracking
        job_output_dir = Path(f"{output_base_dir}/{job_id}")
        job_output_dir.mkdir(parents=True, exist_ok=True)
        progress_file = job_output_dir / "progress.json"
        
        def update_progress(status: str, progress: float = 0.0, **kwargs):
            """Write progress to file for main process to read."""
            progress_data = {
                "status": status,
                "progress": progress,
                "updated_at": datetime.now().isoformat(),
                **kwargs
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)
        
        update_progress("initializing", 5.0)
        
        # 1. Setup Environment
        # Add project root to path
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        pocs_root = project_root.parent
        
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if str(pocs_root) not in sys.path:
            sys.path.insert(0, str(pocs_root))

        # 2. Imports (inside function to avoid circular/top-level issues in parent process)
        update_progress("loading_modules", 10.0)
        try:
            from ai_forge.training import FineTuneTrainer, FineTuneConfig
            from ai_forge.training.schemas import (
                ModelConfig, 
                TrainingConfig, 
                PiSSAConfig, 
                LoggingConfig,
                InitMethod
            )
        except ImportError:
            # Fallback for when running from root without package install
            sys.path.insert(0, str(project_root))
            from training import FineTuneTrainer, FineTuneConfig
            from training.schemas import (
                ModelConfig, 
                TrainingConfig, 
                PiSSAConfig, 
                LoggingConfig,
                InitMethod
            )
            
        # 3. Configure
        init_method = InitMethod.PISSA if config_dict.get("use_pissa", True) else InitMethod.GAUSSIAN
        
        ft_config = FineTuneConfig(
            model=ModelConfig(
                base_model=config_dict["base_model"],
            ),
            training=TrainingConfig(
                num_train_epochs=config_dict.get("epochs", 3),
                learning_rate=config_dict.get("learning_rate", 2e-4),
                per_device_train_batch_size=config_dict.get("batch_size", 2),
            ),
            pissa=PiSSAConfig(
                rank=config_dict.get("rank", 32),
                init_method=init_method,
            ),
            logging=LoggingConfig(
                output_dir=f"{output_base_dir}/{job_id}",
            ),
        )

        # 4. Validate Data
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
            
        with open(data_path) as f:
            json.load(f) # Validate JSON

        # 5. Initialize Trainer
        update_progress("loading_model", 20.0)
        logger.info(f"Worker {os.getpid()}: Initializing trainer for {job_id}")
        forge = FineTuneTrainer(ft_config)
        forge.load_model()
        
        # 6. Load & Process Dataset
        update_progress("preparing_data", 30.0)
        from datasets import load_dataset
        raw_dataset = load_dataset("json", data_files=data_path)["train"]
        
        def tokenize_function(examples):
            texts = []
            instructions = examples.get("instruction", [])
            outputs = examples.get("output", [])
            
            for i in range(len(instructions)):
                inst = instructions[i] or ""
                out = outputs[i] or ""
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                texts.append(text)
            
            tokenized = forge.tokenizer(
                texts,
                truncation=True,
                max_length=forge.config.model.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
        )
        
        n_samples = len(dataset)
        eval_size = 1 if n_samples < 10 else int(n_samples * 0.1)
        if eval_size < 1: eval_size = 1
        
        splits = dataset.train_test_split(test_size=eval_size, seed=42)
        
        # 7. Train
        update_progress("training", 40.0)
        logger.info(f"Worker {os.getpid()}: Starting training loop")
        results = forge.train(train_dataset=splits["train"], eval_dataset=splits["test"])
        
        # 8. Save
        update_progress("saving", 90.0)
        forge.save_model(f"{output_base_dir}/{job_id}/final")
        
        update_progress("completed", 100.0, loss=results.get("train_loss", 0.0))
        
        return {
            "success": True,
            "loss": results.get("train_loss", 0.0),
            "output_dir": f"{output_base_dir}/{job_id}/final"
        }

    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed: {e}")
        traceback.print_exc()
        if progress_file:
            try:
                with open(progress_file, 'w') as f:
                    json.dump({
                        "status": "failed",
                        "error": str(e),
                        "updated_at": datetime.now().isoformat()
                    }, f)
            except:
                pass
        return {
            "success": False,
            "error": str(e)
        }
