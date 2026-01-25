import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from training.schemas import FineTuneConfig, TrainingConfig
from training.forge import FineTuneTrainer
from unittest.mock import MagicMock, patch

def test_epochs_passing():
    target_epochs = 10
    print(f"Testing with target_epochs={target_epochs}")
    
    config = FineTuneConfig(
        training=TrainingConfig(
            num_train_epochs=target_epochs
        )
    )
    
    trainer = FineTuneTrainer(config)
    trainer.model = MagicMock() # Mock model to avoid load_model error
    trainer.tokenizer = MagicMock()
    
    # Mock transformers.Trainer to intercept args
    with patch('transformers.Trainer') as mock_trainer:
         # Mock TrainingArguments to check what's passed to it
        with patch('transformers.TrainingArguments') as mock_args_cls:
             # Just return a mock object that acts like TrainingArguments
             mock_args_instance = MagicMock()
             mock_args_cls.return_value = mock_args_instance
             
             try:
                 trainer.train(train_dataset=[], eval_dataset=[])
             except Exception as e:
                 # It might fail later in Trainer init because of datasets being empty lists not datasets
                 # But we only care if TrainingArguments was initialized correctly before that
                 pass
             
             # Check call args of TrainingArguments
             # The call arguments are passed as kwargs to TrainingArguments constructor
             if mock_args_cls.call_count == 0:
                 print("FAIL: TrainingArguments was not instantiated")
                 sys.exit(1)
                 
             call_args = mock_args_cls.call_args
             # call_args is (args, kwargs)
             kwargs = call_args[1]
             
             passed_epochs = kwargs.get('num_train_epochs')
             print(f"Passed num_train_epochs: {passed_epochs}")
             
             if passed_epochs != target_epochs:
                 print(f"FAIL: Epochs mismatch! Expected {target_epochs}, got {passed_epochs}")
                 sys.exit(1)
             else:
                 print("PASS: Epochs passed correctly.")

if __name__ == "__main__":
    test_epochs_passing()
