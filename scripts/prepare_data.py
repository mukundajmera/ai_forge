#!/usr/bin/env python3
"""
AI Forge - Data Preparation Helper Script

This script scans a local repository, extracts code blocks,
generates RAFT instruction-tuning data, and saves it as a JSON file
ready for the AI Forge training API.

Usage:
    python scripts/prepare_data.py --repo ./my_project --output training_data.json
"""

import argparse
import json
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from data_pipeline.miner import parse_repository
    from data_pipeline.raft_generator import generate_raft_examples
    from data_pipeline.schemas.raft_examples import RAFTConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please run this script from the project root or install requirements.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prepare_data")

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for AI Forge")
    parser.add_argument("--repo", type=str, required=True, help="Path to the repository to train on")
    parser.add_argument("--output", type=str, default="training_data.json", help="Output JSON file path")
    parser.add_argument("--num-examples", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--include-tests", action="store_true", help="Include test files in scanning")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo).resolve()
    output_path = Path(args.output).resolve()
    
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)
        
    logger.info(f"Scanning repository: {repo_path}")
    
    # 1. Parse Repository
    try:
        from data_pipeline.schemas.code_blocks import MinerConfig
        config = MinerConfig(include_tests=args.include_tests)
        
        blocks = parse_repository(str(repo_path), config)
        logger.info(f"Extracted {len(blocks)} suitable code blocks")
        
        if not blocks:
            logger.warning("No code blocks found! Check if the repo contains supported languages (.py, .js, .go, etc)")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Failed to parse repository: {e}")
        sys.exit(1)

    # 2. Generate RAFT Examples
    logger.info(f"Generating synthetic training data ({args.num_examples} examples)...")
    try:
        raft_config = RAFTConfig(
            questions_per_block=3,
            include_reasoning=True  # CoT improves performance using docstrings
        )
        
        # Using the convenience function that returns pure dicts
        examples = generate_raft_examples(
            blocks, 
            num_examples=args.num_examples,
            config=raft_config
        )
        
        logger.info(f"Generated {len(examples)} instruction-tuning examples")
        
    except Exception as e:
        logger.error(f"Failed to generate RAFT examples: {e}")
        sys.exit(1)
        
    # 3. Save to JSON
    # The training service expects a JSON format compatible with 'datasets' library
    # Structure: [{"instruction": "...", "input": "...", "output": "..."}, ...]
    
    training_data = []
    
    for ex in examples:
        # Format RAFT example for instruction tuning
        # Format:
        # Instruction: <question> <context>
        # Output: <reasoning> <answer>
        
        context_str = ""
        for doc in ex.get("oracle_documents", []):
            context_str += f"Document: {doc.get('name', 'code')}\n{doc.get('source_code', '')}\n\n"
        
        for doc in ex.get("distractor_documents", []):
            context_str += f"Document: {doc.get('name', 'code')}\n{doc.get('source_code', '')}\n\n"
            
        full_instruction = (
            f"Question: {ex['question']}\n\n"
            f"Context:\n{context_str}\n"
            "Answer the question based on the context provided."
        )
        
        full_output = f"Thinking Process:\n{ex['reasoning']}\n\nAnswer: {ex['final_answer']}"
        
        training_data.append({
            "instruction": full_instruction,
            "input": "", # Input is part of instruction in this format
            "output": full_output
        })
        
    # Save formatted data
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
        
    logger.info(f"Successfully saved {len(training_data)} training examples to {output_path}")
    print("\nNext Steps:")
    print("1. Start the API service:")
    print("   uvicorn conductor.service:app --reload")
    print("\n2. Submit fine-tuning job:")
    print(f"   curl -X POST http://localhost:8000/v1/fine-tune \\")
    print(f"     -F 'data_file=@{output_path}' \\")
    print("     -F 'request={\"project_name\": \"my-project\", \"epochs\": 3}'")

if __name__ == "__main__":
    main()
