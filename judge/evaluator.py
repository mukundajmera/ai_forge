"""Model Evaluator - Multi-metric evaluation for fine-tuned models.

This module provides comprehensive evaluation using multiple metrics
including CodeBLEU, HumanEval, Perplexity, and Pass@k.

Example:
    >>> evaluator = ModelEvaluator(model, tokenizer)
    >>> results = evaluator.evaluate_all(test_dataset)
    >>> print(f"Perplexity: {results.perplexity:.2f}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from model evaluation.
    
    Attributes:
        perplexity: Model perplexity on test set.
        loss: Average loss on test set.
        accuracy: Token-level accuracy.
        codebleu: CodeBLEU score (for code models).
        pass_at_1: Pass@1 for code execution.
        pass_at_10: Pass@10 for code execution.
        humaneval_score: HumanEval benchmark score.
        domain_accuracy: Domain-specific accuracy.
        hallucination_rate: Estimated hallucination rate.
        latency_ms: Average inference latency.
        additional_metrics: Any additional metrics.
    """
    
    perplexity: float
    loss: float
    accuracy: Optional[float] = None
    codebleu: Optional[float] = None
    pass_at_1: Optional[float] = None
    pass_at_10: Optional[float] = None
    humaneval_score: Optional[float] = None
    domain_accuracy: Optional[float] = None
    hallucination_rate: Optional[float] = None
    latency_ms: Optional[float] = None
    additional_metrics: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        result = {
            "perplexity": self.perplexity,
            "loss": self.loss,
        }
        
        for field_name in ["accuracy", "codebleu", "pass_at_1", "pass_at_10",
                           "humaneval_score", "domain_accuracy", 
                           "hallucination_rate", "latency_ms"]:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        
        result.update(self.additional_metrics)
        return result
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"📊 Evaluation Results",
            f"  Perplexity: {self.perplexity:.2f}",
            f"  Loss: {self.loss:.4f}",
        ]
        
        if self.accuracy is not None:
            lines.append(f"  Accuracy: {self.accuracy:.2%}")
        if self.codebleu is not None:
            lines.append(f"  CodeBLEU: {self.codebleu:.2f}")
        if self.pass_at_1 is not None:
            lines.append(f"  Pass@1: {self.pass_at_1:.2%}")
        if self.domain_accuracy is not None:
            lines.append(f"  Domain Accuracy: {self.domain_accuracy:.2%}")
        if self.latency_ms is not None:
            lines.append(f"  Latency: {self.latency_ms:.1f}ms")
        
        return "\n".join(lines)


@dataclass
class EvaluatorConfig:
    """Configuration for ModelEvaluator.
    
    Attributes:
        batch_size: Batch size for evaluation.
        max_samples: Maximum samples to evaluate.
        compute_perplexity: Whether to compute perplexity.
        compute_codebleu: Whether to compute CodeBLEU.
        compute_pass_at_k: Whether to compute Pass@k.
        num_generations_per_sample: Generations per sample for Pass@k.
        max_new_tokens: Maximum tokens for generation.
        temperature: Temperature for generation.
    """
    
    batch_size: int = 8
    max_samples: Optional[int] = None
    compute_perplexity: bool = True
    compute_codebleu: bool = False
    compute_pass_at_k: bool = False
    num_generations_per_sample: int = 10
    max_new_tokens: int = 256
    temperature: float = 0.8


class ModelEvaluator:
    """Multi-metric model evaluator.
    
    Evaluates fine-tuned models using various metrics appropriate
    for code and text generation tasks.
    
    Attributes:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        config: Evaluation configuration.
        
    Example:
        >>> evaluator = ModelEvaluator(model, tokenizer)
        >>> results = evaluator.evaluate_all(test_data)
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        config: Optional[EvaluatorConfig] = None,
    ) -> None:
        """Initialize ModelEvaluator.
        
        Args:
            model: Model to evaluate.
            tokenizer: Tokenizer for the model.
            config: Evaluation configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluatorConfig()
        
        # Put model in eval mode
        self.model.eval()
        
        logger.info("Initialized ModelEvaluator")
    
    def compute_perplexity(self, dataset: "Dataset") -> tuple[float, float]:
        """Compute perplexity on a dataset.
        
        Args:
            dataset: HuggingFace dataset with 'text' field.
            
        Returns:
            Tuple of (perplexity, average_loss).
        """
        import torch
        
        total_loss = 0.0
        total_tokens = 0
        
        # Limit samples if configured
        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))
        
        with torch.no_grad():
            for i in range(0, len(dataset), self.config.batch_size):
                batch = dataset[i:i + self.config.batch_size]
                texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
                
                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                
                # Move to model device
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                
                loss = outputs.loss.item()
                num_tokens = inputs["attention_mask"].sum().item()
                
                total_loss += loss * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.2f}, Loss: {avg_loss:.4f}")
        return perplexity, avg_loss
    
    def compute_accuracy(
        self,
        dataset: "Dataset",
        answer_extractor: Optional[Callable[[str], str]] = None,
    ) -> float:
        """Compute accuracy on QA-style dataset.
        
        Args:
            dataset: Dataset with 'instruction', 'input', 'output' fields.
            answer_extractor: Optional function to extract answer from output.
            
        Returns:
            Accuracy as a float between 0 and 1.
        """
        import torch
        
        correct = 0
        total = 0
        
        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))
        
        with torch.no_grad():
            for example in dataset:
                # Generate response
                prompt = f"[INST] {example['instruction']}\n\n{example.get('input', '')} [/INST]"
                
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.1,  # Low temp for accuracy
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                
                expected = example["output"]
                
                # Extract answers if function provided
                if answer_extractor:
                    generated = answer_extractor(generated)
                    expected = answer_extractor(expected)
                
                # Simple equality check (can be made more sophisticated)
                if generated.strip().lower() == expected.strip().lower():
                    correct += 1
                
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy
    
    def compute_codebleu(
        self,
        predictions: list[str],
        references: list[str],
        language: str = "python",
    ) -> float:
        """Compute CodeBLEU score.
        
        CodeBLEU combines n-gram matching, syntax matching (AST),
        and semantic matching for code evaluation.
        
        Args:
            predictions: Generated code samples.
            references: Reference code samples.
            language: Programming language.
            
        Returns:
            CodeBLEU score between 0 and 1.
        """
        try:
            from codebleu import calc_codebleu
            
            result = calc_codebleu(
                references=[[ref] for ref in references],
                predictions=predictions,
                lang=language,
            )
            
            score = result["codebleu"]
            logger.info(f"CodeBLEU: {score:.4f}")
            return score
            
        except ImportError:
            logger.warning("codebleu not installed, skipping CodeBLEU")
            return 0.0
    
    def compute_pass_at_k(
        self,
        problems: list[dict[str, Any]],
        k_values: list[int] = [1, 10],
    ) -> dict[str, float]:
        """Compute Pass@k metric for code generation.
        
        Generates multiple solutions per problem and checks
        how many pass unit tests.
        
        Args:
            problems: List of problems with 'prompt' and 'test' fields.
            k_values: Values of k for Pass@k computation.
            
        Returns:
            Dictionary with pass_at_k scores.
        """
        import numpy as np
        
        results = {f"pass_at_{k}": [] for k in k_values}
        
        for problem in problems:
            prompt = problem["prompt"]
            test_code = problem.get("test", "")
            
            # Generate multiple solutions
            num_correct = 0
            num_samples = self.config.num_generations_per_sample
            
            for _ in range(num_samples):
                generated = self._generate_code(prompt)
                
                # Run test
                if self._execute_test(generated, test_code):
                    num_correct += 1
            
            # Compute pass@k for each k
            for k in k_values:
                if num_samples >= k:
                    # Pass@k formula
                    pass_k = 1.0 - np.prod([
                        (num_samples - num_correct - i) / (num_samples - i)
                        for i in range(k)
                    ]) if num_correct > 0 else 0.0
                    results[f"pass_at_{k}"].append(pass_k)
        
        # Average across problems
        avg_results = {
            key: np.mean(values) if values else 0.0
            for key, values in results.items()
        }
        
        logger.info(f"Pass@k: {avg_results}")
        return avg_results
    
    def _generate_code(self, prompt: str) -> str:
        """Generate code from prompt.
        
        Args:
            prompt: Code generation prompt.
            
        Returns:
            Generated code string.
        """
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
    
    def _execute_test(self, code: str, test_code: str) -> bool:
        """Execute code and test.
        
        Args:
            code: Generated code.
            test_code: Test code to run.
            
        Returns:
            True if tests pass, False otherwise.
        """
        try:
            # Create execution environment
            exec_globals: dict[str, Any] = {}
            
            # Execute generated code
            exec(code, exec_globals)
            
            # Execute tests
            exec(test_code, exec_globals)
            
            return True
        except Exception:
            return False
    
    def measure_latency(self, num_samples: int = 10) -> float:
        """Measure average inference latency.
        
        Args:
            num_samples: Number of samples to measure.
            
        Returns:
            Average latency in milliseconds.
        """
        import time
        import torch
        
        prompt = "What is the capital of France?"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Warmup
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=10)
        
        # Measure
        latencies = []
        for _ in range(num_samples):
            start = time.perf_counter()
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=50)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Average latency: {avg_latency:.1f}ms")
        return avg_latency
    
    def evaluate_reconstruction(
        self,
        original_code_blocks: list[str],
        prompts: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Evaluate how well the model can reproduce training code.
        
        Tests pattern memorization by asking the model to complete or
        reproduce code it was trained on.
        
        Args:
            original_code_blocks: Original code samples from training.
            prompts: Optional custom prompts (defaults to first 50 chars).
            
        Returns:
            Dictionary with reconstruction metrics:
            - edit_distance: Average normalized edit distance (0=identical)
            - exact_match_rate: Fraction of exact matches
            - semantic_similarity: Cosine similarity of embeddings
        """
        import torch
        from difflib import SequenceMatcher
        
        if self.config.max_samples:
            original_code_blocks = original_code_blocks[:self.config.max_samples]
        
        edit_distances = []
        exact_matches = 0
        similarities = []
        
        with torch.no_grad():
            for i, original in enumerate(original_code_blocks):
                # Create prompt (first 50 chars or custom)
                if prompts and i < len(prompts):
                    prompt = prompts[i]
                else:
                    prompt = original[:min(50, len(original))] + "\n# Complete this code:\n"
                
                # Generate completion
                generated = self._generate_code(prompt)
                
                # Compute edit distance (normalized)
                matcher = SequenceMatcher(None, original, generated)
                similarity = matcher.ratio()
                edit_distance = 1.0 - similarity
                edit_distances.append(edit_distance)
                
                # Exact match
                if original.strip() == generated.strip():
                    exact_matches += 1
                
                # Store similarity for averaging
                similarities.append(similarity)
        
        total = len(original_code_blocks)
        result = {
            "edit_distance": sum(edit_distances) / total if total > 0 else 0.0,
            "exact_match_rate": exact_matches / total if total > 0 else 0.0,
            "semantic_similarity": sum(similarities) / total if total > 0 else 0.0,
        }
        
        logger.info(f"Reconstruction metrics: {result}")
        return result
    
    def evaluate_hallucination_rate(
        self,
        raft_test_set: list[dict[str, Any]],
    ) -> float:
        """Evaluate hallucination rate using RAFT examples.
        
        RAFT examples contain a question, correct context, and distractor
        documents. The model should answer from context, not invent code.
        
        Args:
            raft_test_set: List of RAFT examples with keys:
                - question: The question to answer
                - context: Correct context document
                - distractors: List of incorrect documents
                - answer: Expected answer
                
        Returns:
            Hallucination rate (0.0 = no hallucinations, 1.0 = all hallucinated).
        """
        import torch
        
        if self.config.max_samples:
            raft_test_set = raft_test_set[:self.config.max_samples]
        
        hallucinations = 0
        total = 0
        
        with torch.no_grad():
            for example in raft_test_set:
                question = example.get("question", "")
                context = example.get("context", "")
                distractors = example.get("distractors", [])
                expected_answer = example.get("answer", "")
                
                # Create prompt with context
                prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
                
                # Generate answer
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                
                # Check for hallucination:
                # If answer contains code not in context or distractors, it's hallucinated
                is_hallucination = self._detect_hallucination(
                    generated, context, distractors, expected_answer
                )
                
                if is_hallucination:
                    hallucinations += 1
                total += 1
        
        rate = hallucinations / total if total > 0 else 0.0
        logger.info(f"Hallucination rate: {rate:.2%} ({hallucinations}/{total})")
        return rate
    
    def _detect_hallucination(
        self,
        generated: str,
        context: str,
        distractors: list[str],
        expected: str,
    ) -> bool:
        """Detect if generated output is a hallucination.
        
        Args:
            generated: Model's generated answer.
            context: Correct context.
            distractors: Distractor documents.
            expected: Expected answer.
            
        Returns:
            True if hallucination detected.
        """
        # If answer closely matches expected, not hallucinated
        from difflib import SequenceMatcher
        
        similarity_to_expected = SequenceMatcher(None, generated.lower(), expected.lower()).ratio()
        if similarity_to_expected > 0.7:
            return False
        
        # Check if generated content appears in any source
        all_sources = context + " ".join(distractors)
        
        # Extract code-like tokens from generated text
        import re
        code_tokens = set(re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', generated))
        source_tokens = set(re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', all_sources))
        
        # If many tokens in generated are not in sources, likely hallucinated
        if code_tokens:
            novel_tokens = code_tokens - source_tokens
            novelty_rate = len(novel_tokens) / len(code_tokens)
            
            # High novelty rate indicates hallucination
            if novelty_rate > 0.5:
                return True
        
        return False
    
    def run_humaneval_subset(
        self,
        num_samples: int = 100,
        k_values: list[int] = [1, 10],
    ) -> dict[str, float]:
        """Run HumanEval benchmark on a subset of problems.
        
        Loads HumanEval problems and evaluates pass@k.
        
        Args:
            num_samples: Number of problems to evaluate.
            k_values: k values for pass@k computation.
            
        Returns:
            Dictionary with pass@k scores.
        """
        try:
            from datasets import load_dataset
            
            # Load HumanEval dataset
            dataset = load_dataset("openai_humaneval", split="test")
            
            # Limit to num_samples
            problems = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                    
                problems.append({
                    "prompt": example["prompt"],
                    "test": example["test"],
                    "entry_point": example["entry_point"],
                })
            
            logger.info(f"Running HumanEval on {len(problems)} problems...")
            
            # Use existing pass@k implementation
            results = self.compute_pass_at_k(problems, k_values)
            
            # Add humaneval prefix
            humaneval_results = {f"humaneval_{k}": v for k, v in results.items()}
            
            logger.info(f"HumanEval results: {humaneval_results}")
            return humaneval_results
            
        except ImportError:
            logger.warning("datasets library not installed, skipping HumanEval")
            return {}
        except Exception as e:
            logger.error(f"HumanEval evaluation failed: {e}")
            return {}
    
    def evaluate_all(self, dataset: "Dataset") -> EvaluationResult:
        """Run all configured evaluations.
        
        Args:
            dataset: Test dataset.
            
        Returns:
            EvaluationResult with all metrics.
        """
        logger.info("Starting full evaluation...")
        
        # Perplexity (always computed)
        perplexity, loss = self.compute_perplexity(dataset)
        
        result = EvaluationResult(
            perplexity=perplexity,
            loss=loss,
        )
        
        # Latency
        result.latency_ms = self.measure_latency()
        
        # CodeBLEU (optional)
        if self.config.compute_codebleu:
            # Would need predictions and references
            pass
        
        # Pass@k (optional)
        if self.config.compute_pass_at_k:
            # Would need problems with tests
            pass
        
        logger.info(f"Evaluation complete:\n{result.summary()}")
        return result

