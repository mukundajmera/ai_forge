"""RAFT Generator - Retrieval-Augmented Fine-Tuning Data Synthesizer.

This module implements the RAFT (Retrieval-Augmented Fine-Tuning) methodology
for generating high-quality training data that combines retrieval context
with fine-tuning objectives.

RAFT Key Concepts:
    - D* (Oracle Document): Contains the answer to the question
    - D_k (Distractor Documents): Irrelevant context for robustness
    - Chain-of-Thought: Reasoning with cited evidence

Reference: https://arxiv.org/abs/2403.10131

Example:
    >>> generator = RAFTGenerator(chunks, config)
    >>> raft_data = generator.generate_dataset(num_samples=1000)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAFTConfig:
    """Configuration for RAFT data generation.
    
    Attributes:
        num_distractor_docs: Number of distractor documents per sample (k).
        oracle_probability: Probability of including oracle doc (P).
        include_chain_of_thought: Whether to generate CoT reasoning.
        max_context_length: Maximum combined context length.
        question_types: Types of questions to generate.
    """
    
    num_distractor_docs: int = 3  # D_k distractors
    oracle_probability: float = 0.8  # P(D* included)
    include_chain_of_thought: bool = True
    max_context_length: int = 4096
    question_types: list[str] = field(
        default_factory=lambda: [
            "what_does",  # What does X do?
            "how_to",     # How to use X?
            "explain",    # Explain X
            "debug",      # Why might X fail?
            "compare",    # Compare X and Y
        ]
    )


@dataclass
class RAFTSample:
    """A single RAFT training sample.
    
    Attributes:
        question: The generated question.
        oracle_doc: The document containing the answer (D*).
        distractor_docs: List of distractor documents (D_k).
        answer: The generated answer.
        chain_of_thought: Optional reasoning chain with citations.
        metadata: Additional metadata.
    """
    
    question: str
    oracle_doc: str
    distractor_docs: list[str]
    answer: str
    chain_of_thought: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_training_format(self, include_distractors: bool = True) -> dict[str, str]:
        """Convert to training format.
        
        Args:
            include_distractors: Whether to include distractor documents.
            
        Returns:
            Dictionary with instruction/input/output fields.
        """
        # Build context
        context_parts = [self.oracle_doc]
        if include_distractors:
            context_parts.extend(self.distractor_docs)
            random.shuffle(context_parts)  # Randomize order
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build response
        if self.chain_of_thought:
            response = f"<reasoning>\n{self.chain_of_thought}\n</reasoning>\n\n{self.answer}"
        else:
            response = self.answer
        
        return {
            "instruction": self.question,
            "input": context,
            "output": response,
        }


class RAFTGenerator:
    """Generates RAFT training data from code chunks.
    
    RAFT (Retrieval-Augmented Fine-Tuning) creates training data that
    teaches the model to extract answers from documents while ignoring
    irrelevant distractors.
    
    Attributes:
        chunks: Source code chunks for data generation.
        config: RAFT generation configuration.
        
    Example:
        >>> from ai_forge.data_pipeline.miner import CodeMiner, CodeChunk
        >>> chunks = CodeMiner("/project").extract_all()
        >>> generator = RAFTGenerator(chunks)
        >>> dataset = generator.generate_dataset(num_samples=500)
    """
    
    # Question templates by type
    QUESTION_TEMPLATES: dict[str, list[str]] = {
        "what_does": [
            "What does the {name} {chunk_type} do?",
            "Explain the purpose of {name}.",
            "What is the functionality of {name}?",
        ],
        "how_to": [
            "How do I use the {name} {chunk_type}?",
            "What are the parameters for {name}?",
            "Show me how to call {name}.",
        ],
        "explain": [
            "Explain the implementation of {name}.",
            "Walk through the logic in {name}.",
            "Describe how {name} works step by step.",
        ],
        "debug": [
            "What could cause {name} to fail?",
            "What are common errors when using {name}?",
            "How should I handle errors in {name}?",
        ],
        "compare": [
            "How does {name} differ from similar functions?",
            "What makes {name} unique?",
            "When should I use {name} vs alternatives?",
        ],
    }
    
    def __init__(
        self,
        chunks: list[Any],  # List[CodeChunk]
        config: Optional[RAFTConfig] = None,
    ) -> None:
        """Initialize RAFTGenerator.
        
        Args:
            chunks: List of CodeChunk objects from CodeMiner.
            config: Optional RAFT configuration.
        """
        self.chunks = chunks
        self.config = config or RAFTConfig()
        
        logger.info(f"Initialized RAFTGenerator with {len(chunks)} chunks")
    
    def _generate_question(
        self,
        chunk: Any,  # CodeChunk
        question_type: str,
    ) -> str:
        """Generate a question for a code chunk.
        
        Args:
            chunk: The CodeChunk to generate a question for.
            question_type: Type of question to generate.
            
        Returns:
            Generated question string.
        """
        templates = self.QUESTION_TEMPLATES.get(question_type, self.QUESTION_TEMPLATES["what_does"])
        template = random.choice(templates)
        
        return template.format(
            name=chunk.name,
            chunk_type=chunk.chunk_type,
        )
    
    def _generate_answer(
        self,
        chunk: Any,  # CodeChunk
        question: str,
    ) -> tuple[str, Optional[str]]:
        """Generate an answer with optional chain-of-thought.
        
        Args:
            chunk: The oracle CodeChunk.
            question: The question being answered.
            
        Returns:
            Tuple of (answer, chain_of_thought).
        """
        # TODO: Use LLM for better answer generation
        # For now, use docstring or generate placeholder
        
        if chunk.docstring:
            answer = chunk.docstring
        else:
            answer = f"The {chunk.chunk_type} '{chunk.name}' performs the following operations."
        
        chain_of_thought = None
        if self.config.include_chain_of_thought:
            chain_of_thought = (
                f"Looking at the provided documents, I found the {chunk.chunk_type} "
                f"'{chunk.name}' in the code. Based on its implementation:\n"
                f"1. It is defined starting at line {chunk.start_line}\n"
                f"2. The relevant code shows: {chunk.content[:200]}..."
            )
        
        return answer, chain_of_thought
    
    def _select_distractors(
        self,
        oracle_chunk: Any,  # CodeChunk
        num_distractors: int,
    ) -> list[str]:
        """Select distractor documents.
        
        Distractors should be plausible but not contain the answer.
        
        Args:
            oracle_chunk: The oracle chunk to avoid.
            num_distractors: Number of distractors to select.
            
        Returns:
            List of distractor document strings.
        """
        available = [c for c in self.chunks if c != oracle_chunk]
        
        if len(available) < num_distractors:
            logger.warning(
                f"Only {len(available)} chunks available for {num_distractors} distractors"
            )
            num_distractors = len(available)
        
        selected = random.sample(available, num_distractors)
        return [c.content for c in selected]
    
    def generate_sample(self, oracle_chunk: Any) -> RAFTSample:  # CodeChunk
        """Generate a single RAFT sample.
        
        Args:
            oracle_chunk: The chunk containing the answer.
            
        Returns:
            RAFTSample object.
        """
        # Select question type
        question_type = random.choice(self.config.question_types)
        
        # Generate question
        question = self._generate_question(oracle_chunk, question_type)
        
        # Generate answer
        answer, chain_of_thought = self._generate_answer(oracle_chunk, question)
        
        # Select distractors
        distractors = self._select_distractors(
            oracle_chunk,
            self.config.num_distractor_docs,
        )
        
        return RAFTSample(
            question=question,
            oracle_doc=oracle_chunk.content,
            distractor_docs=distractors,
            answer=answer,
            chain_of_thought=chain_of_thought,
            metadata={
                "oracle_name": oracle_chunk.name,
                "oracle_type": oracle_chunk.chunk_type,
                "question_type": question_type,
            },
        )
    
    def generate_dataset(
        self,
        num_samples: Optional[int] = None,
        include_oracle_probability: Optional[float] = None,
    ) -> list[dict[str, str]]:
        """Generate full RAFT training dataset.
        
        Args:
            num_samples: Number of samples to generate (default: len(chunks)).
            include_oracle_probability: Probability of including oracle doc.
            
        Returns:
            List of training examples in instruction/input/output format.
        """
        if num_samples is None:
            num_samples = len(self.chunks)
        
        if include_oracle_probability is None:
            include_oracle_probability = self.config.oracle_probability
        
        dataset: list[dict[str, str]] = []
        
        for _ in range(num_samples):
            # Select random oracle chunk
            oracle_chunk = random.choice(self.chunks)
            
            # Generate sample
            sample = self.generate_sample(oracle_chunk)
            
            # Decide whether to include oracle (RAFT's key training signal)
            include_oracle = random.random() < include_oracle_probability
            
            if not include_oracle:
                # Replace oracle with another distractor
                sample.oracle_doc = random.choice(sample.distractor_docs) if sample.distractor_docs else ""
            
            dataset.append(sample.to_training_format(include_distractors=True))
        
        logger.info(f"Generated {len(dataset)} RAFT samples")
        return dataset
    
    def generate_with_llm(
        self,
        llm_client: Any,
        num_samples: int = 100,
    ) -> list[dict[str, str]]:
        """Generate RAFT data using an LLM for question/answer synthesis.
        
        This produces higher quality data by using an LLM to generate
        contextually appropriate questions and detailed answers.
        
        Args:
            llm_client: LLM client for generation (e.g., Ollama client).
            num_samples: Number of samples to generate.
            
        Returns:
            List of training examples.
        """
        # TODO: Implement LLM-powered generation
        raise NotImplementedError("LLM-based generation not yet implemented")
