"""RAFT Data Generator - Retrieval-Augmented Fine-Tuning Data Synthesis.

This module generates training data that simulates the RAG inference process,
teaching models to ground reasoning in provided context while ignoring noise.

RAFT Concept:
    By training on data that includes:
    1. Questions about the codebase
    2. Oracle documents (containing the answer)
    3. Distractor documents (irrelevant noise)
    4. Chain-of-thought reasoning citing oracles
    
    The model learns to:
    - Extract relevant information from noisy context
    - Ground reasoning in provided documents
    - Reduce hallucinations by learning citation habits

Example:
    >>> from data_pipeline.raft_generator import RAFTGenerator
    >>> generator = RAFTGenerator(code_blocks)
    >>> dataset = generator.generate_dataset(num_examples=1000)
    >>> training_data = dataset.to_training_format()
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Optional

from data_pipeline.schemas.code_blocks import CodeBlock
from data_pipeline.schemas.raft_examples import (
    Difficulty,
    QuestionType,
    QUESTION_TEMPLATES,
    RAFTConfig,
    RAFTDataset,
    RAFTExample,
    SimilarityScore,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Text Similarity (fallback when embeddings unavailable)
# =============================================================================

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts.
    
    Args:
        text1: First text.
        text2: Second text.
        
    Returns:
        Similarity score between 0 and 1.
    """
    # Tokenize into words
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def compute_code_similarity(block1: CodeBlock, block2: CodeBlock) -> float:
    """Compute similarity between two code blocks.
    
    Uses a combination of:
    - Language match (bonus)
    - Name similarity
    - Content overlap
    
    Args:
        block1: First code block.
        block2: Second code block.
        
    Returns:
        Similarity score between 0 and 1.
    """
    score = 0.0
    
    # Language match bonus
    if block1.language == block2.language:
        score += 0.1
    
    # Name similarity (helps find related functions)
    name_sim = jaccard_similarity(block1.name, block2.name)
    score += name_sim * 0.3
    
    # Content similarity
    content_sim = jaccard_similarity(block1.source_code, block2.source_code)
    score += content_sim * 0.6
    
    return min(score, 1.0)


# =============================================================================
# Embedding-based Similarity
# =============================================================================

class EmbeddingManager:
    """Manages embeddings for code blocks.
    
    Uses sentence-transformers for computing embeddings when available,
    falling back to Jaccard similarity otherwise.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize embedding manager.
        
        Args:
            model_name: HuggingFace model name for embeddings.
        """
        self.model_name = model_name
        self._model = None
        self._embeddings_cache: dict[str, Any] = {}
        self._available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if embedding model is available."""
        if self._available is not None:
            return self._available
        
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._available = True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to Jaccard similarity. "
                "Install with: pip install sentence-transformers"
            )
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self._available = False
        
        return self._available
    
    def get_embedding(self, text: str) -> Any:
        """Get embedding for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector or None if unavailable.
        """
        if not self.is_available():
            return None
        
        # Check cache
        cache_key = hash(text[:1000])  # Truncate for hashing
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        
        try:
            embedding = self._model.encode(text, convert_to_tensor=False)
            self._embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return jaccard_similarity(text1, text2)
        
        # Cosine similarity
        import numpy as np
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        return float(dot / norm) if norm > 0 else 0.0


# Global embedding manager
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingManager:
    """Get or create the global embedding manager."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(model_name)
    return _embedding_manager


# =============================================================================
# Question Generation
# =============================================================================

def generate_questions_for_block(
    block: CodeBlock,
    num_questions: int = 5,
    question_types: Optional[list[QuestionType]] = None,
) -> list[tuple[str, QuestionType]]:
    """Generate synthetic questions for a code block.
    
    Creates diverse questions probing different aspects:
    - Purpose/functionality
    - Usage patterns
    - Edge cases
    - Dependencies
    - Design decisions
    
    Args:
        block: Code block to generate questions for.
        num_questions: Number of questions to generate.
        question_types: Specific question types to use (optional).
        
    Returns:
        List of (question, question_type) tuples.
    """
    if question_types is None:
        # Select diverse question types
        question_types = list(QuestionType)
    
    questions: list[tuple[str, QuestionType]] = []
    used_templates: set[str] = set()
    
    # Extract patterns from code for template filling
    patterns_found: list[str] = []
    if "decorator" in block.source_code or "@" in block.source_code:
        patterns_found.append("decorator pattern")
    if "async" in block.source_code or "await" in block.source_code:
        patterns_found.append("async/await pattern")
    if "try" in block.source_code or "except" in block.source_code:
        patterns_found.append("exception handling")
    if "class" in block.source_code:
        patterns_found.append("object-oriented design")
    
    # Shuffle question types for diversity
    random.shuffle(question_types)
    
    attempts = 0
    max_attempts = num_questions * 3
    
    while len(questions) < num_questions and attempts < max_attempts:
        attempts += 1
        
        # Select question type
        q_type = question_types[len(questions) % len(question_types)]
        templates = QUESTION_TEMPLATES.get(q_type, [])
        
        if not templates:
            continue
        
        # Select template
        template = random.choice(templates)
        
        # Skip if already used
        if template in used_templates:
            continue
        
        # Fill template
        try:
            question = template.format(
                name=block.name,
                block_type=block.block_type,
                pattern=patterns_found[0] if patterns_found else "this approach",
                feature="new functionality",
                use_case="production",
                other=f"similar {block.block_type}s",
                alternative="other approaches",
            )
            
            questions.append((question, q_type))
            used_templates.add(template)
            
        except KeyError:
            # Template has unfilled placeholders, skip
            continue
    
    return questions[:num_questions]


# =============================================================================
# Oracle and Distractor Retrieval
# =============================================================================

def retrieve_oracle_and_distractors(
    query: str,
    all_blocks: list[CodeBlock],
    oracle_idx: int,
    k: int = 1,
    num_distractors: int = 2,
    use_embeddings: bool = True,
) -> tuple[list[CodeBlock], list[CodeBlock]]:
    """Retrieve oracle documents and distractors for a query.
    
    Oracle documents are the code blocks most relevant to answering the query.
    Distractors are code blocks that are superficially similar but not helpful
    for answering the question.
    
    Args:
        query: The question being asked.
        all_blocks: All available code blocks.
        oracle_idx: Index of the primary oracle block.
        k: Number of oracle documents to retrieve.
        num_distractors: Number of distractor documents.
        use_embeddings: Whether to use embedding-based retrieval.
        
    Returns:
        Tuple of (oracle_documents, distractor_documents).
    """
    if len(all_blocks) == 0:
        return [], []
    
    if oracle_idx >= len(all_blocks):
        oracle_idx = 0
    
    oracle_block = all_blocks[oracle_idx]
    
    # Compute similarities
    scores: list[SimilarityScore] = []
    
    if use_embeddings:
        embedding_mgr = get_embedding_manager()
        query_text = f"{query}\n{oracle_block.source_code}"
    
    for i, block in enumerate(all_blocks):
        if i == oracle_idx:
            # Oracle is always included
            scores.append(SimilarityScore(i, 1.0, is_oracle=True))
            continue
        
        if use_embeddings and embedding_mgr.is_available():
            sim = embedding_mgr.compute_similarity(
                query_text,
                block.source_code
            )
        else:
            sim = compute_code_similarity(oracle_block, block)
        
        scores.append(SimilarityScore(i, sim, is_oracle=False))
    
    # Sort by similarity (descending)
    scores.sort(reverse=True)
    
    # Select oracles (top k, always including the primary oracle)
    oracle_docs: list[CodeBlock] = [oracle_block]
    oracle_indices: set[int] = {oracle_idx}
    
    for score in scores:
        if len(oracle_docs) >= k:
            break
        if score.block_index not in oracle_indices and score.score > 0.5:
            oracle_docs.append(all_blocks[score.block_index])
            oracle_indices.add(score.block_index)
    
    # Select distractors (low similarity but not too low)
    # Ideal distractors are in the 0.1-0.4 similarity range
    distractor_candidates = [
        s for s in scores 
        if s.block_index not in oracle_indices 
        and 0.05 < s.score < 0.4
    ]
    
    # If not enough candidates, include more
    if len(distractor_candidates) < num_distractors:
        distractor_candidates = [
            s for s in scores 
            if s.block_index not in oracle_indices
        ]
    
    # Shuffle to add randomness
    random.shuffle(distractor_candidates)
    
    distractor_docs: list[CodeBlock] = []
    for score in distractor_candidates[:num_distractors]:
        distractor_docs.append(all_blocks[score.block_index])
    
    return oracle_docs, distractor_docs


# =============================================================================
# Chain-of-Thought Generation
# =============================================================================

def generate_chain_of_thought(
    query: str,
    oracle_docs: list[CodeBlock],
    question_type: QuestionType,
) -> str:
    """Generate chain-of-thought reasoning that cites oracle documents.
    
    Creates a reasoning path that explicitly references the oracle documents,
    teaching the model to ground its answers in provided context.
    
    Args:
        query: The question being answered.
        oracle_docs: Oracle documents containing the answer.
        question_type: Type of question being answered.
        
    Returns:
        Chain-of-thought reasoning string.
    """
    if not oracle_docs:
        return "No relevant documents found to answer this question."
    
    reasoning_parts: list[str] = []
    
    # Opening statement based on question type
    openings = {
        QuestionType.PURPOSE: "To understand the purpose, let me examine the code:",
        QuestionType.USAGE: "To explain the usage, I'll look at the implementation:",
        QuestionType.EDGE_CASES: "To identify edge case handling, I'll analyze the code:",
        QuestionType.DEPENDENCIES: "To list the dependencies, I'll check the imports and calls:",
        QuestionType.DESIGN: "To understand the design decisions, let me analyze:",
        QuestionType.EXTENSION: "To suggest extensions, I need to understand the current implementation:",
        QuestionType.DEBUGGING: "To identify potential issues, let me review the code:",
        QuestionType.COMPARISON: "To make a comparison, I'll examine the implementations:",
    }
    
    reasoning_parts.append(openings.get(
        question_type, 
        "Let me analyze the relevant code:"
    ))
    
    # Cite each oracle document
    for i, doc in enumerate(oracle_docs, 1):
        doc_name = doc.name if hasattr(doc, 'name') else f"Document {i}"
        doc_type = doc.block_type if hasattr(doc, 'block_type') else "code"
        
        # Extract key information from the code
        source = doc.source_code if hasattr(doc, 'source_code') else str(doc)
        
        # Generate observation based on question type
        if question_type == QuestionType.PURPOSE:
            if doc.docstring:
                reasoning_parts.append(
                    f"Looking at {doc_type} `{doc_name}`, the docstring states: "
                    f"'{doc.docstring[:100]}{'...' if len(doc.docstring or '') > 100 else ''}'"
                )
            else:
                reasoning_parts.append(
                    f"Examining {doc_type} `{doc_name}`, we can see it "
                    f"{'handles' if 'def' in source else 'defines'} "
                    f"{'the core functionality' if i == 1 else 'related operations'}."
                )
        
        elif question_type == QuestionType.USAGE:
            # Look for parameters
            param_match = re.search(r'\((.*?)\)', source)
            if param_match:
                params = param_match.group(1)
                reasoning_parts.append(
                    f"The {doc_type} `{doc_name}` accepts parameters: {params[:80]}..."
                    if len(params) > 80 else
                    f"The {doc_type} `{doc_name}` accepts: ({params})"
                )
            else:
                reasoning_parts.append(
                    f"Looking at `{doc_name}`, we can see how it should be called."
                )
        
        elif question_type == QuestionType.EDGE_CASES:
            has_try = 'try' in source or 'except' in source
            has_if = 'if' in source
            has_assert = 'assert' in source
            
            if has_try:
                reasoning_parts.append(
                    f"In `{doc_name}`, exception handling is implemented using try/except blocks."
                )
            elif has_if:
                reasoning_parts.append(
                    f"The `{doc_name}` includes conditional checks for validation."
                )
            elif has_assert:
                reasoning_parts.append(
                    f"The `{doc_name}` uses assertions for input validation."
                )
            else:
                reasoning_parts.append(
                    f"Looking at `{doc_name}` for edge case handling patterns."
                )
        
        elif question_type == QuestionType.DEPENDENCIES:
            # Look for imports
            imports = re.findall(r'(?:import|from)\s+(\w+)', source)
            if imports:
                reasoning_parts.append(
                    f"The `{doc_name}` depends on: {', '.join(set(imports[:5]))}."
                )
            else:
                reasoning_parts.append(
                    f"Examining `{doc_name}` for dependencies and external calls."
                )
        
        else:
            # Generic observation
            reasoning_parts.append(
                f"From `{doc_name}`, we can observe the implementation details."
            )
    
    # Closing statement
    reasoning_parts.append(
        "Based on this analysis of the provided code, I can now answer the question."
    )
    
    return "\n\n".join(reasoning_parts)


def generate_final_answer(
    query: str,
    oracle_docs: list[CodeBlock],
    question_type: QuestionType,
) -> str:
    """Generate a concise final answer to the question.
    
    Args:
        query: The question being answered.
        oracle_docs: Oracle documents containing the answer.
        question_type: Type of question.
        
    Returns:
        Concise answer string.
    """
    if not oracle_docs:
        return "Unable to determine from the provided context."
    
    primary_doc = oracle_docs[0]
    name = primary_doc.name
    docstring = primary_doc.docstring or ""
    
    if question_type == QuestionType.PURPOSE:
        if docstring:
            return docstring[:200] + ("..." if len(docstring) > 200 else "")
        return f"The {primary_doc.block_type} `{name}` provides functionality as shown in the implementation."
    
    elif question_type == QuestionType.USAGE:
        # Extract signature
        source = primary_doc.source_code
        sig_match = re.search(rf'{name}\s*\([^)]*\)', source)
        if sig_match:
            return f"Call using: `{sig_match.group(0)}`"
        return f"Use the {primary_doc.block_type} `{name}` as shown in the code."
    
    elif question_type == QuestionType.EDGE_CASES:
        source = primary_doc.source_code
        if 'try' in source and 'except' in source:
            return f"`{name}` handles errors using try/except blocks for graceful error handling."
        elif 'if' in source:
            return f"`{name}` includes conditional validation to handle edge cases."
        return f"`{name}` may need additional error handling for edge cases."
    
    elif question_type == QuestionType.DEPENDENCIES:
        source = primary_doc.source_code
        imports = re.findall(r'(?:import|from)\s+(\w+)', source)
        if imports:
            unique_imports = list(set(imports))[:5]
            return f"Dependencies: {', '.join(unique_imports)}"
        return f"`{name}` has minimal external dependencies."
    
    else:
        if docstring:
            return docstring[:150] + ("..." if len(docstring) > 150 else "")
        return f"See the implementation of `{name}` for details."


# =============================================================================
# RAFT Generator Class
# =============================================================================

class RAFTGenerator:
    """Generator for RAFT training data.
    
    Creates training examples that simulate RAG inference, pairing questions
    with oracle documents (containing answers) and distractor documents (noise).
    
    Attributes:
        blocks: Available code blocks.
        config: Generation configuration.
        
    Example:
        >>> blocks = parse_repository("/path/to/repo")
        >>> generator = RAFTGenerator(blocks)
        >>> dataset = generator.generate_dataset(num_examples=500)
        >>> print(f"Generated {len(dataset.examples)} examples")
    """
    
    def __init__(
        self,
        blocks: list[CodeBlock],
        config: Optional[RAFTConfig] = None,
    ) -> None:
        """Initialize RAFT generator.
        
        Args:
            blocks: Code blocks to generate examples from.
            config: Generation configuration.
        """
        self.blocks = blocks
        self.config = config or RAFTConfig()
        
        if not blocks:
            logger.warning("No code blocks provided to RAFT generator")
    
    def _select_difficulty(self) -> Difficulty:
        """Select difficulty level based on configured distribution."""
        rand = random.random()
        cumulative = 0.0
        
        for diff_name, prob in self.config.difficulty_distribution.items():
            cumulative += prob
            if rand < cumulative:
                return Difficulty(diff_name)
        
        return Difficulty.MEDIUM
    
    def _get_num_distractors(self, difficulty: Difficulty) -> int:
        """Get number of distractors for difficulty level."""
        if difficulty == Difficulty.EASY:
            return self.config.easy_distractors
        elif difficulty == Difficulty.MEDIUM:
            return self.config.medium_distractors
        else:
            return self.config.hard_distractors
    
    def generate_qa_pairs(
        self,
        code_blocks: Optional[list[CodeBlock]] = None,
    ) -> list[tuple[CodeBlock, str, QuestionType]]:
        """Generate question-answer pairs for code blocks.
        
        Args:
            code_blocks: Blocks to generate questions for (default: all blocks).
            
        Returns:
            List of (block, question, question_type) tuples.
        """
        blocks = code_blocks or self.blocks
        qa_pairs: list[tuple[CodeBlock, str, QuestionType]] = []
        
        for block in blocks:
            questions = generate_questions_for_block(
                block,
                num_questions=self.config.questions_per_block,
            )
            
            for question, q_type in questions:
                qa_pairs.append((block, question, q_type))
        
        logger.info(f"Generated {len(qa_pairs)} QA pairs from {len(blocks)} blocks")
        return qa_pairs
    
    def generate_example(
        self,
        block: CodeBlock,
        question: str,
        question_type: QuestionType,
        difficulty: Optional[Difficulty] = None,
    ) -> RAFTExample:
        """Generate a single RAFT example.
        
        Args:
            block: Primary oracle block.
            question: Question about the block.
            question_type: Type of question.
            difficulty: Difficulty level (auto-selected if None).
            
        Returns:
            Complete RAFT example.
        """
        if difficulty is None:
            difficulty = self._select_difficulty()
        
        # Get oracle index
        oracle_idx = 0
        for i, b in enumerate(self.blocks):
            if b.name == block.name and b.path == block.path:
                oracle_idx = i
                break
        
        # Retrieve oracle and distractor documents
        num_distractors = self._get_num_distractors(difficulty)
        oracle_docs, distractor_docs = retrieve_oracle_and_distractors(
            query=question,
            all_blocks=self.blocks,
            oracle_idx=oracle_idx,
            k=self.config.num_oracle_docs,
            num_distractors=num_distractors,
            use_embeddings=self.config.use_embeddings,
        )
        
        # Generate chain-of-thought reasoning
        if self.config.include_reasoning:
            reasoning = generate_chain_of_thought(question, oracle_docs, question_type)
        else:
            reasoning = ""
        
        # Generate final answer
        final_answer = generate_final_answer(question, oracle_docs, question_type)
        
        return RAFTExample(
            question=question,
            question_type=question_type,
            oracle_documents=oracle_docs,
            distractor_documents=distractor_docs,
            reasoning=reasoning,
            final_answer=final_answer,
            difficulty=difficulty,
            metadata={
                "primary_block": block.name,
                "primary_path": block.path,
            },
        )
    
    def generate_dataset(
        self,
        num_examples: Optional[int] = None,
        blocks: Optional[list[CodeBlock]] = None,
    ) -> RAFTDataset:
        """Generate a complete RAFT dataset.
        
        Args:
            num_examples: Maximum number of examples (default: all possible).
            blocks: Specific blocks to use (default: all blocks).
            
        Returns:
            RAFTDataset with generated examples.
        """
        target_blocks = blocks or self.blocks
        
        if not target_blocks:
            logger.warning("No blocks available for dataset generation")
            return RAFTDataset(examples=[], config=self.config)
        
        # Generate all QA pairs
        qa_pairs = self.generate_qa_pairs(target_blocks)
        
        # Limit if specified
        if num_examples and len(qa_pairs) > num_examples:
            random.shuffle(qa_pairs)
            qa_pairs = qa_pairs[:num_examples]
        
        # Generate examples
        dataset = RAFTDataset(config=self.config)
        
        for block, question, q_type in qa_pairs:
            try:
                example = self.generate_example(block, question, q_type)
                dataset.add_example(example)
            except Exception as e:
                logger.warning(f"Failed to generate example for {block.name}: {e}")
        
        # Compute statistics
        dataset.compute_statistics()
        
        logger.info(
            f"Generated RAFT dataset with {len(dataset.examples)} examples. "
            f"Stats: {dataset.statistics}"
        )
        
        return dataset


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_raft_examples(
    blocks: list[CodeBlock],
    num_examples: int = 100,
    config: Optional[RAFTConfig] = None,
) -> list[dict[str, Any]]:
    """Generate RAFT training examples from code blocks.
    
    Convenience function for quick dataset generation.
    
    Args:
        blocks: Code blocks to generate from.
        num_examples: Number of examples to generate.
        config: Optional configuration.
        
    Returns:
        List of training examples in dict format.
    """
    generator = RAFTGenerator(blocks, config)
    dataset = generator.generate_dataset(num_examples=num_examples)
    return dataset.to_training_format()


def export_raft_dataset(
    dataset: RAFTDataset,
    output_path: str,
    format: str = "json",
) -> None:
    """Export RAFT dataset to file.
    
    Args:
        dataset: Dataset to export.
        output_path: Output file path.
        format: Output format ('json' or 'jsonl').
    """
    import json
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset.examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "examples": dataset.examples,
                "statistics": dataset.statistics,
                "config": dataset.config.model_dump() if dataset.config else None,
            }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(dataset.examples)} examples to {output_path}")
