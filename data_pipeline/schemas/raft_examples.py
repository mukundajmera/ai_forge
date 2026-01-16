"""RAFT Example Schemas - Data structures for Retrieval-Augmented Fine-Tuning.

This module defines the Pydantic models and dataclasses for RAFT training data,
which pairs queries with oracle documents, distractors, and chain-of-thought reasoning.

RAFT Concept:
    Training data simulates the RAG inference process:
    1. Question: A query about the codebase
    2. Oracle Documents: Code snippets containing the answer
    3. Distractor Documents: Irrelevant code snippets
    4. Chain-of-Thought: Reasoning path citing oracle documents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    """Difficulty levels for RAFT examples."""
    
    EASY = "easy"      # Single function, obvious oracle, 0 distractors
    MEDIUM = "medium"  # Related functions, 1-2 distractors
    HARD = "hard"      # Complex dependencies, 3+ distractors


class QuestionType(str, Enum):
    """Types of synthetic questions."""
    
    PURPOSE = "purpose"           # What does this do?
    USAGE = "usage"               # How do I use this?
    EDGE_CASES = "edge_cases"     # How does it handle X?
    DEPENDENCIES = "dependencies" # What does it depend on?
    DESIGN = "design"             # Why is it implemented this way?
    EXTENSION = "extension"       # How would you extend this?
    DEBUGGING = "debugging"       # What could go wrong?
    COMPARISON = "comparison"     # How does this compare to Y?


# Question templates organized by type
QUESTION_TEMPLATES: dict[QuestionType, list[str]] = {
    QuestionType.PURPOSE: [
        "What is the purpose of {name}?",
        "What does the {block_type} {name} do?",
        "Explain what {name} accomplishes.",
        "Describe the functionality of {name}.",
        "What problem does {name} solve?",
    ],
    QuestionType.USAGE: [
        "How do I use {name}?",
        "What are the parameters for {name}?",
        "Show an example of using {name}.",
        "What does {name} return?",
        "What arguments does {name} accept?",
    ],
    QuestionType.EDGE_CASES: [
        "How does {name} handle invalid input?",
        "What happens if {name} receives null/None?",
        "How does {name} handle edge cases?",
        "What errors might {name} throw?",
        "How does {name} handle empty input?",
    ],
    QuestionType.DEPENDENCIES: [
        "What are the dependencies of {name}?",
        "What does {name} import or rely on?",
        "What other functions does {name} call?",
        "What modules does {name} depend on?",
        "What external libraries does {name} use?",
    ],
    QuestionType.DESIGN: [
        "Why is {name} implemented this way?",
        "What design pattern does {name} use?",
        "Why does {name} use {pattern}?",
        "What are the alternatives to {name}'s approach?",
        "What trade-offs does {name} make?",
    ],
    QuestionType.EXTENSION: [
        "How would you extend {name} to handle {feature}?",
        "How could {name} be improved?",
        "What features could be added to {name}?",
        "How would you modify {name} for {use_case}?",
        "How could {name} be made more efficient?",
    ],
    QuestionType.DEBUGGING: [
        "What could go wrong with {name}?",
        "How would you debug {name}?",
        "What are common issues with {name}?",
        "How would you test {name}?",
        "What logging would help debug {name}?",
    ],
    QuestionType.COMPARISON: [
        "How does {name} compare to {other}?",
        "What's the difference between {name} and {other}?",
        "When should I use {name} vs {other}?",
        "What are the pros and cons of {name}?",
        "Is {name} better than {alternative}?",
    ],
}


@dataclass
class RAFTExample:
    """A single RAFT training example.
    
    Represents a complete training sample for Retrieval-Augmented Fine-Tuning,
    including the question, oracle documents (ground truth), distractors
    (noise), and the chain-of-thought reasoning.
    
    Attributes:
        question: The user query about the codebase.
        question_type: Category of the question.
        oracle_documents: Code blocks containing the answer.
        distractor_documents: Irrelevant code blocks (noise).
        reasoning: Chain-of-thought explanation citing sources.
        final_answer: Concise answer to the question.
        difficulty: Difficulty level (easy/medium/hard).
        metadata: Additional information.
        
    Example:
        >>> example = RAFTExample(
        ...     question="What does the calculate_sum function do?",
        ...     question_type=QuestionType.PURPOSE,
        ...     oracle_documents=[code_block],
        ...     distractor_documents=[],
        ...     reasoning="Looking at the calculate_sum function, we see it...",
        ...     final_answer="It adds two numbers and returns the sum.",
        ...     difficulty=Difficulty.EASY,
        ... )
    """
    
    question: str
    question_type: QuestionType
    oracle_documents: list[Any]  # List[CodeBlock]
    distractor_documents: list[Any]  # List[CodeBlock]
    reasoning: str
    final_answer: str
    difficulty: Difficulty
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_context_length(self) -> int:
        """Total length of all context documents."""
        total = 0
        for doc in self.oracle_documents + self.distractor_documents:
            if hasattr(doc, 'source_code'):
                total += len(doc.source_code)
            elif isinstance(doc, dict):
                total += len(doc.get('source_code', ''))
        return total
    
    @property
    def oracle_ratio(self) -> float:
        """Ratio of oracle to total documents."""
        total = len(self.oracle_documents) + len(self.distractor_documents)
        return len(self.oracle_documents) / total if total > 0 else 1.0
    
    def to_training_format(self) -> dict[str, str]:
        """Convert to instruction/input/output training format.
        
        The context is formatted as:
        [CONTEXT]
        Document 1: <code>
        Document 2: <code>
        ...
        [/CONTEXT]
        
        [QUESTION]
        <question>
        [/QUESTION]
        
        The output includes chain-of-thought reasoning followed by answer.
        """
        # Format context (shuffle oracle and distractor)
        import random
        all_docs = self.oracle_documents + self.distractor_documents
        random.shuffle(all_docs)
        
        context_parts = []
        for i, doc in enumerate(all_docs, 1):
            if hasattr(doc, 'source_code'):
                code = doc.source_code
                name = doc.name
            elif isinstance(doc, dict):
                code = doc.get('source_code', '')
                name = doc.get('name', f'Document {i}')
            else:
                code = str(doc)
                name = f'Document {i}'
            
            context_parts.append(f"### Document {i}: {name}\n```\n{code}\n```")
        
        context = "\n\n".join(context_parts)
        
        instruction = (
            "You are answering questions about a codebase. "
            "Use ONLY the provided documents to answer. "
            "First reason through the documents, then provide your answer."
        )
        
        input_text = f"[CONTEXT]\n{context}\n[/CONTEXT]\n\n[QUESTION]\n{self.question}\n[/QUESTION]"
        
        output_text = f"[REASONING]\n{self.reasoning}\n[/REASONING]\n\n[ANSWER]\n{self.final_answer}\n[/ANSWER]"
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "difficulty": self.difficulty.value,
            "question_type": self.question_type.value,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "question": self.question,
            "question_type": self.question_type.value,
            "oracle_documents": [
                d.to_dict() if hasattr(d, 'to_dict') else d 
                for d in self.oracle_documents
            ],
            "distractor_documents": [
                d.to_dict() if hasattr(d, 'to_dict') else d 
                for d in self.distractor_documents
            ],
            "reasoning": self.reasoning,
            "final_answer": self.final_answer,
            "difficulty": self.difficulty.value,
            "metadata": self.metadata,
        }


class RAFTConfig(BaseModel):
    """Configuration for RAFT data generation.
    
    Attributes:
        questions_per_block: Number of questions to generate per code block.
        num_oracle_docs: Number of oracle documents per example.
        easy_distractors: Number of distractors for easy examples.
        medium_distractors: Number of distractors for medium examples.
        hard_distractors: Number of distractors for hard examples.
        difficulty_distribution: Distribution of difficulty levels.
        use_embeddings: Whether to use embedding-based retrieval.
        embedding_model: Model for computing embeddings.
        max_context_tokens: Maximum tokens in context.
    """
    
    questions_per_block: int = Field(default=5, ge=1, le=10)
    num_oracle_docs: int = Field(default=1, ge=1, le=5)
    easy_distractors: int = Field(default=0, ge=0, le=5)
    medium_distractors: int = Field(default=2, ge=0, le=5)
    hard_distractors: int = Field(default=4, ge=0, le=10)
    difficulty_distribution: dict[str, float] = Field(
        default={"easy": 0.3, "medium": 0.5, "hard": 0.2}
    )
    use_embeddings: bool = Field(default=True)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    max_context_tokens: int = Field(default=4096)
    include_reasoning: bool = Field(default=True)
    

@dataclass
class SimilarityScore:
    """Similarity score between documents.
    
    Attributes:
        block_index: Index of the code block.
        score: Similarity score (0-1).
        is_oracle: Whether this is an oracle document.
    """
    
    block_index: int
    score: float
    is_oracle: bool = False
    
    def __lt__(self, other: "SimilarityScore") -> bool:
        return self.score < other.score


class RAFTDataset(BaseModel):
    """A collection of RAFT training examples.
    
    Attributes:
        examples: List of RAFT examples.
        config: Configuration used to generate.
        statistics: Dataset statistics.
    """
    
    examples: list[dict[str, Any]] = Field(default_factory=list)
    config: Optional[RAFTConfig] = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    
    def add_example(self, example: RAFTExample) -> None:
        """Add an example to the dataset."""
        self.examples.append(example.to_dict())
    
    def to_training_format(self) -> list[dict[str, str]]:
        """Convert all examples to training format."""
        training_data = []
        
        for ex_dict in self.examples:
            example = RAFTExample(
                question=ex_dict["question"],
                question_type=QuestionType(ex_dict["question_type"]),
                oracle_documents=ex_dict["oracle_documents"],
                distractor_documents=ex_dict["distractor_documents"],
                reasoning=ex_dict["reasoning"],
                final_answer=ex_dict["final_answer"],
                difficulty=Difficulty(ex_dict["difficulty"]),
            )
            training_data.append(example.to_training_format())
        
        return training_data
    
    def compute_statistics(self) -> dict[str, Any]:
        """Compute dataset statistics."""
        stats = {
            "total_examples": len(self.examples),
            "by_difficulty": {d.value: 0 for d in Difficulty},
            "by_question_type": {q.value: 0 for q in QuestionType},
            "avg_oracle_docs": 0,
            "avg_distractor_docs": 0,
        }
        
        total_oracle = 0
        total_distractor = 0
        
        for ex in self.examples:
            stats["by_difficulty"][ex["difficulty"]] += 1
            stats["by_question_type"][ex["question_type"]] += 1
            total_oracle += len(ex["oracle_documents"])
            total_distractor += len(ex["distractor_documents"])
        
        if self.examples:
            stats["avg_oracle_docs"] = total_oracle / len(self.examples)
            stats["avg_distractor_docs"] = total_distractor / len(self.examples)
        
        self.statistics = stats
        return stats
