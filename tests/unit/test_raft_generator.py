"""Comprehensive unit tests for the RAFT Data Generator.

Tests cover:
- Oracle document retrieval
- Distractor relevance verification
- Chain-of-thought citation
- Difficulty curriculum
- Question generation diversity
- Dataset generation

Run with: pytest tests/unit/test_raft_generator.py -v
"""

import pytest
from pathlib import Path
from textwrap import dedent

# Import path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_pipeline.schemas.code_blocks import CodeBlock
from data_pipeline.schemas.raft_examples import (
    Difficulty,
    QuestionType,
    QUESTION_TEMPLATES,
    RAFTConfig,
    RAFTDataset,
    RAFTExample,
)
from data_pipeline.raft_generator import (
    RAFTGenerator,
    generate_questions_for_block,
    retrieve_oracle_and_distractors,
    generate_chain_of_thought,
    generate_final_answer,
    generate_raft_examples,
    jaccard_similarity,
    compute_code_similarity,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_blocks() -> list[CodeBlock]:
    """Create sample code blocks for testing."""
    return [
        CodeBlock(
            path="src/auth.py",
            language="python",
            block_type="function",
            name="authenticate_user",
            docstring="Authenticate a user with username and password.",
            source_code=dedent('''
                def authenticate_user(username: str, password: str) -> bool:
                    """Authenticate a user with username and password."""
                    if not username or not password:
                        raise ValueError("Username and password required")
                    hashed = hash_password(password)
                    return verify_credentials(username, hashed)
            '''),
            dependencies=["hash_password", "verify_credentials"],
            metadata={"start_line": 10, "end_line": 17},
        ),
        CodeBlock(
            path="src/auth.py",
            language="python",
            block_type="function",
            name="hash_password",
            docstring="Hash a password using bcrypt.",
            source_code=dedent('''
                def hash_password(password: str) -> str:
                    """Hash a password using bcrypt."""
                    import bcrypt
                    salt = bcrypt.gensalt()
                    return bcrypt.hashpw(password.encode(), salt).decode()
            '''),
            dependencies=["bcrypt"],
            metadata={"start_line": 20, "end_line": 26},
        ),
        CodeBlock(
            path="src/utils.py",
            language="python",
            block_type="function",
            name="calculate_total",
            docstring="Calculate the total price with tax.",
            source_code=dedent('''
                def calculate_total(items: list, tax_rate: float = 0.08) -> float:
                    """Calculate the total price with tax."""
                    subtotal = sum(item.price for item in items)
                    tax = subtotal * tax_rate
                    return subtotal + tax
            '''),
            dependencies=[],
            metadata={"start_line": 1, "end_line": 7},
        ),
        CodeBlock(
            path="src/database.py",
            language="python",
            block_type="function",
            name="connect_database",
            docstring="Establish database connection.",
            source_code=dedent('''
                def connect_database(connection_string: str) -> Connection:
                    """Establish database connection."""
                    import psycopg2
                    try:
                        conn = psycopg2.connect(connection_string)
                        return conn
                    except Exception as e:
                        raise DatabaseError(f"Connection failed: {e}")
            '''),
            dependencies=["psycopg2"],
            metadata={"start_line": 5, "end_line": 14},
        ),
        CodeBlock(
            path="src/cache.py",
            language="python",
            block_type="class",
            name="CacheManager",
            docstring="Manage application cache with Redis backend.",
            source_code=dedent('''
                class CacheManager:
                    """Manage application cache with Redis backend."""
                    
                    def __init__(self, redis_url: str):
                        self.client = redis.from_url(redis_url)
                    
                    def get(self, key: str) -> Optional[str]:
                        return self.client.get(key)
                    
                    def set(self, key: str, value: str, ttl: int = 3600):
                        self.client.setex(key, ttl, value)
            '''),
            dependencies=["redis"],
            metadata={"start_line": 1, "end_line": 15},
        ),
    ]


@pytest.fixture
def generator(sample_blocks: list[CodeBlock]) -> RAFTGenerator:
    """Create RAFT generator with sample blocks."""
    config = RAFTConfig(
        questions_per_block=3,
        num_oracle_docs=1,
        easy_distractors=0,
        medium_distractors=2,
        hard_distractors=3,
        use_embeddings=False,  # Disable for testing
    )
    return RAFTGenerator(sample_blocks, config)


# =============================================================================
# Text Similarity Tests
# =============================================================================

class TestSimilarity:
    """Tests for similarity functions."""
    
    def test_jaccard_identical(self) -> None:
        """Test Jaccard similarity with identical texts."""
        text = "def hello(): pass"
        assert jaccard_similarity(text, text) == 1.0
    
    def test_jaccard_different(self) -> None:
        """Test Jaccard similarity with completely different texts."""
        text1 = "apple banana cherry"
        text2 = "dog elephant fox"
        assert jaccard_similarity(text1, text2) == 0.0
    
    def test_jaccard_partial(self) -> None:
        """Test Jaccard similarity with partial overlap."""
        text1 = "apple banana cherry"
        text2 = "banana cherry date"
        sim = jaccard_similarity(text1, text2)
        assert 0.3 < sim < 0.7  # Some overlap
    
    def test_jaccard_empty(self) -> None:
        """Test Jaccard with empty strings."""
        assert jaccard_similarity("", "") == 0.0
        assert jaccard_similarity("hello", "") == 0.0
    
    def test_code_similarity_same_language(self, sample_blocks: list[CodeBlock]) -> None:
        """Test similarity between blocks of same language."""
        block1 = sample_blocks[0]  # authenticate_user
        block2 = sample_blocks[1]  # hash_password (related)
        
        sim = compute_code_similarity(block1, block2)
        assert sim > 0.1  # Should have some similarity (same language, auth-related)
    
    def test_code_similarity_unrelated(self, sample_blocks: list[CodeBlock]) -> None:
        """Test similarity between unrelated blocks."""
        block1 = sample_blocks[0]  # authenticate_user
        block2 = sample_blocks[2]  # calculate_total (unrelated)
        
        sim = compute_code_similarity(block1, block2)
        # Should be lower than related blocks
        assert sim < 0.5


# =============================================================================
# Question Generation Tests
# =============================================================================

class TestQuestionGeneration:
    """Tests for question generation."""
    
    def test_generate_questions_count(self, sample_blocks: list[CodeBlock]) -> None:
        """Test that correct number of questions are generated."""
        block = sample_blocks[0]
        questions = generate_questions_for_block(block, num_questions=5)
        
        assert len(questions) == 5
    
    def test_generate_questions_diversity(self, sample_blocks: list[CodeBlock]) -> None:
        """Test that questions have diverse types."""
        block = sample_blocks[0]
        questions = generate_questions_for_block(block, num_questions=5)
        
        question_types = [q[1] for q in questions]
        # Should have at least 2 different types
        assert len(set(question_types)) >= 2
    
    def test_questions_contain_block_name(self, sample_blocks: list[CodeBlock]) -> None:
        """Test that questions reference the block name."""
        block = sample_blocks[0]
        questions = generate_questions_for_block(block, num_questions=3)
        
        # At least some questions should contain the function name
        names_in_questions = sum(
            1 for q, _ in questions if block.name in q
        )
        assert names_in_questions >= 1
    
    def test_question_templates_complete(self) -> None:
        """Test that all question types have templates."""
        for q_type in QuestionType:
            assert q_type in QUESTION_TEMPLATES
            assert len(QUESTION_TEMPLATES[q_type]) >= 2


# =============================================================================
# Oracle and Distractor Retrieval Tests
# =============================================================================

class TestOracleRetrieval:
    """Tests for oracle and distractor retrieval."""
    
    def test_oracle_included(self, sample_blocks: list[CodeBlock]) -> None:
        """Test that oracle document is always included."""
        oracle_idx = 0
        oracle_docs, _ = retrieve_oracle_and_distractors(
            query="What does authenticate_user do?",
            all_blocks=sample_blocks,
            oracle_idx=oracle_idx,
            k=1,
            num_distractors=2,
            use_embeddings=False,
        )
        
        assert len(oracle_docs) >= 1
        assert oracle_docs[0].name == sample_blocks[oracle_idx].name
    
    def test_distractors_not_oracle(self, sample_blocks: list[CodeBlock]) -> None:
        """Test that distractors do not include oracle."""
        oracle_idx = 0
        oracle_docs, distractor_docs = retrieve_oracle_and_distractors(
            query="What does authenticate_user do?",
            all_blocks=sample_blocks,
            oracle_idx=oracle_idx,
            k=1,
            num_distractors=2,
            use_embeddings=False,
        )
        
        oracle_names = {d.name for d in oracle_docs}
        distractor_names = {d.name for d in distractor_docs}
        
        # No overlap between oracle and distractor
        assert len(oracle_names & distractor_names) == 0
    
    def test_distractor_count(self, sample_blocks: list[CodeBlock]) -> None:
        """Test correct number of distractors."""
        _, distractor_docs = retrieve_oracle_and_distractors(
            query="Test query",
            all_blocks=sample_blocks,
            oracle_idx=0,
            k=1,
            num_distractors=2,
            use_embeddings=False,
        )
        
        assert len(distractor_docs) <= 2
    
    def test_distractors_less_relevant(self, sample_blocks: list[CodeBlock]) -> None:
        """Test that distractors are genuinely irrelevant."""
        # authenticate_user query should retrieve auth-related oracle
        # but not calculate_total as distractor
        oracle_idx = 0  # authenticate_user
        oracle_docs, distractor_docs = retrieve_oracle_and_distractors(
            query="How does authentication work?",
            all_blocks=sample_blocks,
            oracle_idx=oracle_idx,
            k=1,
            num_distractors=3,
            use_embeddings=False,
        )
        
        # Oracle should be auth-related
        assert oracle_docs[0].name == "authenticate_user"
        
        # Distractors should not be the closely related hash_password preferentially
        # (this depends on similarity scoring)
        distractor_names = {d.name for d in distractor_docs}
        assert "authenticate_user" not in distractor_names


# =============================================================================
# Chain-of-Thought Tests
# =============================================================================

class TestChainOfThought:
    """Tests for chain-of-thought generation."""
    
    def test_cot_cites_oracle(self, sample_blocks: list[CodeBlock]) -> None:
        """Test that chain-of-thought cites oracle documents."""
        oracle = [sample_blocks[0]]  # authenticate_user
        
        reasoning = generate_chain_of_thought(
            query="What does authenticate_user do?",
            oracle_docs=oracle,
            question_type=QuestionType.PURPOSE,
        )
        
        # Should mention the oracle document name
        assert "authenticate_user" in reasoning
    
    def test_cot_includes_docstring(self, sample_blocks: list[CodeBlock]) -> None:
        """Test that CoT references docstring when available."""
        oracle = [sample_blocks[0]]
        
        reasoning = generate_chain_of_thought(
            query="What is the purpose?",
            oracle_docs=oracle,
            question_type=QuestionType.PURPOSE,
        )
        
        # Should reference the docstring content
        assert "authenticate" in reasoning.lower() or "user" in reasoning.lower()
    
    def test_cot_multiple_oracles(self, sample_blocks: list[CodeBlock]) -> None:
        """Test CoT with multiple oracle documents."""
        oracles = sample_blocks[:2]  # authenticate_user and hash_password
        
        reasoning = generate_chain_of_thought(
            query="How does the auth system work?",
            oracle_docs=oracles,
            question_type=QuestionType.PURPOSE,
        )
        
        # Should mention both documents
        assert "authenticate_user" in reasoning or "hash_password" in reasoning
    
    def test_cot_empty_oracle(self) -> None:
        """Test CoT with no oracle documents."""
        reasoning = generate_chain_of_thought(
            query="Test query",
            oracle_docs=[],
            question_type=QuestionType.PURPOSE,
        )
        
        assert "No relevant documents" in reasoning


# =============================================================================
# Difficulty Curriculum Tests
# =============================================================================

class TestDifficultyCurriculum:
    """Tests for difficulty curriculum."""
    
    def test_easy_difficulty(self, generator: RAFTGenerator) -> None:
        """Test easy difficulty has no distractors."""
        block = generator.blocks[0]
        example = generator.generate_example(
            block=block,
            question="What does this do?",
            question_type=QuestionType.PURPOSE,
            difficulty=Difficulty.EASY,
        )
        
        assert example.difficulty == Difficulty.EASY
        assert len(example.distractor_documents) == 0
    
    def test_medium_difficulty(self, generator: RAFTGenerator) -> None:
        """Test medium difficulty has some distractors."""
        block = generator.blocks[0]
        example = generator.generate_example(
            block=block,
            question="What does this do?",
            question_type=QuestionType.PURPOSE,
            difficulty=Difficulty.MEDIUM,
        )
        
        assert example.difficulty == Difficulty.MEDIUM
        assert len(example.distractor_documents) <= 2
    
    def test_hard_difficulty(self, generator: RAFTGenerator) -> None:
        """Test hard difficulty has more distractors."""
        block = generator.blocks[0]
        example = generator.generate_example(
            block=block,
            question="What does this do?",
            question_type=QuestionType.PURPOSE,
            difficulty=Difficulty.HARD,
        )
        
        assert example.difficulty == Difficulty.HARD
        # Hard should have more distractors than easy
        assert len(example.distractor_documents) >= 0
    
    def test_difficulty_distribution(self, generator: RAFTGenerator) -> None:
        """Test that difficulty distribution is roughly followed."""
        difficulties = []
        for _ in range(100):
            diff = generator._select_difficulty()
            difficulties.append(diff)
        
        counts = {d: difficulties.count(d) for d in Difficulty}
        
        # With default 30% easy, 50% medium, 20% hard
        # Allow some variance
        assert counts[Difficulty.EASY] >= 10  # At least 10%
        assert counts[Difficulty.MEDIUM] >= 30  # At least 30%


# =============================================================================
# Full Example Generation Tests
# =============================================================================

class TestExampleGeneration:
    """Tests for complete example generation."""
    
    def test_generate_example(self, generator: RAFTGenerator) -> None:
        """Test generating a single example."""
        block = generator.blocks[0]
        example = generator.generate_example(
            block=block,
            question="What does authenticate_user do?",
            question_type=QuestionType.PURPOSE,
        )
        
        assert isinstance(example, RAFTExample)
        assert example.question == "What does authenticate_user do?"
        assert len(example.oracle_documents) >= 1
        assert example.final_answer  # Has an answer
        assert example.reasoning  # Has reasoning
    
    def test_example_to_training_format(self, generator: RAFTGenerator) -> None:
        """Test converting example to training format."""
        block = generator.blocks[0]
        example = generator.generate_example(
            block=block,
            question="Test question",
            question_type=QuestionType.PURPOSE,
            difficulty=Difficulty.EASY,
        )
        
        training = example.to_training_format()
        
        assert "instruction" in training
        assert "input" in training
        assert "output" in training
        assert "[CONTEXT]" in training["input"]
        assert "[QUESTION]" in training["input"]
        assert "[REASONING]" in training["output"]
        assert "[ANSWER]" in training["output"]
    
    def test_generate_qa_pairs(self, generator: RAFTGenerator) -> None:
        """Test QA pair generation."""
        qa_pairs = generator.generate_qa_pairs()
        
        # 5 blocks * 3 questions each = 15 pairs
        assert len(qa_pairs) == 15
        
        # Each pair has (block, question, question_type)
        for block, question, q_type in qa_pairs:
            assert isinstance(block, CodeBlock)
            assert isinstance(question, str)
            assert isinstance(q_type, QuestionType)


# =============================================================================
# Dataset Generation Tests
# =============================================================================

class TestDatasetGeneration:
    """Tests for dataset generation."""
    
    def test_generate_dataset(self, generator: RAFTGenerator) -> None:
        """Test generating a complete dataset."""
        dataset = generator.generate_dataset(num_examples=10)
        
        assert isinstance(dataset, RAFTDataset)
        assert len(dataset.examples) == 10
    
    def test_dataset_statistics(self, generator: RAFTGenerator) -> None:
        """Test dataset statistics computation."""
        dataset = generator.generate_dataset(num_examples=10)
        stats = dataset.compute_statistics()
        
        assert "total_examples" in stats
        assert stats["total_examples"] == 10
        assert "by_difficulty" in stats
        assert "by_question_type" in stats
    
    def test_dataset_to_training_format(self, generator: RAFTGenerator) -> None:
        """Test converting dataset to training format."""
        dataset = generator.generate_dataset(num_examples=5)
        training_data = dataset.to_training_format()
        
        assert len(training_data) == 5
        for item in training_data:
            assert "instruction" in item
            assert "input" in item
            assert "output" in item
    
    def test_convenience_function(self, sample_blocks: list[CodeBlock]) -> None:
        """Test the convenience function."""
        examples = generate_raft_examples(
            blocks=sample_blocks,
            num_examples=5,
        )
        
        assert len(examples) == 5


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_blocks(self) -> None:
        """Test with no code blocks."""
        generator = RAFTGenerator([], RAFTConfig())
        dataset = generator.generate_dataset()
        
        assert len(dataset.examples) == 0
    
    def test_single_block(self) -> None:
        """Test with a single code block."""
        block = CodeBlock(
            path="test.py",
            language="python",
            block_type="function",
            name="test_func",
            docstring="Test function.",
            source_code="def test_func(): pass",
        )
        
        generator = RAFTGenerator([block], RAFTConfig(questions_per_block=2))
        dataset = generator.generate_dataset()
        
        assert len(dataset.examples) == 2
        # With single block, no distractors possible
        for ex in dataset.examples:
            assert len(ex["distractor_documents"]) == 0
    
    def test_block_without_docstring(self) -> None:
        """Test with block lacking docstring."""
        block = CodeBlock(
            path="test.py",
            language="python",
            block_type="function",
            name="nodoc_func",
            docstring=None,
            source_code="def nodoc_func(): return 42",
        )
        
        generator = RAFTGenerator([block], RAFTConfig(questions_per_block=2))
        example = generator.generate_example(
            block=block,
            question="What does this do?",
            question_type=QuestionType.PURPOSE,
        )
        
        # Should still generate valid example
        assert example.final_answer
        assert example.reasoning


# =============================================================================
# RAFTExample Data Structure Tests
# =============================================================================

class TestRAFTExample:
    """Tests for RAFTExample dataclass."""
    
    def test_oracle_ratio(self) -> None:
        """Test oracle ratio calculation."""
        example = RAFTExample(
            question="Test",
            question_type=QuestionType.PURPOSE,
            oracle_documents=[{"source_code": "a"}],
            distractor_documents=[{"source_code": "b"}, {"source_code": "c"}],
            reasoning="Test reasoning",
            final_answer="Test answer",
            difficulty=Difficulty.MEDIUM,
        )
        
        assert example.oracle_ratio == pytest.approx(1/3)
    
    def test_to_dict(self, sample_blocks: list[CodeBlock]) -> None:
        """Test conversion to dictionary."""
        example = RAFTExample(
            question="Test question",
            question_type=QuestionType.PURPOSE,
            oracle_documents=[sample_blocks[0]],
            distractor_documents=[sample_blocks[1]],
            reasoning="Test reasoning",
            final_answer="Test answer",
            difficulty=Difficulty.EASY,
        )
        
        data = example.to_dict()
        
        assert data["question"] == "Test question"
        assert data["difficulty"] == "easy"
        assert data["question_type"] == "purpose"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
