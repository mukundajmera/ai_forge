Production-Grade Antigravity Prompts: AI Forge Implementation
Complete Phase-by-Phase Prompts for SOTA LLM Fine-Tuning Service
Document Version: 1.0 (January 2026)
Target: Google Antigravity + Opul Model
Architecture: PiSSA + QLoRA + RAG + Ollama on Mac
Status: Production-Ready Prompts for Autonomous Code Generation

PHASE 1: PROJECT ARCHITECTURE & SCAFFOLDING
Prompt 1.1: Foundation Setup & Project Structure
text
Agent Mission: Architect the "AI Forge" Project Structure

You are an expert DevOps architect specializing in ML systems. Your task is to create a 
production-grade project scaffold for a Local LLM Fine-Tuning Service that runs on Mac (Apple Silicon).

REQUIREMENTS:
1. Create a monorepo structure that separates concerns: data pipeline, training engine, 
   validation layer, API service, Ollama orchestration, monitoring, and testing.

2. Initialize these directories with appropriate configs:
   ├── ai_forge/
   │   ├── config/
   │   │   ├── defaults.yaml           # Global configuration
   │   │   ├── models.yaml             # Model registry
   │   │   └── pissa_config.yaml       # PiSSA-specific tuning
   │   ├── data_pipeline/              # Data extraction & chunking
   │   │   ├── miner.py                # AST-based code parser (Tree-sitter)
   │   │   ├── raft_generator.py       # RAFT training data synthesizer
   │   │   ├── validator.py            # Data quality checks
   │   │   └── schemas/
   │   ├── training/                   # Fine-tuning engine
   │   │   ├── forge.py                # Unsloth-MLX trainer (PiSSA + QLoRA)
   │   │   ├── callbacks/
   │   │   │   ├── metrics_logger.py
   │   │   │   ├── early_stopping.py
   │   │   │   └── memory_monitor.py
   │   │   └── losses/
   │   │       ├── dpo_loss.py         # Direct Preference Optimization
   │   │       └── raft_loss.py        # RAFT-specific loss
   │   ├── judge/                      # Validation & export
   │   │   ├── evaluator.py            # CodeBLEU, HumanEval, Perplexity
   │   │   ├── exporter.py             # GGUF conversion pipeline
   │   │   └── benchmarks/
   │   ├── conductor/                  # API & Ollama orchestration
   │   │   ├── service.py              # FastAPI service
   │   │   ├── ollama_manager.py       # Ollama interaction
   │   │   └── job_queue.py            # Async task management
   │   ├── antigravity_agent/          # Agentic orchestration
   │   │   ├── repo_guardian.py        # Mission Control agent
   │   │   ├── skills.yaml             # Skills definition
   │   │   └── artifacts_templates/
   │   ├── tests/
   │   │   ├── unit/                   # Unit tests for each module
   │   │   ├── integration/            # E2E pipeline tests
   │   │   └── fixtures/
   │   ├── docs/
   │   │   ├── architecture.md
   │   │   ├── api_reference.md
   │   │   └── user_guide.md
   │   ├── requirements.txt             # Python dependencies
   │   ├── pyproject.toml               # Modern Python packaging
   │   ├── Dockerfile                   # Optional container support
   │   ├── .env.example                 # Config template
   │   └── README.md                    # Main documentation

3. Create stub files with docstrings and type hints (all functions should be typed).

4. Generate a dependency graph showing how modules interact (mermaid diagram).

5. Create a comprehensive README explaining:
   - Project purpose & architecture
   - Hardware requirements (M1/M2/M3/M4 specs)
   - Getting started guide
   - Key design decisions (why PiSSA over LoRA, why Tree-sitter for AST, etc.)
   - Folder structure explanation

OUTPUT FORMAT:
- Artifact 1: Full directory tree (text)
- Artifact 2: pyproject.toml with all dependencies (TOML)
- Artifact 3: Mermaid dependency diagram (embedded in README)
- Artifact 4: requirements.txt (Python dependencies)
- Artifact 5: Comprehensive README.md with architecture overview
- Artifact 6: .env.example with all configuration variables
- Code Files: Scaffold all stub modules with proper structure

RESEARCH CONTEXT:
From "Local-LLM-Fine-Tuning-for-Projects.pdf":
- System uses PiSSA (3-5x faster convergence than LoRA, +5.16% accuracy)
- QLoRA enables 4-bit quantization on Mac (8GB-16GB RAM viable)
- Unsloth-MLX as bridge for MLX + Unsloth optimizations (80% memory savings)
- Tree-sitter for semantic AST-based code chunking (prevent mid-function splits)
- RAFT (Retrieval-Augmented Fine-Tuning) for hybrid RAG+FT data synthesis
- Ollama as deployment target (GGUF format, local inference)
- Google Antigravity as agentic orchestrator with Mission Control interface

ACCEPTANCE CRITERIA:
✅ All directories created with proper nesting
✅ All stub modules have complete docstrings and type hints
✅ Dependencies reflect current SOTA (Unsloth-MLX, MLX, tree-sitter, FastAPI, etc.)
✅ Documentation explains reasoning behind each architectural choice
✅ Project structure follows Python best practices (src layout, tests separation)
✅ README includes quick-start instructions
✅ Environment variables documented
PHASE 2: DATA PIPELINE - AST-BASED CODE EXTRACTION & RAFT SYNTHESIS
Prompt 2.1: Semantic Code Chunker (Tree-sitter AST Parser)
text
Agent Mission: Implement Production-Grade Semantic Code Chunking

You are an expert in code analysis and abstract syntax trees. Implement a semantic code 
chunker that uses Tree-sitter to parse code into complete logical units, preventing 
destructive mid-function splits.

REQUIREMENTS:

1. Implement ai_forge/data_pipeline/miner.py with the following functions:

   a) parse_repository(repo_path: str) -> List[CodeBlock]:
      - Recursively traverses repo, identifies all code files (.py, .js, .ts, .go, etc.)
      - For each file, uses Tree-sitter to build AST
      - Extracts complete semantic units (functions, classes, modules)
      - Returns structured CodeBlock objects (path, type, language, content, docstring)

   b) extract_functions(code_bytes: bytes, language: str) -> List[Tuple[str, str, str]]:
      - Parses code using language-specific Tree-sitter parser
      - Walks AST to find function/method nodes
      - Returns (function_name, docstring, full_source_code)
      - Filters out private/test functions unless explicitly included

   c) extract_classes(code_bytes: bytes, language: str) -> List[Tuple[str, str, List[str]]]:
      - Extracts class definitions and their methods
      - Returns (class_name, docstring, method_names)

   d) extract_module_info(code_bytes: bytes) -> Dict[str, Any]:
      - Extracts module-level imports, constants, dependencies
      - Returns metadata for context enrichment

2. Data structures (define in data_pipeline/schemas/):

   class CodeBlock:
       path: str                    # Relative path to file
       language: str               # "python", "javascript", etc.
       block_type: str            # "function", "class", "module"
       name: str                  # Function/class name
       docstring: str             # Extracted docstring
       source_code: str           # Full source (cleaned)
       dependencies: List[str]    # Referenced functions/classes
       metadata: Dict[str, Any]   # Additional info (line numbers, complexity)

3. Quality filters:
   - Minimum code length: 20 tokens
   - Maximum code length: 2048 tokens
   - Reject if: no docstring, trivial getters/setters, auto-generated code
   - Score each block 0-1 on quality (based on docstring length, code complexity)

4. Handle multiple languages:
   - Python: Use tree-sitter-python
   - JavaScript/TypeScript: Use tree-sitter-javascript
   - Go: Use tree-sitter-go
   - Java: Use tree-sitter-java

5. Comprehensive error handling:
   - Syntax errors in code files (skip gracefully)
   - Encoding issues (try multiple encodings)
   - Missing Tree-sitter parsers (fallback behavior or error message)

6. Unit tests (tests/unit/test_miner.py):
   - Test parsing of Python, JavaScript, Go code samples
   - Verify complete functions are extracted (not split)
   - Verify docstrings are correctly attached
   - Test edge cases: nested functions, decorators, lambdas, async functions

OUTPUT DELIVERABLES:
- Code: ai_forge/data_pipeline/miner.py (complete, production-ready)
- Schemas: ai_forge/data_pipeline/schemas/code_blocks.py (Pydantic models)
- Tests: tests/unit/test_miner.py (comprehensive test suite)
- Documentation: Docstrings + inline comments explaining AST traversal logic
- Artifact: Sample output showing extracted code blocks from a real repo

RESEARCH CONTEXT:
From PDF:
"The 'Miner' uses Tree-sitter to build an Abstract Syntax Tree (AST) of the code. It then 
walks the tree to extract complete syntactic units: entire functions, class definitions, or 
module docstrings. This ensures the model learns complete logical thoughts, not fragments."

ACCEPTANCE CRITERIA:
✅ Successfully parses 5+ languages
✅ Zero mid-function splits (verified by unit tests)
✅ Handles 100+ files efficiently (< 5 min for typical repo)
✅ Quality scoring implemented and filters out low-quality blocks
✅ 100% test coverage for miner module
✅ Comprehensive error handling with meaningful error messages
✅ Extracts and preserves docstrings
Prompt 2.2: RAFT Data Synthesis (Retrieval-Augmented Fine-Tuning)
text
Agent Mission: Implement RAFT Training Data Synthesis

You are an expert in curriculum learning and data synthesis. Implement a RAFT (Retrieval-
Augmented Fine-Tuning) data generator that creates training examples simulating the RAG 
inference process.

REQUIREMENTS:

1. Understand RAFT concept (from PDF):
   "RAFT involves fine-tuning the model not just on raw code, but on datasets that simulate 
   the RAG inference process. Training data consists of:
   1. Question: A query about the codebase
   2. Oracle Documents: The actual code snippets containing the answer
   3. Distractor Documents: Irrelevant code snippets
   4. Chain-of-Thought Answer: A reasoning path that explicitly cites Oracle Documents"

2. Implement ai_forge/data_pipeline/raft_generator.py with:

   a) generate_qa_pairs(code_blocks: List[CodeBlock]) -> List[RAFTExample]:
      - For each code block, generate 3-5 synthetic questions
      - Questions should probe understanding of: purpose, usage, edge cases, dependencies
      - Example: If code_block is an authentication function, questions could be:
        * "What does this function do?"
        * "How does it handle invalid credentials?"
        * "What exceptions might it raise?"

   b) retrieve_oracle_and_distractors(query: str, all_blocks: List[CodeBlock], 
                                       oracle_idx: int, k=3, num_distractors=2):
      - oracle_docs: The k most relevant code blocks (including oracle_idx)
      - distractor_docs: num_distractors irrelevant or weakly-relevant blocks
      - Uses embedding-based retrieval (CodeBERT) for relevance scoring

   c) generate_chain_of_thought(query: str, oracle_docs: List[CodeBlock]) -> str:
      - Creates a reasoning path that explicitly cites oracle documents
      - Format: "Looking at function X, we see it [does A]. Comparing with function Y, [does B]."
      - Ensures model learns to ground reasoning in provided context

3. Data structures:

   class RAFTExample:
       question: str                   # User query about codebase
       oracle_documents: List[CodeBlock]  # Ground truth code snippets
       distractor_documents: List[CodeBlock]  # Irrelevant context
       reasoning: str                  # Chain-of-thought explanation
       final_answer: str              # Concise answer to question
       difficulty: str                # "easy", "medium", "hard"

4. Synthetic question generation templates:
   - "What is the purpose of [function_name]?"
   - "How does [function_name] handle [edge_case]?"
   - "What are the dependencies of [function_name]?"
   - "Why does [function_name] use [specific_pattern]?"
   - "How would you extend [function_name] to [new_requirement]?"

5. Difficulty curriculum:
   - Easy: Single function, obvious oracle, 0 distractors
   - Medium: Related functions, 1-2 distractors, requires understanding relationships
   - Hard: Complex dependencies, 3+ distractors, requires deep reasoning

6. Unit tests (tests/unit/test_raft_generator.py):
   - Verify oracle documents are retrieved correctly
   - Verify distractors are genuinely irrelevant
   - Verify reasoning properly cites oracle documents
   - Test difficulty curriculum logic

OUTPUT DELIVERABLES:
- Code: ai_forge/data_pipeline/raft_generator.py
- Schemas: ai_forge/data_pipeline/schemas/raft_examples.py (Pydantic models)
- Tests: tests/unit/test_raft_generator.py
- Documentation: Explain RAFT concept and implementation
- Artifact: Sample RAFT training examples (3-5 examples showing question → oracle → reasoning)

RESEARCH CONTEXT:
From PDF: "By training on this data, the model learns to ignore noise (distractors) and 
ground its reasoning in the provided context (Oracle). This effectively combines the 
benefits of both worlds: the model learns the project's style (via Fine-Tuning) and learns 
how to use retrieved context effectively (via RAFT), significantly reducing hallucinations."

ACCEPTANCE CRITERIA:
✅ RAFT examples correctly pair queries with oracle documents
✅ Distractors are genuinely irrelevant (verified by inspection & tests)
✅ Chain-of-thought explanations properly cite oracle documents
✅ Difficulty curriculum generates all three difficulty levels
✅ Synthetic questions are diverse and probe different understanding dimensions
✅ 100% test coverage
✅ Generates 5+ examples per code block (thousands of training examples from typical repo)
Prompt 2.3: Data Validation & Quality Metrics
text
Agent Mission: Implement Data Validation & Quality Scoring Pipeline

You are an expert in data quality assurance for ML systems. Implement comprehensive 
validation and quality scoring for training data.

REQUIREMENTS:

1. Implement ai_forge/data_pipeline/validator.py with:

   a) validate_code_block(block: CodeBlock) -> Tuple[bool, List[str]]:
      - Checks: non-empty, syntactically valid, has docstring, reasonable length
      - Returns (is_valid, error_messages)

   b) validate_raft_example(example: RAFTExample) -> Tuple[bool, List[str]]:
      - Checks: oracle docs are relevant, distractors are irrelevant, reasoning is coherent
      - Uses embedding distance to verify quality heuristics

   c) score_data_quality(dataset: List[RAFTExample]) -> Dict[str, float]:
      - Overall quality score (0-1)
      - Per-dimension scores: relevance, diversity, difficulty_balance
      - Detects duplicates, near-duplicates

   d) generate_quality_report(dataset: List[RAFTExample]) -> str:
      - Markdown report with: dataset size, quality distribution, failure modes
      - Histogram of quality scores
      - Top N worst examples (for manual review)

2. Quality metrics:

   class DataQualityMetrics:
       total_examples: int
       valid_examples: int
       quality_score_mean: float
       quality_score_std: float
       difficulty_distribution: Dict[str, int]  # easy/medium/hard counts
       duplicate_rate: float
       avg_oracle_relevance: float
       avg_distractor_irrelevance: float
       failure_modes: List[str]

3. Filtering thresholds (configurable):
   - Min quality score: 0.6 (filter out low-quality examples)
   - Min oracle relevance: 0.75 (oracle docs must be truly relevant)
   - Max distractor relevance: 0.3 (distractors must be different enough)

4. Generate validation artifacts:
   - Quality histograms (Matplotlib)
   - Failure analysis report
   - Before/after stats (before vs after filtering)

5. Unit tests (tests/unit/test_validator.py):
   - Test validation of valid/invalid examples
   - Test quality scoring on edge cases
   - Test filtering removes genuinely low-quality data

OUTPUT DELIVERABLES:
- Code: ai_forge/data_pipeline/validator.py
- Metrics: ai_forge/data_pipeline/schemas/metrics.py (Pydantic models)
- Tests: tests/unit/test_validator.py
- Artifact: Sample quality report (HTML or Markdown with charts)

ACCEPTANCE CRITERIA:
✅ Validation catches all error modes (invalid code, missing docstrings, etc.)
✅ Quality scoring is calibrated (most examples score 0.7-0.9)
✅ Filtering removes genuinely problematic examples
✅ Quality report is actionable (identifies improvement opportunities)
✅ < 1 minute to validate dataset of 5000+ examples
PHASE 3: FINE-TUNING ENGINE - PISSA + QLORA TRAINER
Prompt 3.1: PiSSA + QLoRA Training Engine
text
Agent Mission: Implement Production-Grade PiSSA + QLoRA Trainer

You are an expert ML systems engineer specializing in efficient fine-tuning. Implement the 
training engine using Unsloth-MLX, PiSSA initialization, and 4-bit quantization.

REQUIREMENTS:

1. Core architecture (ai_forge/training/forge.py):

   class FineTuneTrainer:
       def __init__(self, config: FineTuneConfig):
           # Initialize base model (quantized, 4-bit)
           # Load Unsloth-MLX optimizations
           # Setup PiSSA adapter configuration
           
       def compute_pissa_init(self) -> Tuple[torch.Tensor, torch.Tensor]:
           """
           Compute SVD of base model weights to initialize PiSSA adapters.
           This is the key innovation: instead of random Gaussian init (LoRA),
           PiSSA initializes with principal singular components.
           
           Returns: (A_init, B_init) where W ≈ A @ B + W_residual
           """
           # For each target module:
           # 1. Extract weight matrix W
           # 2. Compute SVD: W = U @ S @ V^T
           # 3. A_init = U[:, :rank] @ sqrt(S[:rank])
           # 4. B_init = sqrt(S[:rank]) @ V[:, :rank]^T
           # Store W_residual in base model
       
       def train(self, train_dataset, eval_dataset, num_epochs=3):
           """
           Execute training loop with PiSSA + QLoRA.
           """
           # 1. Initialize PiSSA adapters (non-random initialization)
           # 2. Setup optimizer (AdamW 8-bit)
           # 3. Training loop with:
           #    - Gradient accumulation (4-8 steps)
           #    - Early stopping based on eval loss
           #    - Memory monitoring (alert if > 80% usage)
           #    - Loss visualization callback
           # 4. Save adapters and checkpoints

2. Configuration (ai_forge/config/pissa_config.yaml):

   pissa_config:
     # Initialization
     init_method: "pissa"  # vs "gaussian" for LoRA
     rank: 64              # PiSSA can use higher ranks stably
     lora_alpha: 128       # Scaling factor (typically 2x rank)
     
     # Quantization
     quantization_bits: 4
     quantization_type: "nf4"  # Native format for 4-bit
     
     # Training
     learning_rate: 2e-4
     optimizer: "adamw_8bit"
     batch_size: 2         # Mac optimized (8-16GB RAM)
     gradient_accumulation_steps: 4
     num_train_epochs: 3
     eval_strategy: "steps"
     eval_steps: 50
     save_steps: 100
     warmup_steps: 100
     max_grad_norm: 1.0
     
     # Model
     base_model: "unsloth/Llama-3.2-3B-Instruct-4bit"
     max_seq_length: 2048
     target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

3. Callbacks (ai_forge/training/callbacks/):

   a) MetricsLogger: Logs loss, accuracy, perplexity to file
   b) EarlyStopping: Halts training if eval loss doesn't improve
   c) MemoryMonitor: Alerts if memory usage exceeds threshold
   d) LossPlotter: Generates real-time loss curve (for Artifacts)

4. DPO Phase (Direct Preference Optimization):
   
   Optional second phase post-SFT:
   - Takes preference data (better_answer, worse_answer)
   - Optimizes model to prefer correct answers
   - Reduces hallucinations significantly
   
   Include but make optional

5. Unit tests (tests/unit/test_forge.py):
   - Test PiSSA initialization produces correct SVD factorization
   - Test QLoRA quantization reduces model size by 75%
   - Test training loop completes without OOM
   - Test checkpoints save/load correctly

OUTPUT DELIVERABLES:
- Code: ai_forge/training/forge.py (complete trainer class)
- Config: ai_forge/config/pissa_config.yaml
- Callbacks: ai_forge/training/callbacks/ (all callback classes)
- Tests: tests/unit/test_forge.py
- Documentation: Explain PiSSA math, QLoRA quantization, why these choices
- Artifact: Training config template with comments explaining each parameter

RESEARCH CONTEXT:
From PDF:
"PiSSA represents a significant theoretical and practical leap over standard Low-Rank 
Adaptation (LoRA). Instead of random initialization, it initializes adapter matrices using 
principal singular components. Benchmarks consistently indicate that PiSSA converges 3-5x 
faster than standard LoRA and achieves higher final accuracy (approximately +5.16% on code 
and reasoning benchmarks)."

"QLoRA (Quantized LoRA) freezes the base model in 4-bit precision and backpropagates 
gradients through to the LoRA adapters. When combined with PiSSA (termed QPiSSA), 
developers can fine-tune a 7B parameter model using less than 6GB of memory."

ACCEPTANCE CRITERIA:
✅ PiSSA SVD initialization correctly computed
✅ QLoRA quantization reduces model by 75%
✅ Training on 3B model completes in < 30 min on Mac M3
✅ Training on 7B model completes in < 60 min on Mac M3
✅ Memory usage stays < 12GB on 16GB unified memory Mac
✅ Early stopping works (halts if no improvement)
✅ All checkpoints save and load correctly
✅ DPO phase (optional) runs without errors
PHASE 4: VALIDATION & EXPORT
Prompt 4.1: Model Evaluation & Export Pipeline
text
Agent Mission: Implement Comprehensive Evaluation & GGUF Export Pipeline

You are an ML evaluation expert. Implement evaluation metrics (CodeBLEU, HumanEval, 
Perplexity) and export fine-tuned models to GGUF format.

REQUIREMENTS:

1. Implement ai_forge/judge/evaluator.py with:

   a) compute_perplexity(model, eval_dataset) -> float:
      - Standard perplexity metric (lower = better)
      - Tracks per-batch to detect divergence

   b) compute_codebleu(predictions, references) -> float:
      - CodeBLEU metric for code quality (0-1 scale)
      - Accounts for code structure, not just exact match

   c) run_humaneval_subset(model, num_samples=100) -> Dict[str, float]:
      - Runs subset of HumanEval benchmark
      - Returns: pass@1, pass@10, etc.

   d) evaluate_reconstruction(model, original_code_blocks) -> Dict[str, float]:
      - How well can the model reproduce its own training code?
      - Metric: edit distance, semantic similarity
      - High scores indicate good memorization of project patterns

   e) evaluate_hallucination_rate(model, raft_test_set) -> float:
      - On RAFT examples, how often does model invent code not in distractors?
      - Uses answer parsing to detect hallucinations

2. Export pipeline (ai_forge/judge/exporter.py):

   a) merge_adapters_to_base(model, adapter_path) -> PreTrainedModel:
      - Merges PiSSA adapters back into base model
      - Produces standalone model (no adapter needed at inference)

   b) convert_to_gguf(model, output_path, quantization="f16"):
      - Converts HF model to GGUF format (llama.cpp compatible)
      - Supports quantization levels: f16, q4_k_m, q5_k_m, q6_k
      - f16: High quality, larger file
      - q4_k_m: RECOMMENDED (good balance of quality/size/speed)
      - q6_k: Maximum quality

   c) create_ollama_modelfile(model_name, gguf_path, system_prompt) -> str:
      - Generates Ollama Modelfile with embedded system prompt
      - Sets temperature, top_k, top_p for code tasks
      - Returns Modelfile content

3. Evaluation report generation:

   class EvaluationReport:
       model_name: str
       base_model: str
       training_date: str
       num_training_examples: int
       num_eval_examples: int
       
       # Metrics
       perplexity: float
       codebleu_score: float
       humaneval_pass_rate: float
       reconstruction_score: float
       hallucination_rate: float
       
       # Comparisons
       improvement_over_base: Dict[str, float]  # % improvement vs base model
       
       # Artifacts
       loss_curve: bytes  # PNG
       metric_distribution: bytes  # PNG

4. Unit tests (tests/unit/test_evaluator.py):
   - Test perplexity computation
   - Test CodeBLEU scoring
   - Test GGUF export produces valid files
   - Test Modelfile generation is valid syntax

OUTPUT DELIVERABLES:
- Code: ai_forge/judge/evaluator.py & exporter.py
- Tests: tests/unit/test_evaluator.py
- Artifact: Sample evaluation report (HTML or Markdown)
- Artifact: Sample GGUF export with size/quality tradeoffs shown

ACCEPTANCE CRITERIA:
✅ Perplexity computed correctly (matches HF implementation)
✅ CodeBLEU scoring working (0-1 range)
✅ GGUF export produces valid files (verified with llama.cpp)
✅ Ollama Modelfile is syntactically valid
✅ Evaluation runs in < 5 minutes on test set
✅ Evaluation report includes comparison to base model
✅ Export produces files that work with Ollama
PHASE 5: API SERVICE & OLLAMA ORCHESTRATION
Prompt 5.1: FastAPI Service with Async Job Management
text
Agent Mission: Build Production FastAPI Service for LLM Fine-Tuning

You are a backend systems expert. Implement a production-grade REST API using FastAPI 
with async job management, OpenAI-compatible endpoints, and comprehensive error handling.

REQUIREMENTS:

1. API endpoints (ai_forge/conductor/service.py):

   POST /v1/fine-tune
     Request: {
       project_name: str,
       base_model: str,
       epochs: int,
       data_file: UploadFile
     }
     Response: { job_id: str, status: str }
     Queues async fine-tuning job

   GET /status/{job_id}
     Returns: { status: str, progress: float, metrics: Dict, error: str? }

   GET /models
     Returns: { available_models: List[str], active_model: str }

   POST /v1/chat/completions (OpenAI-compatible)
     Request: { model: str, messages: List, temperature: float }
     Response: { choices: [{ message: { content: str } }] }

   POST /v1/query
     Request: { prompt: str, model: str? }
     Response: { answer: str, metadata: Dict }

   POST /deploy/{job_id}
     Deploys trained model to Ollama as active model

   POST /validate/{job_id}
     Runs validation suite on trained model, returns report

2. Data structures:

   class FineTuneRequest(BaseModel):
       project_name: str
       base_model: str
       epochs: int = 3
       learning_rate: float = 2e-4
       rank: int = 64
       batch_size: int = 2

   class JobStatus(BaseModel):
       job_id: str
       status: str  # queued, training, validating, exporting, ready, failed
       progress: float  # 0-1
       started_at: str
       current_step: str
       error: Optional[str]

3. Job management:

   - Job queue (async with FastAPI BackgroundTasks)
   - Job persistence (JSON file or SQLite)
   - Automatic cleanup of old jobs
   - Concurrent job limits (1-2 training jobs max on Mac)

4. Error handling:

   - Comprehensive try-catch with meaningful error messages
   - Validation of inputs (model size checks, data file format)
   - Graceful handling of Ollama unavailability
   - Rate limiting (if needed)

5. Logging & monitoring:

   - All requests/responses logged
   - Job progress logged in real-time
   - Error stack traces captured
   - Prometheus metrics exported (optional)

6. Unit/Integration tests (tests/integration/test_service.py):
   - Test all endpoints with valid/invalid inputs
   - Test concurrent job handling
   - Test OpenAI compatibility
   - Test error cases

OUTPUT DELIVERABLES:
- Code: ai_forge/conductor/service.py
- Schemas: ai_forge/conductor/schemas.py
- Tests: tests/integration/test_service.py
- Artifact: OpenAPI documentation (auto-generated from FastAPI)
- Artifact: Example curl commands for all endpoints

ACCEPTANCE CRITERIA:
✅ All endpoints respond correctly
✅ Job queue handles 2+ concurrent jobs without errors
✅ OpenAI API compatibility verified (same request/response format)
✅ Error messages are clear and actionable
✅ Service starts/stops cleanly
✅ All endpoints have comprehensive tests
Prompt 5.2: Ollama Integration & Model Orchestration
text
Agent Mission: Implement Ollama Integration for Model Serving

You are a DevOps engineer. Implement comprehensive Ollama integration for serving fine-
tuned models locally.

REQUIREMENTS:

1. Ollama manager (ai_forge/conductor/ollama_manager.py):

   a) check_ollama_status() -> bool:
      - Verifies Ollama is running on localhost:11434
      - Returns True/False

   b) list_ollama_models() -> List[str]:
      - Calls `ollama list`, parses output
      - Returns model names available locally

   c) create_ollama_model(model_name: str, gguf_path: str, 
                          system_prompt: str) -> bool:
      - Creates Modelfile from template
      - Runs `ollama create model_name -f Modelfile`
      - Returns success/failure

   d) get_active_model() -> str:
      - Reads from config which model is "active"
      - Used for queries without explicit model param

   e) set_active_model(model_name: str) -> bool:
      - Updates config, makes model active
      - Used after successful deployment

   f) query_model(model_name: str, prompt: str, 
                  temperature: float = 0.7) -> str:
      - Calls `ollama run model_name prompt`
      - Returns generated text
      - Handles errors gracefully

2. Modelfile generation template:

   Modelfile template stored in: ai_forge/conductor/modelfile_template

   FROM {gguf_path}
   SYSTEM "{system_prompt}"
   PARAMETER temperature {temperature}
   PARAMETER top_k {top_k}
   PARAMETER top_p {top_p}
   PARAMETER num_predict 512

3. System prompts:

   - Default: General-purpose code assistant
   - Custom: Project-specific prompt (e.g., "You are expert in Django...")
   - Configured per model in config

4. Integration with FastAPI:

   - When training completes, call ollama_manager to create model
   - When user requests model swap, call set_active_model
   - When user sends query, call query_model

5. Error handling:

   - Ollama not installed → helpful error message
   - Ollama not running → try to start it or error
   - GGUF file missing → clear error
   - Model creation fails → log details, return error

6. Tests (tests/integration/test_ollama_manager.py):
   - Test model creation/deletion
   - Test query execution
   - Test error cases (missing ollama, missing gguf, etc.)

OUTPUT DELIVERABLES:
- Code: ai_forge/conductor/ollama_manager.py
- Template: ai_forge/conductor/modelfile_template
- Tests: tests/integration/test_ollama_manager.py
- Documentation: How Ollama integration works

ACCEPTANCE CRITERIA:
✅ Creates valid Ollama models
✅ Queries work correctly
✅ Error handling is robust
✅ Modelfile generation is correct
PHASE 6: ANTIGRAVITY AGENTIC ORCHESTRATION
Prompt 6.1: Repo Guardian Agent & Mission Control
text
Agent Mission: Implement Antigravity "Repo Guardian" Agentic Orchestrator

You are an agentic AI expert. Implement a specialized Antigravity agent that autonomously 
manages the LLM fine-tuning lifecycle.

REQUIREMENTS:

1. Repo Guardian Agent Definition (ai_forge/antigravity_agent/repo_guardian.py):

   class RepoGuardian:
       """
       Specialized agent persona: "You are a DevOps AI responsible for maintaining 
       the freshness and quality of the project's local LLM. Your role is to monitor 
       the repository for significant changes, trigger retraining when appropriate, 
       validate model quality, and manage deployments."
       """
       
       def __init__(self, project_config):
           # Initialize with project path, config, model registry
           
       def monitor_repository(self) -> Dict[str, Any]:
           """
           Called periodically (e.g., hourly). Checks:
           - Git commit count since last training
           - Number of files changed
           - Has critical paths changed (src/core, etc.)
           
           Returns: { should_retrain: bool, reason: str, metrics: Dict }
           """
       
       def plan_training_cycle(self) -> Dict[str, Any]:
           """
           Creates a structured training plan:
           1. Data extraction: identify repo changes
           2. Data validation: ensure quality
           3. Training: execute fine-tune
           4. Evaluation: run test suite
           5. Deployment: swap Ollama model
           
           Returns: { plan: List[Task], estimated_duration: float }
           """
       
       def execute_task(self, task: Task) -> Artifact:
           """
           Executes a single task (data extraction, training, etc.)
           Generates Artifacts for user inspection
           
           Returns: Artifact with results, logs, metrics
           """

2. Task types:

   - ExtractDataTask: Run data pipeline, generate training data
   - ValidateDataTask: Check data quality, generate quality report
   - TrainTask: Execute fine-tuning, log metrics in real-time
   - ValidateModelTask: Run evaluation suite
   - ExportTask: Convert to GGUF
   - DeployTask: Create Ollama model, set as active

3. Artifacts generated by agent:

   a) Training Dashboard:
      - Real-time loss curve (line plot, updates every batch)
      - Memory usage monitor
      - ETA counter
      - STOP button (halts training if diverging)

   b) Data Quality Report:
      - Dataset statistics (size, quality distribution)
      - Per-example quality scores (table)
      - Failure analysis
      - Recommendations for improvement

   c) Validation Report:
      - Comparison table: base model vs fine-tuned model
      - Metrics: perplexity, CodeBLEU, hallucination rate
      - Improvement percentages
      - Pass/Fail recommendation (deploy if metrics acceptable)

   d) Deployment Checklist:
      - Pre-deployment checks (model exists, compatible with Ollama)
      - Deployment log (what was deployed, when)
      - Rollback instructions if needed

4. Skill definitions (ai_forge/antigravity_agent/skills.yaml):

   skills:
     - name: "extract_data"
       description: "Extract semantic code chunks from repository"
       command: "python -m ai_forge.data_pipeline.miner"
       
     - name: "validate_data"
       description: "Validate training data quality"
       command: "python -m ai_forge.data_pipeline.validator"
       
     - name: "train_model"
       description: "Execute PiSSA+QLoRA fine-tuning"
       command: "python -m ai_forge.training.forge"
       
     - name: "evaluate_model"
       description: "Run comprehensive model evaluation"
       command: "python -m ai_forge.judge.evaluator"
       
     - name: "export_model"
       description: "Export model to GGUF"
       command: "python -m ai_forge.judge.exporter"
       
     - name: "deploy_model"
       description: "Deploy model to Ollama"
       command: "python -m ai_forge.conductor.ollama_manager"

5. Example Mission Control Prompt:

   "Agent, monitor the src/ directory for changes. When you detect:
   - 20+ files changed, OR
   - A git tag matching 'release/*'
   
   Execute the full training cycle:
   1. Extract data from recent commits
   2. Generate RAFT training examples
   3. Validate data quality (stop if < 0.6 avg quality)
   4. Train with PiSSA+QLoRA (display loss curve)
   5. Evaluate model (compare vs base)
   6. If metrics improve, deploy to Ollama
   7. Send summary report
   
   For each step, generate an Artifact showing progress."

6. Integration with FastAPI:
   - Agent can be triggered manually via /v1/retrain endpoint
   - Agent can auto-trigger based on repo monitoring
   - Agent can cancel running jobs if requested

7. Tests (tests/integration/test_repo_guardian.py):
   - Test repo monitoring detects changes correctly
   - Test training cycle executes in correct order
   - Test artifacts are generated
   - Test error handling (e.g., if training fails, agent retries or notifies)

OUTPUT DELIVERABLES:
- Code: ai_forge/antigravity_agent/repo_guardian.py
- Config: ai_forge/antigravity_agent/skills.yaml
- Prompts: ai_forge/antigravity_agent/prompts.txt (mission control examples)
- Tests: tests/integration/test_repo_guardian.py
- Documentation: Agent architecture and capabilities
- Artifact: Sample training dashboard, quality report, validation report

ACCEPTANCE CRITERIA:
✅ Agent monitors repository correctly
✅ Agent executes full training cycle autonomously
✅ Artifacts are generated and informative
✅ Agent handles errors gracefully
✅ Manual triggers work via FastAPI endpoint
✅ Agent can be paused/resumed
PHASE 7: COMPREHENSIVE TESTING & CI/CD
Prompt 7.1: Complete Test Suite & Verification
text
Agent Mission: Implement Comprehensive Test Suite with E2E Verification

You are a QA expert. Implement unit tests, integration tests, and end-to-end tests 
covering all system components.

REQUIREMENTS:

1. Unit tests (tests/unit/):

   - test_miner.py: Code extraction, AST parsing, semantic chunking
   - test_raft_generator.py: RAFT data synthesis, quality heuristics
   - test_validator.py: Data validation, quality scoring
   - test_forge.py: PiSSA initialization, QLoRA quantization, training loop
   - test_evaluator.py: Metrics computation, GGUF export
   - test_ollama_manager.py: Ollama interaction, model management

   Each test file should have:
   - Setup fixtures (sample code blocks, datasets)
   - Unit tests for core functions
   - Edge case testing
   - Error condition testing
   - 90%+ code coverage

2. Integration tests (tests/integration/):

   - test_data_pipeline_e2e.py: Extract data → RAFT synthesis → validation
   - test_training_pipeline_e2e.py: Data → Training → Evaluation → Export
   - test_service_e2e.py: API endpoints → job execution → model deployment
   - test_agent_e2e.py: Agent monitoring → training cycle → deployment

3. End-to-end test:

   test_full_system_e2e.py:
   - Take a real small repository (e.g., sample Python project)
   - Run full pipeline: extract → raft → train → eval → export → deploy
   - Query deployed model in Ollama
   - Verify results are reasonable
   - Takes ~30 minutes on Mac M3

4. Performance tests:

   - Time data extraction for 100 files: target < 5 min
   - Time training for 500 examples: target < 30 min (3B) or < 60 min (7B)
   - Memory usage: target < 12GB peak

5. Test fixtures (tests/fixtures/):

   - sample_python_repo/: Small Python project for testing
   - sample_code_blocks.json: Pre-extracted code blocks
   - sample_raft_examples.json: Pre-generated RAFT examples
   - sample_trained_model/: Checkpoint from previous training

6. Test runner configuration:

   pytest.ini with:
   - Markers for unit/integration/e2e tests
   - Coverage configuration (target: 85%+)
   - Timeout settings
   - Test data paths

OUTPUT DELIVERABLES:
- Tests: All test files in tests/ directory
- Fixtures: Sample data for reproducible tests
- Coverage: Coverage report (coverage.xml)
- Documentation: How to run tests, what they verify

ACCEPTANCE CRITERIA:
✅ 85%+ code coverage
✅ All unit tests pass
✅ All integration tests pass
✅ E2E test completes successfully
✅ Performance targets met
✅ Tests run in < 5 minutes (unit only) or < 45 minutes (all)
Prompt 7.2: CI/CD Pipeline & Automated Testing
text
Agent Mission: Implement CI/CD Pipeline with Automated Testing

You are a DevOps engineer. Implement GitHub Actions CI/CD pipeline that runs tests 
automatically on each commit.

REQUIREMENTS:

1. GitHub Actions workflow (.github/workflows/tests.yml):

   on: [push, pull_request]
   
   jobs:
     unit_tests:
       runs-on: ubuntu-latest / macos-latest
       steps:
         - Checkout code
         - Set up Python 3.11
         - Install dependencies
         - Run unit tests
         - Upload coverage report
     
     integration_tests:
       runs-on: ubuntu-latest / macos-latest
       steps:
         - Checkout code
         - Set up Python 3.11
         - Install dependencies + Ollama
         - Run integration tests
         - Upload logs

     code_quality:
       runs-on: ubuntu-latest
       steps:
         - Checkout code
         - Run linting (pylint, black, isort)
         - Run type checking (mypy)
         - Check docstrings

2. Linting & formatting:

   pyproject.toml configuration for:
   - black (code formatting)
   - isort (import sorting)
   - pylint (linting)
   - mypy (type checking)

3. Pre-commit hooks (.pre-commit-config.yaml):

   - Auto-format code on commit
   - Run linting
   - Check test coverage

4. Release workflow (.github/workflows/release.yml):

   on: [tag]
   
   - Build package
   - Run full test suite
   - Create GitHub release
   - Publish to PyPI (optional)

OUTPUT DELIVERABLES:
- CI/CD: .github/workflows/tests.yml & release.yml
- Config: pyproject.toml with linting configuration
- Pre-commit: .pre-commit-config.yaml
- Documentation: How to run tests locally, CI/CD explanation

ACCEPTANCE CRITERIA:
✅ CI pipeline runs on each commit
✅ Tests pass on Mac OS (if applicable)
✅ Code quality checks pass
✅ Coverage report generated
PHASE 8: DOCUMENTATION & DEPLOYMENT
Prompt 8.1: Comprehensive Documentation Package
text
Agent Mission: Create Production-Grade Documentation Suite

You are a technical writer. Create comprehensive documentation covering all aspects of 
the system.

REQUIREMENTS:

1. README.md:
   - Project overview
   - Key features & innovations
   - Quick start (5 min setup)
   - Requirements (hardware, software)
   - Architecture diagram
   - Links to detailed docs

2. Architecture.md:
   - System design overview (diagram)
   - Component breakdown (Miner, Forge, Judge, Conductor, Repo Guardian)
   - Data flow (request → response)
   - PiSSA + QLoRA explanation
   - Comparison to alternatives

3. API_Reference.md:
   - All endpoints documented
   - Request/response examples
   - Error codes & meanings
   - OpenAI compatibility note

4. User_Guide.md:
   - Step-by-step: how to fine-tune a model
   - How to deploy to Ollama
   - How to query the model
   - Troubleshooting common issues

5. Developer_Guide.md:
   - How to develop/extend the system
   - Code structure explanation
   - How to add new PEFT methods
   - How to add new evaluation metrics

6. Configuration.md:
   - All config parameters explained
   - How to tune for different hardware (8GB vs 16GB vs 32GB)
   - Performance tuning tips

7. Troubleshooting.md:
   - Common issues & solutions
   - Debugging tips
   - Performance optimization

8. Research_Summary.md:
   - Link to original PDF research
   - Summary of key papers (PiSSA, QLoRA, RAFT, etc.)
   - Benchmarks & results

9. API Documentation (auto-generated):
   - OpenAPI/Swagger spec
   - HTML documentation page

OUTPUT DELIVERABLES:
- Markdown docs in docs/ directory
- API documentation (HTML)
- Diagrams (mermaid, embedded in docs)

ACCEPTANCE CRITERIA:
✅ All features documented
✅ Examples for all major workflows
✅ Troubleshooting section covers common issues
✅ API documentation is complete and accurate
Prompt 8.2: Deployment & Production Checklist
text
Agent Mission: Create Deployment Guide & Production Checklist

You are a deployment engineer. Create comprehensive guides for deploying to production.

REQUIREMENTS:

1. Local Deployment Guide:
   - System requirements (Mac M3/M4, 16GB+ RAM)
   - Installation steps
   - Configuration
   - Starting the service
   - Testing deployment
   - Backup/restore procedures

2. Cloud Deployment Guide (optional):
   - Docker containerization (Dockerfile)
   - Deploy to AWS EC2
   - Deploy to Google Cloud
   - Deploy to Azure
   - Cost optimization tips

3. Production Checklist:

   Pre-deployment:
   ☐ All tests pass
   ☐ Security review completed
   ☐ Performance benchmarks acceptable
   ☐ Documentation reviewed
   ☐ Disaster recovery plan in place
   
   During deployment:
   ☐ Backup existing model
   ☐ Verify Ollama running
   ☐ Test endpoints with curl
   ☐ Monitor logs
   ☐ Monitor resource usage
   
   Post-deployment:
   ☐ Monitor for errors (first 24 hours)
   ☐ Verify query latency < 500ms
   ☐ Verify no OOM errors
   ☐ Document deployment notes
   ☐ Update monitoring dashboard

4. Monitoring & Observability:
   - Logging setup (what to log, where)
   - Metrics to track (latency, error rate, memory)
   - Alerting rules
   - Dashboard examples

OUTPUT DELIVERABLES:
- Deployment guide (Markdown)
- Docker configuration (Dockerfile, docker-compose.yml)
- Production checklist (Markdown)
- Monitoring setup (Prometheus config, alerts, dashboard)

ACCEPTANCE CRITERIA:
✅ Deployment guide is clear and complete
✅ Docker image builds and runs successfully
✅ Checklist is practical and covers all critical items
✅ Monitoring is configured and working
PHASE 9: VALIDATION & ACCEPTANCE
Prompt 9.1: Final System Validation & Acceptance Tests
text
Agent Mission: Execute Final System Validation

You are a QA lead. Execute comprehensive validation of the entire system before 
production release.

REQUIREMENTS:

1. System Validation Checklist:

   Functionality:
   ☐ Data extraction works on multiple languages
   ☐ RAFT synthesis produces valid examples
   ☐ Training completes without errors
   ☐ Evaluation metrics computed correctly
   ☐ GGUF export produces valid models
   ☐ Ollama integration works
   ☐ FastAPI service responds to all endpoints
   ☐ Antigravity agent can be controlled
   
   Performance:
   ☐ Data extraction: < 5 min for 100 files
   ☐ Training: < 30 min (3B) or < 60 min (7B) on Mac M3
   ☐ Inference latency: < 200ms per token
   ☐ Memory: < 12GB peak
   
   Quality:
   ☐ Model accuracy: 90%+ on domain-specific tasks
   ☐ Hallucination rate: < 5%
   ☐ Code compilation rate: > 95%
   
   Testing:
   ☐ Code coverage: 85%+
   ☐ All unit tests pass
   ☐ All integration tests pass
   ☐ E2E test completes successfully
   
   Documentation:
   ☐ All modules documented
   ☐ All APIs documented
   ☐ User guide complete
   ☐ Developer guide complete
   ☐ Troubleshooting guide complete
   
   Security:
   ☐ Input validation on all endpoints
   ☐ Error messages don't leak sensitive info
   ☐ File uploads validated
   ☐ No hardcoded credentials

2. Acceptance Test Scenarios:

   Scenario 1: Simple Project
   - User uploads small Python project
   - System extracts data
   - Trains for 2 epochs
   - Deploys to Ollama
   - Queries work correctly
   
   Scenario 2: Large Project
   - User uploads larger repo (500+ files)
   - System handles efficiently
   - Training completes successfully
   
   Scenario 3: Error Recovery
   - User interrupts training
   - System gracefully stops
   - User can resume or retry
   
   Scenario 4: Concurrent Jobs
   - User submits 2 training jobs
   - System queues both
   - Executes sequentially
   - Both complete successfully

3. Sign-off:

   Create acceptance sign-off document with:
   - Date of validation
   - Test results summary
   - Issues found (if any)
   - Blockers (if any)
   - Recommendation (go/no-go for production)

OUTPUT DELIVERABLES:
- Validation report (Markdown)
- Test results summary
- Sign-off document
- Known limitations document

ACCEPTANCE CRITERIA:
✅ All checklist items: ✅ (or documented as acceptable exception)
✅ All acceptance test scenarios pass
✅ No critical issues
✅ System ready for production deployment
EXECUTION SUMMARY & GUIDELINES
How to Execute These Prompts with Antigravity
Use Opul Model: Specify "use Gemini 3 with extended reasoning" or "use Opul" to trigger your most capable reasoning engine.

Sequential Execution: Execute phases in order (1 → 2 → 3 → ... → 9). Each phase depends on prior phases.

Iterate & Verify: After each phase, Antigravity should generate:

Code files (production-ready)

Tests (comprehensive coverage)

Documentation (clear and complete)

Artifacts (visual outputs for verification)

Research Integration: Each prompt includes RESEARCH CONTEXT citations to the PDF. Antigravity should:

Reference these specific findings

Justify architectural choices using research

Implement exact SOTA techniques (PiSSA, QLoRA, RAFT, etc.)

Error Handling: If Antigravity encounters issues:

Request clarification explicitly

Suggest alternatives backed by research

Document decisions for later review

Code Quality Standards:

All code must be production-ready (not stub)

100% type hints

Comprehensive docstrings

Error handling on all paths

Logging/observability built-in

Success Metrics
After all phases complete, you should have:

✅ Complete Codebase: 5,000+ lines of production Python code
✅ Comprehensive Tests: 85%+ coverage, 1,000+ lines of test code
✅ Full Documentation: Architecture, API, user guide, developer guide
✅ Working System: Can extract → train → evaluate → deploy end-to-end
✅ Automated Workflows: Antigravity agent can orchestrate full cycles autonomously
✅ SOTA Implementation: Uses PiSSA, QLoRA, RAFT, proper RAG + FT architecture

Final Deliverable
A production-grade, SOTA Local LLM Fine-Tuning Service that:

Runs entirely locally on Mac (no cloud required)

Uses cutting-edge techniques (PiSSA, QLoRA, RAFT)

Manages full lifecycle autonomously (via Antigravity agent)

Serves models via Ollama

Provides REST API for integration

Includes comprehensive testing & documentation

Ready for immediate deployment and use

Status: Ready for Antigravity Execution
Timeline: 2-4 weeks (depending on complexity and iteration cycles)
Outcome: Production-Grade SOTA LLM Fine-Tuning System