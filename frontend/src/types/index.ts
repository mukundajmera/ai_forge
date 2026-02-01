// =============================================================================
// Data Source Types
// =============================================================================

export type DataSourceType = 'git' | 'upload' | 'local';
export type DataSourceStatus = 'syncing' | 'parsing' | 'ready' | 'error';

export interface DataSourceConfig {
    includePatterns: string[];
    excludePatterns: string[];
    fileTypes: string[];
}

export interface DataSource {
    id: string;
    name: string;
    type: DataSourceType;
    path?: string;
    url?: string;
    branch?: string;
    status: DataSourceStatus;
    fileCount: number;
    totalSize: number; // bytes
    lastSynced: string; // ISO timestamp
    config: DataSourceConfig;
    error?: string;
}

// =============================================================================
// Parsed File Types
// =============================================================================

export type FileType = 'code' | 'text' | 'markdown' | 'pdf';
export type ParseStatus = 'pending' | 'parsing' | 'success' | 'failed';

export interface FileMetadata {
    functions?: number;
    classes?: number;
    docstrings?: number;
    pages?: number; // for PDFs
    startLine?: number;
    endLine?: number;
}

export interface ParsedFile {
    id: string;
    sourceId: string;
    filename: string;
    path: string;
    type: FileType;
    language?: string;
    size: number;
    parseStatus: ParseStatus;
    chunksExtracted: number;
    qualityScore: number; // 0.0-1.0
    error?: string;
    metadata?: FileMetadata;
}

export interface CodeChunk {
    id: string;
    fileId: string;
    name: string;
    type: 'function' | 'class' | 'method' | 'module';
    content: string;
    tokens: number;
    qualityScore: number;
    docstring?: string;
    dependencies: string[];
}

// =============================================================================
// Training Example Types
// =============================================================================

export type QuestionType =
    | 'purpose'
    | 'usage'
    | 'edge_cases'
    | 'dependencies'
    | 'design'
    | 'extension'
    | 'debugging'
    | 'comparison';

export type Difficulty = 'easy' | 'medium' | 'hard';

export interface TrainingExample {
    id: string;
    question: string;
    questionType: QuestionType;
    context: string;
    answer: string;
    reasoning?: string;
    distractors?: string[];
    qualityScore: number;
    difficulty: Difficulty;
}

// =============================================================================
// Dataset Types
// =============================================================================

export type DatasetStatus = 'generating' | 'ready' | 'error';
export type DatasetFormat = 'alpaca' | 'sharegpt';

export interface QualityMetrics {
    avgScore: number;
    diversity: number;
    raftDistractorQuality: number;
}

export interface TrainingDataset {
    id: string;
    name: string;
    sourceIds: string[];
    exampleCount: number;
    createdAt: string;
    updatedAt: string;
    status: DatasetStatus;
    qualityMetrics: QualityMetrics;
    format: DatasetFormat;
    filePath?: string;
    error?: string;
    version?: number;
}

// =============================================================================
// Upload & Job Types
// =============================================================================

export type UploadStatus = 'pending' | 'uploading' | 'complete' | 'error';

export interface UploadProgress {
    fileId: string;
    filename: string;
    size: number;
    progress: number; // 0-100
    status: UploadStatus;
    error?: string;
}

export type JobStatus = 'pending' | 'running' | 'complete' | 'failed';

export interface ParsingJob {
    jobId: string;
    status: JobStatus;
    files: ParsedFile[];
    progress: number; // 0-100
    startedAt: string;
    completedAt?: string;
    error?: string;
}

export interface GenerationJob {
    jobId: string;
    status: JobStatus;
    datasetId?: string;
    progress: number;
    currentStep: string;
    examplesGenerated: number;
    totalExpected: number;
    startedAt: string;
    completedAt?: string;
    error?: string;
}

// =============================================================================
// API Request/Response Types
// =============================================================================

export interface AddGitSourceRequest {
    type: 'git';
    url: string;
    branch?: string;
    name?: string;
    config?: Partial<DataSourceConfig>;
}

export interface AddLocalSourceRequest {
    type: 'local';
    path: string;
    name?: string;
    config?: Partial<DataSourceConfig>;
}

export type AddDataSourceRequest = AddGitSourceRequest | AddLocalSourceRequest;

export interface GenerateDatasetRequest {
    sourceIds: string[];
    name?: string;
    format?: DatasetFormat;
    questionsPerBlock?: number;
    difficultyDistribution?: Record<Difficulty, number>;
}

export interface FilePreview {
    file: ParsedFile;
    content: string;
    highlightedContent?: string;
    chunks: CodeChunk[];
}

export interface DatasetPreview {
    dataset: TrainingDataset;
    examples: TrainingExample[];
    statistics: {
        byDifficulty: Record<Difficulty, number>;
        byQuestionType: Record<QuestionType, number>;
    };
}

// =============================================================================
// Training Job Types (Enhanced)
// =============================================================================

export type TrainingJobStatus = 'queued' | 'running' | 'training' | 'completed' | 'failed' | 'cancelled';
export type TrainingMethod = 'pissa' | 'lora' | 'qlora';

export interface TrainingConfig {
    epochs: number;
    learningRate: number;
    rank: number;
    batchSize: number;
    method: TrainingMethod;
}

export interface TrainingMetrics {
    currentEpoch: number;
    totalEpochs: number;
    currentStep: number;
    totalSteps: number;
    loss: number;
    accuracy?: number;
    perplexity?: number;
}

export interface TrainingError {
    message: string;
    type: 'OOM' | 'ValidationError' | 'DataError' | 'Unknown';
    suggestions?: string[];
}

export interface TrainingJob {
    id: string;
    projectName: string;
    status: TrainingJobStatus;
    baseModel: string;
    datasetId: string;
    datasetName?: string;
    // Config (flattened for API compatibility)
    epochs: number;
    learningRate: number;
    rank: number;
    batchSize: number;
    method?: TrainingMethod;
    // Progress metrics (flattened)
    progress: number; // 0-100
    currentEpoch?: number;
    currentStep?: number;
    totalSteps?: number;
    loss?: number;
    accuracy?: number;
    perplexity?: number;
    eta?: string;
    // Timestamps
    createdAt: string;
    startedAt?: string;
    completedAt?: string;
    duration?: number; // seconds
    // Output
    outputDir?: string;
    checkpoints?: string[];
    // Error handling
    error?: TrainingError | string;
}

// =============================================================================
// Model Types (Enhanced)
// =============================================================================

export type ModelStatus = 'active' | 'candidate' | 'deprecated' | 'ready';
export type ModelFormat = 'gguf' | 'ggml';
export type QuantizationType = 'q4_k_m' | 'q5_k_m' | 'q8_0' | 'f16';

export interface ModelMetrics {
    codeBleu: number;
    codebleu?: number; // alias for API compatibility
    humanEval: number;
    humaneval?: number; // alias for API compatibility  
    perplexity: number;
    avgLatency: number; // ms
}

export interface Model {
    id: string;
    name: string;
    version: string;
    status: ModelStatus;
    baseModel: string;
    trainingJobId: string;
    size: number; // bytes
    format: ModelFormat;
    quantization: QuantizationType;
    metrics: ModelMetrics;
    createdAt: string;
    deployedAt?: string;
    ollamaName?: string;
    isActive?: boolean;
}

// =============================================================================
// Mission Types (Antigravity Integration)
// =============================================================================

export type MissionStatus = 'active' | 'pending_approval' | 'completed' | 'failed' | 'approved' | 'rejected';
export type MissionType = 'retrain_suggestion' | 'deployment_approval' | 'quality_alert' | 'performance_drift' | 'data_quality_issue';
export type MissionPriority = 'low' | 'medium' | 'high';
export type ArtifactType = 'chart' | 'report' | 'log' | 'dashboard';

export interface Artifact {
    id: string;
    missionId: string;
    type: ArtifactType;
    title: string;
    description?: string;
    url?: string;
    format?: string;
    size?: number;
    payload?: Record<string, unknown>;
    createdAt: string;
}

export interface MissionApproval {
    approvedAt?: string;
    approvedBy?: string;
    rejectedAt?: string;
    rejectedBy?: string;
    reason?: string;
    rejectionReason?: string;
}

export interface MissionReasoning {
    trigger?: string;
    analysis?: string;
    expectedOutcome?: string;
}

export interface RecommendedAction {
    type: 'start_training' | 'deploy_model' | 'rollback_model';
    params?: Record<string, unknown>;
    estimatedDuration?: string;
}

export interface Mission {
    id: string;
    title: string;
    description: string;
    status: MissionStatus;
    type: MissionType;
    priority?: MissionPriority;
    confidence: number; // 0-1
    reasoning?: MissionReasoning | string;
    createdAt: string;
    completedAt?: string;
    relatedJobIds: string[];
    relatedModelIds: string[];
    relatedDatasetIds?: string[];
    recommendedAction?: RecommendedAction;
    artifacts: Artifact[];
    approval?: MissionApproval;
}

// =============================================================================
// UI State Types
// =============================================================================

export interface Toast {
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    title: string;
    message?: string;
    duration?: number;
}

export interface DialogState {
    isOpen: boolean;
    activeTab?: 'git' | 'upload' | 'local';
}

export interface ValidationWarning {
    type: 'small_dataset' | 'low_quality' | 'no_docstrings' | 'single_file_type' | 'large_dataset';
    message: string;
    severity: 'warning' | 'info';
}

// =============================================================================
// Dataset Version Types
// =============================================================================

export type DatasetVersionStatus = 'draft' | 'ready' | 'archived';

export interface DatasetVersion {
    id: string;
    datasetId: string;
    version: number;
    status: DatasetVersionStatus;
    exampleCount: number;
    filtersApplied: Record<string, unknown>;
    qualityStats: Record<string, number>;
    snapshotHash: string;
    createdAt: string;
    parentVersionId?: string;
}

// =============================================================================
// Experiment / Run Types
// =============================================================================

export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type TaskType = 'instruction_tuning' | 'domain_adaptation' | 'code_specialization' | 'qa_finetuning';

export type HardwareProfile = 'low' | 'medium' | 'high';

export interface EvalMetrics {
    loss?: number;
    evalLoss?: number;
    perplexity?: number;
    codebleu?: number;
    humanevalPassAt1?: number;
    rougeL?: number;
    exactMatch?: number;
    custom: Record<string, number>;
}

export interface Experiment {
    id: string;
    name: string;
    description: string;
    status: ExperimentStatus;
    baseModel: string;
    datasetId?: string;
    datasetVersionId?: string;
    recipeId?: string;
    hyperparameters: Record<string, unknown>;
    metrics: EvalMetrics;
    tags: string[];
    artifacts: Record<string, string>;
    jobId?: string;
    createdAt: string;
    startedAt?: string;
    completedAt?: string;
    durationSeconds?: number;
    error?: string;
}

export interface ExperimentComparison {
    experiments: Experiment[];
    metricSummary: Record<string, Record<string, number | null>>;
}

// =============================================================================
// Training Recipe Types
// =============================================================================

export interface RecipeDefaults {
    epochs: number;
    learningRate: number;
    rank: number;
    batchSize: number;
    usePissa: boolean;
    gradientAccumulationSteps: number;
    warmupRatio: number;
    scheduler: string;
    maxSeqLength: number;
}

export interface EvalSuiteConfig {
    metrics: string[];
    passThresholds: Record<string, number>;
}

export interface Recipe {
    id: string;
    name: string;
    description: string;
    taskType: TaskType;
    supportedModels: string[];
    defaults: RecipeDefaults;
    datasetRequirements: Record<string, unknown>;
    evalSuite: EvalSuiteConfig;
    tags: string[];
    isBuiltin: boolean;
}

export interface RecipeWithDefaults {
    recipe: Recipe;
    adjustedDefaults: RecipeDefaults;
}
