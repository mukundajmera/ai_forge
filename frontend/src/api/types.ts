// =============================================================================
// API Types - Request/Response interfaces for backend integration
// =============================================================================

import type {
    TrainingJob,
    Model,
    DataSource,
    TrainingDataset,
    TrainingExample,
} from '@/types';

// =============================================================================
// Fine-Tuning
// =============================================================================

export interface FineTuneConfig {
    project_name: string;
    base_model: string;
    dataset_id: string;
    epochs: number;
    learning_rate: number;
    rank: number;
    batch_size: number;
    use_pissa: boolean;
    // Camel case aliases for frontend convenience
    projectName?: string;
    baseModel?: string;
    datasetId?: string;
    learningRate?: number;
    batchSize?: number;
    usePissa?: boolean;
    method?: 'pissa' | 'lora' | 'qlora';
}

export interface FineTuneResponse {
    jobId: string;
    status: 'queued';
    message: string;
}

export interface JobMetrics {
    steps: number[];
    losses: number[];
    valLosses?: number[];
    learningRates?: number[];
}

export interface JobLogs {
    logs: string[];
    lastUpdated: string;
}

// =============================================================================
// Validation & Export
// =============================================================================

export interface ValidationResults {
    jobId: string;
    modelId: string;
    runAt: string;
    metrics: {
        codebleu: number;
        humaneval: number;
        perplexity: number;
        latency: number;
    };
    passed: boolean;
    notes?: string;
}

export interface ExportResponse {
    exportId: string;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    outputPath?: string;
    ollamaName?: string;
}

// =============================================================================
// Data Sources & Datasets
// =============================================================================

export interface DataSourceConfig {
    name: string;
    type: 'git' | 'upload' | 'local';
    url?: string;
    branch?: string;
    path?: string;
    includePatterns?: string[];
    excludePatterns?: string[];
    fileTypes?: string[];
}

export interface ParsedFile {
    id: string;
    sourceId: string;
    path: string;
    language: string;
    size: number;
    chunkCount: number;
    status: 'pending' | 'parsed' | 'failed';
    error?: string;
}

export interface ParsingStatus {
    jobId: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
    totalFiles: number;
    processedFiles: number;
    errors?: string[];
}

export interface GenerateDatasetRequest {
    sourceIds: string[];
    name: string;
    format?: 'alpaca' | 'sharegpt';
    options?: {
        numDistractors?: number;
        temperature?: number;
        maxExamples?: number;
    };
}

export interface DatasetPreview {
    examples: TrainingExample[];
    totalCount: number;
    statistics: {
        avgTokens: number;
        byDifficulty: Record<string, number>;
        byQuestionType: Record<string, number>;
    };
}

// =============================================================================
// System
// =============================================================================

export interface SystemStatus {
    healthy: boolean;
    version: string;
    uptime: number;
    gpu: {
        name: string;
        memoryUsed: number;
        memoryTotal: number;
        utilization: number;
        temperature: number;
    } | null;
    cpu: {
        utilization: number;
        cores: number;
    };
    memory: {
        used: number;
        total: number;
    };
    ollama: {
        status: 'running' | 'stopped' | 'error';
        version?: string;
        modelsLoaded: string[];
    };
    runningJobs: number;
}

export interface HealthCheck {
    status: 'healthy' | 'degraded' | 'unhealthy';
    timestamp: string;
    checks: {
        database: boolean;
        ollama: boolean;
        gpu: boolean;
    };
}

// =============================================================================
// Re-exports for convenience
// =============================================================================

// =============================================================================
// Backend DTOs
// =============================================================================

export interface ModelInfo {
    id: string;
    object: string;
    owned_by: string;
    created: number;
}

export type {
    TrainingJob,
    Model,
    DataSource,
    TrainingDataset,
    TrainingExample,
};
