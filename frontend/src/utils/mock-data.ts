// =============================================================================
// AI Forge Mock Data
// Realistic mock data for Phase 3 UI skeleton
// =============================================================================

import type {
    TrainingJob,
    Model,
    DataSource,
    TrainingDataset,
    Mission,
} from '@/types';

// =============================================================================
// Training Jobs
// =============================================================================

export const mockJobs: TrainingJob[] = [
    {
        id: 'job-1',
        projectName: 'myproject',
        status: 'running',
        baseModel: 'Llama-3.2-3B',
        datasetId: 'ds-1',
        epochs: 3,
        learningRate: 0.0002,
        rank: 64,
        batchSize: 2,
        method: 'pissa',
        progress: 67,
        currentEpoch: 2,
        currentStep: 120,
        totalSteps: 450,
        loss: 1.23,
        accuracy: 0.78,
        createdAt: '2026-01-17T10:00:00Z',
        startedAt: '2026-01-17T10:05:00Z',
    },
    {
        id: 'job-2',
        projectName: 'myproject',
        status: 'completed',
        baseModel: 'Llama-3.2-3B',
        datasetId: 'ds-1',
        epochs: 3,
        learningRate: 0.0002,
        rank: 64,
        batchSize: 2,
        method: 'pissa',
        progress: 100,
        currentEpoch: 3,
        currentStep: 450,
        totalSteps: 450,
        loss: 0.89,
        accuracy: 0.92,
        perplexity: 12.4,
        createdAt: '2026-01-16T14:00:00Z',
        startedAt: '2026-01-16T14:05:00Z',
        completedAt: '2026-01-16T16:30:00Z',
        duration: 8700,
    },
    {
        id: 'job-3',
        projectName: 'another-project',
        status: 'failed',
        baseModel: 'Llama-3.2-7B',
        datasetId: 'ds-2',
        epochs: 5,
        learningRate: 0.0001,
        rank: 128,
        batchSize: 4,
        method: 'qlora',
        progress: 7,
        currentEpoch: 1,
        currentStep: 42,
        totalSteps: 600,
        loss: 3.45,
        createdAt: '2026-01-15T09:00:00Z',
        startedAt: '2026-01-15T09:10:00Z',
        completedAt: '2026-01-15T09:35:00Z',
        duration: 1500,
        error: {
            message: 'CUDA out of memory',
            type: 'OOM',
            suggestions: [
                'Reduce batch size (current: 4, try: 2)',
                'Reduce rank (current: 128, try: 64)',
                'Use smaller model (current: 7B, try: 3B)',
            ],
        },
    },
    {
        id: 'job-4',
        projectName: 'api-helper',
        status: 'queued',
        baseModel: 'Llama-3.2-3B',
        datasetId: 'ds-3',
        epochs: 3,
        learningRate: 0.0002,
        rank: 32,
        batchSize: 2,
        method: 'lora',
        progress: 0,
        currentEpoch: 0,
        currentStep: 0,
        totalSteps: 300,
        loss: 0,
        createdAt: '2026-01-17T11:00:00Z',
    },
];

// =============================================================================
// Models
// =============================================================================

export const mockModels: Model[] = [
    {
        id: 'model-1',
        name: 'myproject',
        version: 'v2',
        status: 'active',
        baseModel: 'Llama-3.2-3B',
        trainingJobId: 'job-2',
        size: 2147483648,
        format: 'gguf',
        quantization: 'q4_k_m',
        metrics: {
            codeBleu: 0.84,
            humanEval: 0.72,
            perplexity: 12.4,
            avgLatency: 85,
        },
        createdAt: '2026-01-16T16:30:00Z',
        deployedAt: '2026-01-16T16:45:00Z',
        ollamaName: 'myproject:v2',
        isActive: true,
    },
    {
        id: 'model-2',
        name: 'myproject',
        version: 'v1',
        status: 'deprecated',
        baseModel: 'Llama-3.2-3B',
        trainingJobId: 'job-0',
        size: 2000000000,
        format: 'gguf',
        quantization: 'q4_k_m',
        metrics: {
            codeBleu: 0.78,
            humanEval: 0.65,
            perplexity: 15.2,
            avgLatency: 90,
        },
        createdAt: '2026-01-10T10:00:00Z',
        deployedAt: '2026-01-10T10:15:00Z',
        ollamaName: 'myproject:v1',
        isActive: false,
    },
    {
        id: 'model-3',
        name: 'api-helper',
        version: 'v1',
        status: 'candidate',
        baseModel: 'Llama-3.2-3B',
        trainingJobId: 'job-5',
        size: 2100000000,
        format: 'gguf',
        quantization: 'q5_k_m',
        metrics: {
            codeBleu: 0.81,
            humanEval: 0.68,
            perplexity: 13.8,
            avgLatency: 78,
        },
        createdAt: '2026-01-14T10:00:00Z',
        isActive: false,
    },
];

// =============================================================================
// Data Sources
// =============================================================================

export const mockDataSources: DataSource[] = [
    {
        id: 'ds-1',
        name: 'myproject',
        type: 'git',
        url: 'https://github.com/user/myproject',
        branch: 'main',
        status: 'ready',
        fileCount: 42,
        totalSize: 5242880,
        lastSynced: '2026-01-17T08:00:00Z',
        config: {
            includePatterns: ['src/**/*.py', 'docs/**/*.md'],
            excludePatterns: ['**/test/**', '**/__pycache__/**'],
            fileTypes: ['.py', '.md'],
        },
    },
    {
        id: 'ds-2',
        name: 'documentation',
        type: 'upload',
        status: 'ready',
        fileCount: 8,
        totalSize: 1048576,
        lastSynced: '2026-01-16T12:00:00Z',
        config: {
            includePatterns: [],
            excludePatterns: [],
            fileTypes: ['.md', '.pdf'],
        },
    },
    {
        id: 'ds-3',
        name: 'api-specs',
        type: 'local',
        path: '/Users/dev/projects/api-specs',
        status: 'syncing',
        fileCount: 15,
        totalSize: 2097152,
        lastSynced: '2026-01-17T09:30:00Z',
        config: {
            includePatterns: ['**/*.yaml', '**/*.json'],
            excludePatterns: [],
            fileTypes: ['.yaml', '.json'],
        },
    },
];

// =============================================================================
// Training Datasets
// =============================================================================

export const mockDatasets: TrainingDataset[] = [
    {
        id: 'dataset-1',
        name: 'myproject-dataset-v1',
        sourceIds: ['ds-1'],
        exampleCount: 512,
        createdAt: '2026-01-15T10:00:00Z',
        updatedAt: '2026-01-15T10:30:00Z',
        status: 'ready',
        qualityMetrics: {
            avgScore: 0.82,
            diversity: 0.78,
            raftDistractorQuality: 0.88,
        },
        format: 'alpaca',
        filePath: '/datasets/myproject-dataset-v1.json',
    },
    {
        id: 'dataset-2',
        name: 'docs-dataset-v1',
        sourceIds: ['ds-2'],
        exampleCount: 128,
        createdAt: '2026-01-16T14:00:00Z',
        updatedAt: '2026-01-16T14:20:00Z',
        status: 'ready',
        qualityMetrics: {
            avgScore: 0.75,
            diversity: 0.65,
            raftDistractorQuality: 0.72,
        },
        format: 'sharegpt',
        filePath: '/datasets/docs-dataset-v1.json',
    },
];

// =============================================================================
// Missions (Antigravity Integration)
// =============================================================================

export const mockMissions: Mission[] = [
    {
        id: 'mission-1',
        title: 'Recommend retrain for myproject',
        description: '15 new commits detected with significant API changes. Model performance may degrade.',
        status: 'pending_approval',
        type: 'retrain_suggestion',
        confidence: 0.89,
        createdAt: '2026-01-17T09:00:00Z',
        relatedJobIds: [],
        relatedModelIds: ['model-1'],
        recommendedAction: {
            type: 'start_training',
            params: {
                datasetId: 'dataset-1',
                epochs: 3,
                baseModel: 'Llama-3.2-3B',
            },
        },
        artifacts: [
            {
                id: 'artifact-1',
                missionId: 'mission-1',
                type: 'report',
                title: 'Code Change Analysis',
                createdAt: '2026-01-17T09:00:00Z',
            },
        ],
    },
    {
        id: 'mission-2',
        title: 'Deploy myproject:v2 to production',
        description: 'Model myproject:v2 passed all validation checks. CodeBLEU: 0.84, HumanEval: 72%.',
        status: 'completed',
        type: 'deployment_approval',
        confidence: 0.95,
        createdAt: '2026-01-16T16:30:00Z',
        completedAt: '2026-01-16T16:45:00Z',
        relatedJobIds: ['job-2'],
        relatedModelIds: ['model-1'],
        artifacts: [
            {
                id: 'artifact-2',
                missionId: 'mission-2',
                type: 'chart',
                title: 'Validation Results',
                createdAt: '2026-01-16T16:30:00Z',
            },
            {
                id: 'artifact-3',
                missionId: 'mission-2',
                type: 'dashboard',
                title: 'Deployment Dashboard',
                createdAt: '2026-01-16T16:35:00Z',
            },
        ],
    },
    {
        id: 'mission-3',
        title: 'Quality degradation alert',
        description: 'Model myproject:v1 showing 12% increase in perplexity over last 7 days.',
        status: 'active',
        type: 'quality_alert',
        confidence: 0.76,
        createdAt: '2026-01-17T08:00:00Z',
        relatedJobIds: [],
        relatedModelIds: ['model-2'],
        artifacts: [
            {
                id: 'artifact-4',
                missionId: 'mission-3',
                type: 'chart',
                title: 'Performance Trend',
                createdAt: '2026-01-17T08:00:00Z',
            },
        ],
    },
];

// =============================================================================
// System Health Mock Data
// =============================================================================

export interface SystemHealth {
    gpu: {
        name: string;
        memoryUsed: number;
        memoryTotal: number;
        utilization: number;
        temperature: number;
    };
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
        modelsLoaded: string[];
    };
}

export const mockSystemHealth: SystemHealth = {
    gpu: {
        name: 'Apple M3 Max',
        memoryUsed: 24 * 1024 * 1024 * 1024,
        memoryTotal: 48 * 1024 * 1024 * 1024,
        utilization: 65,
        temperature: 72,
    },
    cpu: {
        utilization: 45,
        cores: 14,
    },
    memory: {
        used: 32 * 1024 * 1024 * 1024,
        total: 64 * 1024 * 1024 * 1024,
    },
    ollama: {
        status: 'running',
        modelsLoaded: ['myproject:v2'],
    },
};

// =============================================================================
// Validation Results Mock Data
// =============================================================================

export interface ValidationResult {
    id: string;
    modelId: string;
    modelName: string;
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

export const mockValidationResults: ValidationResult[] = [
    {
        id: 'val-1',
        modelId: 'model-1',
        modelName: 'myproject:v2',
        runAt: '2026-01-17T10:00:00Z',
        metrics: {
            codebleu: 0.84,
            humaneval: 0.72,
            perplexity: 12.4,
            latency: 85,
        },
        passed: true,
    },
    {
        id: 'val-2',
        modelId: 'model-2',
        modelName: 'myproject:v1',
        runAt: '2026-01-16T10:00:00Z',
        metrics: {
            codebleu: 0.78,
            humaneval: 0.65,
            perplexity: 15.2,
            latency: 90,
        },
        passed: true,
        notes: 'Baseline comparison',
    },
    {
        id: 'val-3',
        modelId: 'model-3',
        modelName: 'api-helper:v1',
        runAt: '2026-01-15T10:00:00Z',
        metrics: {
            codebleu: 0.81,
            humaneval: 0.68,
            perplexity: 13.8,
            latency: 78,
        },
        passed: true,
    },
];
