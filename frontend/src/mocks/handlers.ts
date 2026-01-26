// =============================================================================
// MSW Handlers - Mock API endpoints for testing
// =============================================================================

import { http, HttpResponse, delay } from 'msw';

const API_BASE = 'http://localhost:8000';

// =============================================================================
// Mock Data
// =============================================================================

export const mockJobs = [
    {
        id: 'job-1',
        projectName: 'my-project',
        status: 'running',
        baseModel: 'Llama-3.2-3B',
        method: 'pissa',
        progress: 78,
        metrics: {
            loss: 1.234,
            currentEpoch: 2,
            totalEpochs: 3,
            currentStep: 450,
            totalSteps: 1000,
        },
        config: {
            epochs: 3,
            learningRate: 0.0001,
            rank: 64,
            batchSize: 4,
        },
        startedAt: '2024-01-15T10:00:00Z',
        datasetName: 'test-dataset',
    },
    {
        id: 'job-2',
        projectName: 'api-helper',
        status: 'completed',
        baseModel: 'Qwen2.5-Coder-7B',
        method: 'pissa',
        progress: 100,
        metrics: {
            loss: 0.892,
            currentEpoch: 3,
            totalEpochs: 3,
            currentStep: 1000,
            totalSteps: 1000,
        },
        config: {
            epochs: 3,
            learningRate: 0.0001,
            rank: 128,
            batchSize: 2,
        },
        startedAt: '2024-01-14T08:00:00Z',
        completedAt: '2024-01-14T11:30:00Z',
        datasetName: 'production-data',
    },
];

export const mockModels = [
    {
        id: 'model-1',
        name: 'myproject-v1',
        baseModel: 'Llama-3.2-3B',
        status: 'active',
        createdAt: '2024-01-14T12:00:00Z',
        metrics: {
            codebleu: 0.78,
            humaneval: 0.65,
            perplexity: 12.3,
            avgLatency: 45,
        },
        ollamaName: 'myproject:latest',
    },
    {
        id: 'model-2',
        name: 'old-project-v1',
        baseModel: 'Qwen2.5-Coder-3B',
        status: 'ready',
        createdAt: '2024-01-10T09:00:00Z',
        metrics: {
            codebleu: 0.72,
            humaneval: 0.58,
            perplexity: 15.8,
            avgLatency: 38,
        },
    },
];

export const mockDatasets = [
    {
        id: 'ds-1',
        name: 'test-dataset',
        exampleCount: 1500,
        format: 'alpaca',
        createdAt: '2024-01-13T14:00:00Z',
        sourceIds: ['src-1'],
    },
];

export const mockMissions = [
    {
        id: 'mission-1',
        title: 'Recommend retrain for myproject',
        description: 'New commits detected. Training on updated data could improve model quality.',
        status: 'pending_approval',
        type: 'retrain',
        priority: 'medium',
        confidence: 85,
        createdAt: '2024-01-15T09:00:00Z',
        reasoning: {
            trigger: 'New commits detected',
            analysis: 'Detected 15 new commits since last training',
            expectedOutcome: '~5% improvement in accuracy',
        },
        recommendedAction: {
            type: 'start_training',
            parameters: { datasetId: 'ds-1', epochs: 2 },
        },
        artifacts: [],
        relatedJobIds: ['job-1'],
        relatedModelIds: ['model-1'],
        relatedDatasetIds: ['ds-1'],
        approval: {},
    },
];

export const mockSystemStatus = {
    healthy: true,
    version: '1.0.0',
    uptime: 86400,
    cpu: { utilization: 45, cores: 8 },
    memory: { total: 16000000000, used: 8000000000 },
    gpu: {
        name: 'Apple M2 Pro',
        memoryUsed: 4000000000,
        memoryTotal: 16000000000,
        utilization: 30,
        temperature: 45,
    },
    ollama: { status: 'running', version: '0.1.0', modelsLoaded: ['myproject:latest'] },
    runningJobs: 1,
};

// =============================================================================
// Handlers
// =============================================================================

export const handlers = [
    // Jobs
    http.get(`${API_BASE}/jobs`, async () => {
        await delay(100);
        return HttpResponse.json(mockJobs);
    }),

    // GET /v1/fine-tune - Returns list of training jobs (mapped to backend format)
    http.get(`${API_BASE}/v1/fine-tune`, async () => {
        await delay(100);
        // Map to backend format that apiClient.getJobs expects
        return HttpResponse.json(mockJobs.map(job => ({
            job_id: job.id,
            status: job.status,
            progress: job.progress,
            current_epoch: job.metrics?.currentEpoch,
            current_step: job.metrics?.currentStep,
            config: {
                project_name: job.projectName,
                base_model: job.baseModel,
                dataset_id: job.datasetName,
                epochs: job.config?.epochs,
                learning_rate: job.config?.learningRate,
                rank: job.config?.rank,
                batch_size: job.config?.batchSize,
            },
            started_at: job.startedAt,
            completed_at: job.completedAt,
        })));
    }),

    http.get(`${API_BASE}/jobs/:jobId`, async ({ params }) => {
        await delay(50);
        const job = mockJobs.find(j => j.id === params.jobId);
        if (!job) {
            return HttpResponse.json({ message: 'Not found' }, { status: 404 });
        }
        return HttpResponse.json(job);
    }),

    http.get(`${API_BASE}/jobs/:jobId/metrics`, async () => {
        await delay(50);
        return HttpResponse.json({
            steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            losses: [3.2, 2.8, 2.5, 2.2, 1.9, 1.7, 1.5, 1.3, 1.2, 1.1],
        });
    }),

    http.get(`${API_BASE}/jobs/:jobId/logs`, async () => {
        await delay(50);
        return HttpResponse.json({
            logs: [
                '[INFO] Starting training...',
                '[INFO] Epoch 1/3',
                '[INFO] Step 100/1000, Loss: 2.5',
            ],
            lastUpdated: new Date().toISOString(),
        });
    }),

    http.post(`${API_BASE}/v1/fine-tune`, async () => {
        await delay(200);
        return HttpResponse.json({
            jobId: 'job-new',
            status: 'queued',
            message: 'Training job queued successfully',
        }, { status: 201 });
    }),

    http.delete(`${API_BASE}/jobs/:jobId`, async () => {
        await delay(100);
        return HttpResponse.json({ message: 'Job cancelled' });
    }),

    http.get(`${API_BASE}/jobs/:jobId/validation`, async () => {
        await delay(100);
        return HttpResponse.json({
            jobId: 'job-1',
            modelId: 'model-1',
            runAt: new Date().toISOString(),
            metrics: {
                codebleu: 0.78,
                humaneval: 0.65,
                perplexity: 12.3,
                latency: 45,
            },
            passed: true,
        });
    }),

    http.post(`${API_BASE}/jobs/:jobId/export`, async () => {
        await delay(200);
        return HttpResponse.json({
            exportId: 'export-1',
            status: 'completed',
            outputPath: '/models/myproject.gguf',
            ollamaName: 'myproject:latest',
        });
    }),

    // Models
    http.get(`${API_BASE}/models`, async () => {
        await delay(100);
        return HttpResponse.json(mockModels);
    }),

    http.get(`${API_BASE}/models/active`, async () => {
        await delay(50);
        const active = mockModels.find(m => m.status === 'active');
        return HttpResponse.json(active);
    }),

    http.post(`${API_BASE}/models/:modelId/deploy`, async () => {
        await delay(200);
        return HttpResponse.json({ message: 'Deployed successfully' });
    }),

    http.post(`${API_BASE}/models/:modelId/activate`, async () => {
        await delay(100);
        return HttpResponse.json({ message: 'Activated' });
    }),

    http.post(`${API_BASE}/models/:modelId/rollback`, async () => {
        await delay(100);
        return HttpResponse.json({ message: 'Rolled back' });
    }),

    // Datasets
    http.get(`${API_BASE}/datasets`, async () => {
        await delay(100);
        return HttpResponse.json(mockDatasets);
    }),

    http.get(`${API_BASE}/datasets/:datasetId`, async ({ params }) => {
        await delay(50);
        const dataset = mockDatasets.find(d => d.id === params.datasetId);
        if (!dataset) {
            return HttpResponse.json({ message: 'Not found' }, { status: 404 });
        }
        return HttpResponse.json(dataset);
    }),

    http.get(`${API_BASE}/data-sources`, async () => {
        await delay(100);
        return HttpResponse.json([
            { id: 'src-1', name: 'my-repo', type: 'git', status: 'synced', fileCount: 50 },
        ]);
    }),

    // Also handle /api/ prefixed routes
    http.get(`${API_BASE}/api/data-sources`, async () => {
        await delay(100);
        return HttpResponse.json([
            { id: 'src-1', name: 'my-repo', type: 'git', status: 'synced', fileCount: 50 },
        ]);
    }),

    http.get(`${API_BASE}/api/datasets`, async () => {
        await delay(100);
        return HttpResponse.json(mockDatasets);
    }),

    http.get(`${API_BASE}/api/datasets/:datasetId`, async ({ params }) => {
        await delay(50);
        const dataset = mockDatasets.find(d => d.id === params.datasetId);
        if (!dataset) {
            return HttpResponse.json({ message: 'Not found' }, { status: 404 });
        }
        return HttpResponse.json(dataset);
    }),

    http.post(`${API_BASE}/api/datasets/generate`, async () => {
        await delay(200);
        return HttpResponse.json({
            datasetId: 'ds-new',
            status: 'generating',
        }, { status: 201 });
    }),

    http.delete(`${API_BASE}/api/data-sources/:sourceId`, async () => {
        await delay(100);
        return HttpResponse.json({ message: 'Deleted' });
    }),

    http.post(`${API_BASE}/data-sources`, async () => {
        await delay(200);
        return HttpResponse.json({
            sourceId: 'src-new',
            parsingJobId: 'parse-1',
        }, { status: 201 });
    }),

    http.post(`${API_BASE}/datasets/generate`, async () => {
        await delay(200);
        return HttpResponse.json({
            datasetId: 'ds-new',
            status: 'generating',
        }, { status: 201 });
    }),

    // Missions
    http.get(`${API_BASE}/missions`, async ({ request }) => {
        await delay(100);
        const url = new URL(request.url);
        const status = url.searchParams.get('status');

        let filtered = mockMissions;
        if (status) {
            filtered = mockMissions.filter(m => m.status === status);
        }

        return HttpResponse.json({
            missions: filtered,
            total: filtered.length,
            pending: mockMissions.filter(m => m.status === 'pending_approval').length,
        });
    }),

    http.get(`${API_BASE}/missions/:missionId`, async ({ params }) => {
        await delay(50);
        const mission = mockMissions.find(m => m.id === params.missionId);
        if (!mission) {
            return HttpResponse.json({ message: 'Not found' }, { status: 404 });
        }
        return HttpResponse.json(mission);
    }),

    http.post(`${API_BASE}/missions/:missionId/approve`, async () => {
        await delay(100);
        return HttpResponse.json({ message: 'Approved' });
    }),

    http.post(`${API_BASE}/missions/:missionId/reject`, async () => {
        await delay(100);
        return HttpResponse.json({ message: 'Rejected' });
    }),

    // System
    http.get(`${API_BASE}/status`, async () => {
        await delay(50);
        return HttpResponse.json(mockSystemStatus);
    }),

    http.get(`${API_BASE}/health`, async () => {
        return HttpResponse.json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            checks: { database: true, ollama: true, gpu: true },
        });
    }),
];
