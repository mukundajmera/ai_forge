// =============================================================================
// API Client - Type-safe HTTP client for AI Forge backend
// =============================================================================

import type {
    TrainingJob,
    Model,
    DataSource,
    TrainingDataset,
    FineTuneConfig,
    FineTuneResponse,
    JobMetrics,
    JobLogs,
    ValidationResults,
    ExportResponse,
    DataSourceConfig,
    ParsedFile,
    ParsingStatus,
    GenerateDatasetRequest,
    DatasetPreview,
    ModelInfo,
    SystemStatus,
    HealthCheck,
} from './types';

import type {
    Mission,
    MissionsResponse,
    Artifact,
} from '@/lib/types/mission.types';

// =============================================================================
// Configuration
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// =============================================================================
// Error Handling
// =============================================================================

export class APIError extends Error {
    constructor(
        public status: number,
        message: string,
        public data?: Record<string, unknown>
    ) {
        super(message);
        this.name = 'APIError';
    }

    get isNetworkError(): boolean {
        return this.status === 0;
    }

    get isNotFound(): boolean {
        return this.status === 404;
    }

    get isServerError(): boolean {
        return this.status >= 500;
    }

    get isValidationError(): boolean {
        return this.status === 422;
    }
}

// =============================================================================
// API Client Class
// =============================================================================


import { mockApi } from '@/lib/api/mock-adapter';

// ... (imports remain)

// =============================================================================
// API Client Class
// =============================================================================

class APIClient {
    private baseURL: string;
    private abortControllers: Map<string, AbortController> = new Map();
    private useMock: boolean = import.meta.env.VITE_USE_MOCK === 'true';

    constructor(baseURL: string) {
        this.baseURL = baseURL;
    }

    // =========================================================================
    // Core Request Method
    // =========================================================================

    private async request<T>(
        endpoint: string,
        options?: RequestInit & { requestId?: string }
    ): Promise<T> {
        const url = `${this.baseURL}${endpoint}`;
        const { requestId, ...fetchOptions } = options || {};

        // Setup abort controller for cancellable requests
        let abortController: AbortController | undefined;
        if (requestId) {
            // Cancel previous request with same ID
            this.abortControllers.get(requestId)?.abort();
            abortController = new AbortController();
            this.abortControllers.set(requestId, abortController);
        }

        // Force Mock API if enabled (bypassing network completely during emergency)
        if (this.useMock) {
            return this.handleMockFallback<T>(endpoint, options);
        }

        try {
            // Attempt real network request first
            const response = await fetch(url, {
                ...fetchOptions,
                signal: abortController?.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...fetchOptions?.headers,
                },
            });

            if (!response.ok) {
                const error = await response.json().catch(() => null);
                throw new APIError(
                    response.status,
                    error?.detail || error?.message || response.statusText,
                    error
                );
            }

            // Handle empty responses
            const contentType = response.headers.get('content-type');
            if (contentType?.includes('application/json')) {
                return response.json();
            }
            return {} as T;
        } catch (error) {
            // Network error (fetch failed) or explicit mock usage
            // Fallback to Mock API for Critical Paths
            if (this.useMock || (error instanceof TypeError && error.message === 'Failed to fetch')) {
                console.warn(`[API] Network request failed for ${endpoint}. Falling back to Mock API.`);
                return this.handleMockFallback<T>(endpoint, options);
            }

            if (error instanceof APIError) throw error;
            if (error instanceof DOMException && error.name === 'AbortError') {
                throw new APIError(0, 'Request cancelled');
            }
            throw new APIError(0, 'Network error. Check your connection.');
        } finally {
            if (requestId) {
                this.abortControllers.delete(requestId);
            }
        }
    }

    // =========================================================================
    // Generic HTTP Methods (used by hooks)
    // =========================================================================

    public async get<T>(url: string, config?: RequestInit): Promise<T> {
        return this.request<T>(url, { ...config, method: 'GET' });
    }

    public async post<T>(url: string, data?: any, config?: RequestInit): Promise<T> {
        return this.request<T>(url, {
            ...config,
            method: 'POST',
            body: data ? JSON.stringify(data) : undefined,
        });
    }

    public async put<T>(url: string, data?: any, config?: RequestInit): Promise<T> {
        return this.request<T>(url, {
            ...config,
            method: 'PUT',
            body: data ? JSON.stringify(data) : undefined,
        });
    }

    public async delete<T>(url: string, config?: RequestInit): Promise<T> {
        return this.request<T>(url, { ...config, method: 'DELETE' });
    }

    // =========================================================================
    // Mock Fallback Handler
    // =========================================================================

    private async handleMockFallback<T>(endpoint: string, options?: RequestInit): Promise<T> {
        const method = options?.method || 'GET';

        // Jobs
        if (endpoint === '/jobs' && method === 'GET') return mockApi.getJobs() as Promise<T>;
        if (endpoint.match(/^\/jobs\/[^/]+$/) && method === 'GET') {
            const id = endpoint.split('/')[2];
            return mockApi.getJob(id) as Promise<T>;
        }
        if (endpoint === '/v1/fine-tune' && method === 'POST') {
            // Extract config from FormData
            const body = options?.body as FormData;
            const config = JSON.parse(body.get('config') as string);
            return mockApi.startFineTune(config) as Promise<T>;
        }
        if (endpoint.match(/\/metrics$/)) return mockApi.getJobMetrics('id') as Promise<T>;
        if (endpoint.match(/\/logs$/)) return mockApi.getJobLogs('id') as Promise<T>;

        // Models
        if (endpoint === '/models' && method === 'GET') return mockApi.getModels() as Promise<T>;
        if (endpoint === '/models/active') return mockApi.getActiveModel() as Promise<T>;
        if (endpoint.match(/\/activate$/)) { // models/:id/activate
            const id = endpoint.split('/')[2];
            return mockApi.activateModel(id) as Promise<T>;
        }
        if (endpoint.match(/^\/models\/[^/]+$/) && method === 'DELETE') {
            const id = endpoint.split('/')[2];
            await mockApi.deleteModel(id);
            return {} as T;
        }

        // Datasets & Sources
        if (endpoint === '/api/datasets' && method === 'GET') return mockApi.getDatasets() as Promise<T>;
        if (endpoint === '/api/data-sources' && method === 'GET') return mockApi.getDataSources() as Promise<T>;
        if (endpoint === '/api/data-sources/upload' && method === 'POST') {
            // Simulate upload
            const body = options?.body as FormData;
            const files = Array.from(body.getAll('files')) as File[];
            const config = JSON.parse(body.get('config') as string);
            return mockApi.uploadFiles(files, config) as Promise<T>;
        }
        if (endpoint.match(/\/api\/data-sources\/[^/]+$/) && method === 'DELETE') {
            const id = endpoint.split('/')[3];
            await mockApi.deleteDataSource(id);
            return {} as T;
        }

        // System
        if (endpoint === '/status') return mockApi.getSystemStatus() as Promise<T>;
        if (endpoint === '/health') return mockApi.healthCheck() as Promise<T>;

        // Missions
        // if (endpoint.startsWith('/missions')) return mockApi.getMissions() as Promise<T>;

        throw new APIError(404, `Mock endpoint not found: ${method} ${endpoint}`);
    }

    // =========================================================================
    // Jobs API
    // =========================================================================

    async getJobs(): Promise<TrainingJob[]> {
        return this.request<TrainingJob[]>('/v1/fine-tune');
    }

    async getJob(id: string): Promise<TrainingJob> {
        return this.request<TrainingJob>(`/v1/fine-tune/${id}`);
    }

    async getJobMetrics(id: string): Promise<JobMetrics> {
        return this.request<JobMetrics>(`/jobs/${id}/metrics`, {
            requestId: `job-metrics-${id}`,
        });
    }

    async getJobLogs(id: string): Promise<JobLogs> {
        return this.request<JobLogs>(`/jobs/${id}/logs`, {
            requestId: `job-logs-${id}`,
        });
    }

    async startFineTune(config: FineTuneConfig): Promise<FineTuneResponse> {
        const formData = new FormData();
        formData.append('config', JSON.stringify(config));

        return this.request<FineTuneResponse>('/v1/fine-tune', {
            method: 'POST',
            body: formData,
        });
    }

    async cancelJob(id: string): Promise<void> {
        return this.request(`/v1/fine-tune/${id}`, { method: 'DELETE' });
    }

    async getValidation(jobId: string): Promise<ValidationResults> {
        return this.request<ValidationResults>(`/jobs/${jobId}/validation`);
    }

    async exportModel(jobId: string): Promise<ExportResponse> {
        return this.request<ExportResponse>(`/jobs/${jobId}/export`, {
            method: 'POST',
        });
    }

    // =========================================================================
    // Models API
    // =========================================================================

    async getModels(): Promise<Model[]> {
        const response = await this.request<{ data: ModelInfo[] }>('/v1/models');
        // Map backend ModelInfo to frontend Model interface
        return response.data.map(m => ({
            id: m.id,
            name: m.id,
            status: 'ready',
            isActive: false,
            size: 0,
            format: 'gguf',
            quantization: 'q4_k_m',
            metrics: {
                codeBleu: 0,
                humanEval: 0,
                perplexity: 0,
                avgLatency: 0
            },
            baseModel: 'unknown',
            version: '1.0.0',
            trainingJobId: 'unknown',
            createdAt: new Date(m.created * 1000).toISOString()
        }));
    }

    async getActiveModel(): Promise<Model | null> {
        try {
            return await this.request<Model>('/models/active');
        } catch (error) {
            if (error instanceof APIError && error.isNotFound) {
                return null;
            }
            throw error;
        }
    }

    async deployModel(id: string): Promise<{ status: string; ollamaName: string }> {
        return this.request(`/models/${id}/deploy`, { method: 'POST' });
    }

    async activateModel(id: string): Promise<{ status: string }> {
        return this.request(`/models/${id}/activate`, { method: 'POST' });
    }

    async rollbackModel(id: string): Promise<{ status: string }> {
        return this.request(`/models/${id}/rollback`, { method: 'POST' });
    }

    async deleteModel(id: string): Promise<void> {
        return this.request(`/models/${id}`, { method: 'DELETE' });
    }

    // =========================================================================
    // Data Sources API
    // =========================================================================

    async getDataSources(): Promise<DataSource[]> {
        return this.request<DataSource[]>('/api/data-sources');
    }

    async getDataSourceFiles(id: string): Promise<ParsedFile[]> {
        return this.request<ParsedFile[]>(`/api/data-sources/${id}/files`);
    }


    async uploadFiles(
        files: File[],
        config: DataSourceConfig,
        _onProgress?: (progress: number) => void
    ): Promise<DataSource> {
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));
        formData.append('config', JSON.stringify(config));

        // Note: Progress not supported in basic fetch wrapper, use axios or xhr if robust progress needed
        // For now, we mock it or trust the helper
        // Reverting to fetch for consistency with fallback wrapper
        return this.request<DataSource>('/api/data-sources/upload', {
            method: 'POST',
            body: formData
        });
    }

    async syncDataSource(id: string): Promise<{ jobId: string }> {
        return this.request(`/api/data-sources/${id}/sync`, { method: 'POST' });
    }

    async deleteDataSource(id: string): Promise<void> {
        return this.request(`/api/data-sources/${id}`, { method: 'DELETE' });
    }

    async getParsingStatus(jobId: string): Promise<ParsingStatus> {
        return this.request<ParsingStatus>(`/api/parsing/${jobId}`);
    }

    // =========================================================================
    // Datasets API
    // =========================================================================

    async getDatasets(): Promise<TrainingDataset[]> {
        return this.request<TrainingDataset[]>('/api/datasets');
    }

    async getDataset(id: string): Promise<TrainingDataset> {
        return this.request<TrainingDataset>(`/api/datasets/${id}`);
    }

    async getDatasetPreview(id: string): Promise<DatasetPreview> {
        return this.request<DatasetPreview>(`/api/datasets/${id}/preview`);
    }

    async generateDataset(config: GenerateDatasetRequest): Promise<{ datasetId: string; jobId: string }> {
        return this.request('/api/datasets/generate', {
            method: 'POST',
            body: JSON.stringify(config),
        });
    }

    downloadDataset(id: string): void {
        window.open(`${this.baseURL}/api/datasets/${id}/download`, '_blank');
    }

    // =========================================================================
    // System API
    // =========================================================================

    async getSystemStatus(): Promise<SystemStatus> {
        return this.request<SystemStatus>('/status');
    }

    async healthCheck(): Promise<HealthCheck> {
        return this.request<HealthCheck>('/health');
    }

    // =========================================================================
    // Missions API (Antigravity/Repo Guardian)
    // =========================================================================

    async getMissions(filter?: { status?: string; type?: string }): Promise<MissionsResponse> {
        const params = new URLSearchParams();
        if (filter?.status) params.append('status', filter.status);
        if (filter?.type) params.append('type', filter.type);
        const query = params.toString();
        return this.request<MissionsResponse>(`/missions${query ? `?${query}` : ''}`);
    }

    async getMission(id: string): Promise<Mission> {
        return this.request<Mission>(`/missions/${id}`);
    }

    async approveMission(id: string, comment?: string): Promise<{ status: string }> {
        return this.request(`/missions/${id}/approve`, {
            method: 'POST',
            body: JSON.stringify({ comment }),
        });
    }

    async rejectMission(id: string, reason: string): Promise<{ status: string }> {
        return this.request(`/missions/${id}/reject`, {
            method: 'POST',
            body: JSON.stringify({ reason }),
        });
    }

    async getArtifact(id: string): Promise<Artifact> {
        return this.request<Artifact>(`/artifacts/${id}`);
    }

    async getArtifactContent(id: string): Promise<unknown> {
        return this.request(`/artifacts/${id}/content`);
    }
}

// =============================================================================
// Singleton Export
// =============================================================================

export const apiClient = new APIClient(API_BASE_URL);
export const api = apiClient; // Alias for hooks compatibility
export { API_BASE_URL };

