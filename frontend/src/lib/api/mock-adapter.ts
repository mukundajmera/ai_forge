// =============================================================================
// Mock API Adapter
// =============================================================================
// This is a placeholder for mocking API responses during development

// Mock methods matching the API client interface
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const mockApi: Record<string, any> = {
    enabled: false,

    // Mock handlers can be added here for testing
    handlers: new Map<string, (params?: Record<string, unknown>) => Promise<unknown>>(),

    register(endpoint: string, handler: (params?: Record<string, unknown>) => Promise<unknown>): void {
        this.handlers.set(endpoint, handler);
    },

    async call(endpoint: string, params?: Record<string, unknown>): Promise<unknown> {
        const handler = this.handlers.get(endpoint);
        if (handler) {
            return handler(params);
        }
        throw new Error(`No mock handler for ${endpoint}`);
    },

    // API method stubs - these return mock data
    async getJobs() { return []; },
    async getJob(_id: string) { return null; },
    async startFineTune(_config: unknown) { return { jobId: 'mock-job' }; },
    async getJobMetrics(_id: string) { return { steps: [], losses: [] }; },
    async getJobLogs(_id: string) { return { logs: [], lastUpdated: new Date().toISOString() }; },
    async getModels() { return []; },
    async getActiveModel() { return null; },
    async activateModel(_id: string) { return { success: true }; },
    async deleteModel(_id: string) { return { success: true }; },
    async getDatasets() { return []; },
    async getDataSources() { return []; },
    async uploadFiles(_files: unknown) { return { success: true }; },
    async deleteDataSource(_id: string) { return { success: true }; },
    async getSystemStatus() { return { healthy: true }; },
    async healthCheck() { return { status: 'healthy' }; },
};

export default mockApi;
