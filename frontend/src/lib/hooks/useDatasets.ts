// =============================================================================
// Datasets Hooks - React Query hooks for data source and dataset operations
// =============================================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, APIError } from '@/lib/api';
import type { DataSourceConfig, GenerateDatasetRequest } from '@/lib/api';
import { toast } from '@/components/ui/Toast';

// =============================================================================
// Query Keys
// =============================================================================

export const dataSourceKeys = {
    all: ['dataSources'] as const,
    lists: () => [...dataSourceKeys.all, 'list'] as const,
    list: () => dataSourceKeys.lists(),
    detail: (id: string) => [...dataSourceKeys.all, 'detail', id] as const,
    files: (id: string) => [...dataSourceKeys.detail(id), 'files'] as const,
};

export const datasetKeys = {
    all: ['datasets'] as const,
    lists: () => [...datasetKeys.all, 'list'] as const,
    list: () => datasetKeys.lists(),
    detail: (id: string) => [...datasetKeys.all, 'detail', id] as const,
    preview: (id: string) => [...datasetKeys.detail(id), 'preview'] as const,
};

export const parsingKeys = {
    status: (jobId: string) => ['parsing', jobId] as const,
};

// =============================================================================
// Data Source Queries
// =============================================================================

/**
 * Fetch all data sources
 */
export function useDataSources() {
    return useQuery({
        queryKey: dataSourceKeys.list(),
        queryFn: () => apiClient.getDataSources(),
        staleTime: 5000,
    });
}

/**
 * Fetch files for a data source
 */
export function useDataSourceFiles(sourceId: string | undefined) {
    return useQuery({
        queryKey: dataSourceKeys.files(sourceId!),
        queryFn: () => apiClient.getDataSourceFiles(sourceId!),
        enabled: !!sourceId,
        staleTime: 5000,
    });
}

/**
 * Poll parsing job status
 */
export function useParsingStatus(jobId: string | undefined) {
    return useQuery({
        queryKey: parsingKeys.status(jobId!),
        queryFn: () => apiClient.getParsingStatus(jobId!),
        enabled: !!jobId,
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === 'pending' || status === 'processing' ? 2000 : false;
        },
    });
}

// =============================================================================
// Data Source Mutations
// =============================================================================

/**
 * Upload files to create a new data source
 */
export function useUploadFiles() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({
            files,
            config,
            onProgress,
        }: {
            files: File[];
            config: DataSourceConfig;
            onProgress?: (progress: number) => void;
        }) => apiClient.uploadFiles(files, config, onProgress),
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: dataSourceKeys.all });
            toast.success('Files uploaded', `${data.fileCount} files ready for parsing`);
        },
        onError: (error: APIError) => {
            toast.error('Upload failed', error.message);
        },
    });
}

/**
 * Sync/refresh a data source
 */
export function useSyncDataSource() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (sourceId: string) => apiClient.syncDataSource(sourceId),
        onSuccess: (data, sourceId) => {
            queryClient.invalidateQueries({ queryKey: dataSourceKeys.detail(sourceId) });
            toast.success('Sync started', `Parsing job ${data.jobId} queued`);
        },
        onError: (error: APIError) => {
            toast.error('Sync failed', error.message);
        },
    });
}

/**
 * Delete a data source
 */
export function useDeleteDataSource() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (sourceId: string) => apiClient.deleteDataSource(sourceId),
        onMutate: async (sourceId) => {
            // Cancel queries
            await queryClient.cancelQueries({ queryKey: dataSourceKeys.list() });

            // Snapshot
            const previousSources = queryClient.getQueryData(dataSourceKeys.list());

            // Optimistically remove
            queryClient.setQueryData(dataSourceKeys.list(), (old: unknown) => {
                if (!Array.isArray(old)) return old;
                return old.filter((source) => source.id !== sourceId);
            });

            return { previousSources };
        },
        onError: (error: APIError, _, context) => {
            // Rollback
            if (context?.previousSources) {
                queryClient.setQueryData(dataSourceKeys.list(), context.previousSources);
            }
            toast.error('Delete failed', error.message);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: dataSourceKeys.all });
            toast.success('Data source deleted');
        },
    });
}

// =============================================================================
// Dataset Queries
// =============================================================================

/**
 * Fetch all training datasets
 */
export function useDatasets() {
    return useQuery({
        queryKey: datasetKeys.list(),
        queryFn: () => apiClient.getDatasets(),
        staleTime: 5000,
    });
}

/**
 * Fetch a single dataset
 */
export function useDataset(datasetId: string | undefined) {
    return useQuery({
        queryKey: datasetKeys.detail(datasetId!),
        queryFn: () => apiClient.getDataset(datasetId!),
        enabled: !!datasetId,
    });
}

/**
 * Fetch dataset preview (sample examples)
 */
export function useDatasetPreview(datasetId: string | undefined) {
    return useQuery({
        queryKey: datasetKeys.preview(datasetId!),
        queryFn: () => apiClient.getDatasetPreview(datasetId!),
        enabled: !!datasetId,
        staleTime: 60000, // Previews don't change often
    });
}

// =============================================================================
// Dataset Mutations
// =============================================================================

/**
 * Generate a training dataset from sources using RAFT
 */
export function useGenerateDataset() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (config: GenerateDatasetRequest) => apiClient.generateDataset(config),
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: datasetKeys.all });
            toast.success('Dataset generation started', `Dataset ${data.datasetId} being created`);
        },
        onError: (error: APIError) => {
            toast.error('Generation failed', error.message);
        },
    });
}

/**
 * Download dataset as JSON file
 */
export function useDownloadDataset() {
    return useMutation({
        mutationFn: (datasetId: string) => {
            apiClient.downloadDataset(datasetId);
            return Promise.resolve();
        },
        onSuccess: () => {
            toast.success('Download started');
        },
    });
}
