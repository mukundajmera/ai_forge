// =============================================================================
// Jobs Hooks - React Query hooks for training job operations
// =============================================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, APIError } from '@/lib/api';
import type { FineTuneConfig } from '@/lib/api';
import { toast } from '@/components/ui/Toast';

// =============================================================================
// Query Keys
// =============================================================================

export const jobKeys = {
    all: ['jobs'] as const,
    lists: () => [...jobKeys.all, 'list'] as const,
    list: () => jobKeys.lists(),
    details: () => [...jobKeys.all, 'detail'] as const,
    detail: (id: string) => [...jobKeys.details(), id] as const,
    metrics: (id: string) => [...jobKeys.detail(id), 'metrics'] as const,
    logs: (id: string) => [...jobKeys.detail(id), 'logs'] as const,
    validation: (id: string) => [...jobKeys.detail(id), 'validation'] as const,
};

// =============================================================================
// List Queries
// =============================================================================

/**
 * Fetch all training jobs with 5s polling
 */
export function useJobs() {
    return useQuery({
        queryKey: jobKeys.list(),
        queryFn: () => apiClient.getJobs(),
        refetchInterval: 5000,
        staleTime: 2000,
    });
}

// =============================================================================
// Detail Queries
// =============================================================================

/**
 * Fetch single job details with conditional polling
 * Polls every 3s while job is running, stops when complete/failed
 */
export function useJob(jobId: string | undefined) {
    return useQuery({
        queryKey: jobKeys.detail(jobId!),
        queryFn: () => apiClient.getJob(jobId!),
        enabled: !!jobId,
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === 'running' || status === 'queued' ? 3000 : false;
        },
        staleTime: 1000,
    });
}

/**
 * Fetch job training metrics (loss curve data)
 */
export function useJobMetrics(jobId: string | undefined) {
    return useQuery({
        queryKey: jobKeys.metrics(jobId!),
        queryFn: () => apiClient.getJobMetrics(jobId!),
        enabled: !!jobId,
        refetchInterval: 5000,
        staleTime: 2000,
    });
}

/**
 * Fetch job logs with frequent polling
 */
export function useJobLogs(jobId: string | undefined, enabled = true) {
    return useQuery({
        queryKey: jobKeys.logs(jobId!),
        queryFn: () => apiClient.getJobLogs(jobId!),
        enabled: !!jobId && enabled,
        refetchInterval: 2000,
        staleTime: 500,
    });
}

/**
 * Fetch validation results for a completed job
 */
export function useValidation(jobId: string | undefined) {
    return useQuery({
        queryKey: jobKeys.validation(jobId!),
        queryFn: () => apiClient.getValidation(jobId!),
        enabled: !!jobId,
        staleTime: 60000, // Validation results don't change
    });
}

// =============================================================================
// Mutations
// =============================================================================

/**
 * Start a new fine-tuning job
 */
export function useStartFineTune() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (config: FineTuneConfig) => apiClient.startFineTune(config),
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: jobKeys.all });
            toast.success('Training started', `Job ${data.jobId} queued successfully`);
        },
        onError: (error: APIError) => {
            toast.error('Failed to start training', error.message);
        },
    });
}

/**
 * Cancel a running job
 */
export function useCancelJob() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (jobId: string) => apiClient.cancelJob(jobId),
        onMutate: async (jobId) => {
            // Cancel any outgoing refetches
            await queryClient.cancelQueries({ queryKey: jobKeys.detail(jobId) });

            // Snapshot the previous value
            const previousJob = queryClient.getQueryData(jobKeys.detail(jobId));

            // Optimistically update to cancelled
            queryClient.setQueryData(jobKeys.detail(jobId), (old: unknown) => {
                if (old && typeof old === 'object' && 'status' in old) {
                    return { ...old, status: 'cancelled' };
                }
                return old;
            });

            return { previousJob };
        },
        onError: (error: APIError, jobId, context) => {
            // Rollback on error
            if (context?.previousJob) {
                queryClient.setQueryData(jobKeys.detail(jobId), context.previousJob);
            }
            toast.error('Failed to cancel job', error.message);
        },
        onSuccess: (_) => {
            queryClient.invalidateQueries({ queryKey: jobKeys.all });
            toast.success('Job cancelled');
        },
    });
}

/**
 * Export model to GGUF format
 */
export function useExportModel() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (jobId: string) => apiClient.exportModel(jobId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['models'] });
            toast.success('Model export started', 'You will be notified when complete');
        },
        onError: (error: APIError) => {
            toast.error('Export failed', error.message);
        },
    });
}
