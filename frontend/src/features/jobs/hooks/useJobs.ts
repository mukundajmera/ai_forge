// =============================================================================
// Jobs Feature Hooks - React Query hooks for training job operations
// Re-exports from lib/hooks for feature-level access
// =============================================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, APIError } from '@/lib/api'
import type { FineTuneConfig } from '@/lib/api'
import { toast } from '@/components/ui/Toast'

// =============================================================================
// Types
// =============================================================================

export interface TrainingJob {
    id: string
    projectName: string
    baseModel: string
    status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
    progress: number
    currentEpoch?: number
    epochs: number
    currentStep?: number
    totalSteps?: number
    loss?: number
    learningRate: number
    rank: number
    batchSize: number
    datasetId: string
    datasetName?: string
    createdAt: string
    startedAt?: string
    completedAt?: string
    duration?: string | number
    eta?: string
    error?: string | { message: string; type: string }
    outputDir?: string
    checkpoints?: string[]
    method?: 'pissa' | 'lora' | 'qlora'
}

export interface NewJobRequest {
    projectName: string
    datasetId: string
    baseModel: string
    epochs: number
    learningRate: number
    rank: number
    batchSize: number
    method: 'pissa' | 'lora' | 'qlora'
}

export interface JobLog {
    timestamp: string
    level: 'info' | 'warning' | 'error'
    message: string
}

// =============================================================================
// Query Keys
// =============================================================================

export const jobQueryKeys = {
    all: ['jobs'] as const,
    list: (filters?: { status?: string; limit?: number }) =>
        ['jobs', 'list', filters] as const,
    detail: (id: string) => ['jobs', 'detail', id] as const,
    metrics: (id: string) => ['jobs', 'detail', id, 'metrics'] as const,
    logs: (id: string) => ['jobs', 'detail', id, 'logs'] as const,
    validation: (id: string) => ['jobs', 'detail', id, 'validation'] as const,
}

// =============================================================================
// Queries
// =============================================================================

/**
 * Fetch all jobs with optional filters
 */
export function useJobs(filters?: { status?: string; limit?: number }) {
    return useQuery({
        queryKey: jobQueryKeys.list(filters),
        queryFn: async () => {
            const jobs = await apiClient.getJobs()
            // Apply client-side filtering if needed
            let filtered = jobs
            if (filters?.status) {
                filtered = filtered.filter(j => j.status === filters.status)
            }
            if (filters?.limit) {
                filtered = filtered.slice(0, filters.limit)
            }
            return filtered as TrainingJob[]
        },
        refetchInterval: 5000,
        staleTime: 2000,
    })
}

/**
 * Fetch single job with live updates while running
 */
export function useJob(jobId: string | undefined) {
    return useQuery({
        queryKey: jobQueryKeys.detail(jobId!),
        queryFn: () => apiClient.getJob(jobId!) as Promise<TrainingJob>,
        enabled: !!jobId,
        refetchInterval: (query) => {
            const status = query.state.data?.status
            return status === 'running' || status === 'queued' ? 3000 : false
        },
        staleTime: 1000,
    })
}

/**
 * Fetch job training metrics (loss curve data)
 */
export function useJobMetrics(jobId: string | undefined) {
    return useQuery({
        queryKey: jobQueryKeys.metrics(jobId!),
        queryFn: () => apiClient.getJobMetrics(jobId!),
        enabled: !!jobId,
        refetchInterval: 5000,
        staleTime: 2000,
    })
}

/**
 * Fetch job logs with frequent polling
 */
export function useJobLogs(jobId: string | undefined, _tail = 100) {
    return useQuery({
        queryKey: jobQueryKeys.logs(jobId!),
        queryFn: async () => {
            const result = await apiClient.getJobLogs(jobId!)
            // Transform to JobLog format if needed
            return result.logs.map((msg) => ({
                timestamp: result.lastUpdated,
                level: 'info' as const,
                message: msg,
            })) as JobLog[]
        },
        enabled: !!jobId,
        refetchInterval: 3000,
        staleTime: 500,
    })
}

/**
 * Fetch validation results for a completed job
 */
export function useValidation(jobId: string | undefined) {
    return useQuery({
        queryKey: jobQueryKeys.validation(jobId!),
        queryFn: () => apiClient.getValidation(jobId!),
        enabled: !!jobId,
        staleTime: 60000,
    })
}

// =============================================================================
// Mutations
// =============================================================================

/**
 * Start new fine-tuning job
 */
export function useStartJob() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (request: NewJobRequest) => {
            // API expects snake_case properties
            const config: FineTuneConfig = {
                project_name: request.projectName,
                base_model: request.baseModel,
                dataset_id: request.datasetId,
                epochs: request.epochs,
                learning_rate: request.learningRate,
                rank: request.rank,
                batch_size: request.batchSize,
                use_pissa: request.method === 'pissa',
            }
            return apiClient.startFineTune(config)
        },
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: jobQueryKeys.all })
            toast.success('Training started', `Job ${data.jobId} queued successfully`)
        },
        onError: (error: APIError) => {
            toast.error('Failed to start training', error.message)
        },
    })
}

/**
 * Cancel running job with optimistic update
 */
export function useCancelJob() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (jobId: string) => apiClient.cancelJob(jobId),
        onMutate: async (jobId) => {
            await queryClient.cancelQueries({ queryKey: jobQueryKeys.detail(jobId) })
            const previousJob = queryClient.getQueryData(jobQueryKeys.detail(jobId))

            queryClient.setQueryData(jobQueryKeys.detail(jobId), (old: TrainingJob | undefined) => {
                if (old) {
                    return { ...old, status: 'cancelled' as const }
                }
                return old
            })

            return { previousJob }
        },
        onError: (error: APIError, jobId, context) => {
            if (context?.previousJob) {
                queryClient.setQueryData(jobQueryKeys.detail(jobId), context.previousJob)
            }
            toast.error('Failed to cancel job', error.message)
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: jobQueryKeys.all })
            toast.success('Job cancelled')
        },
    })
}

/**
 * Re-run a job with same or modified config
 */
export function useRerunJob() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async ({ jobId, overrides }: { jobId: string; overrides?: Partial<NewJobRequest> }) => {
            // First get the original job config
            const originalJob = await apiClient.getJob(jobId) as TrainingJob

            const config: FineTuneConfig = {
                project_name: overrides?.projectName || originalJob.projectName,
                base_model: overrides?.baseModel || originalJob.baseModel,
                dataset_id: overrides?.datasetId || originalJob.datasetId,
                epochs: overrides?.epochs || originalJob.epochs,
                learning_rate: overrides?.learningRate || originalJob.learningRate,
                rank: overrides?.rank || originalJob.rank,
                batch_size: overrides?.batchSize || originalJob.batchSize,
                use_pissa: overrides?.method === 'pissa' || true,
            }

            return apiClient.startFineTune(config)
        },
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: jobQueryKeys.all })
            toast.success('Job restarted', `New job ${data.jobId} queued`)
        },
        onError: (error: APIError) => {
            toast.error('Failed to re-run job', error.message)
        },
    })
}

/**
 * Export model to GGUF format
 */
export function useExportModel() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (jobId: string) => apiClient.exportModel(jobId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['models'] })
            toast.success('Model export started', 'You will be notified when complete')
        },
        onError: (error: APIError) => {
            toast.error('Export failed', error.message)
        },
    })
}
