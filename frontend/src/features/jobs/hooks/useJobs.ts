import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import { POLLING_INTERVALS } from '@/utils/constants'

export interface TrainingJob {
    id: string;
    project: string;
    baseModel: string;
    status: 'queued' | 'training' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    currentEpoch?: number;
    totalEpochs: number;
    currentStep?: number;
    totalSteps?: number;
    loss?: number;
    learningRate: number;
    rank: number;
    batchSize: number;
    datasetId: string;
    datasetName: string;
    startedAt: string;
    completedAt?: string;
    duration?: string;
    eta?: string;
    error?: string;
    outputDir?: string;
    checkpoints: string[];
}

export interface NewJobRequest {
    project: string;
    datasetId: string;
    baseModel: string;
    epochs: number;
    learningRate: number;
    rank: number;
    batchSize: number;
    usePissa: boolean;
}

export interface JobLog {
    timestamp: string;
    level: 'info' | 'warning' | 'error';
    message: string;
}

// Fetch all jobs
export function useJobs(filters?: { status?: string; limit?: number }) {
    const params = new URLSearchParams()
    if (filters?.status) params.append('status', filters.status)
    if (filters?.limit) params.append('limit', String(filters.limit))

    return useQuery({
        queryKey: ['jobs', filters],
        queryFn: () => api.get<TrainingJob[]>(`/jobs?${params.toString()}`),
    })
}

// Fetch single job with live updates
export function useJob(jobId: string | undefined) {
    return useQuery({
        queryKey: ['jobs', jobId],
        queryFn: () => api.get<TrainingJob>(`/jobs/${jobId}`),
        enabled: !!jobId,
        refetchInterval: (query) => {
            const job = query.state.data
            // Only poll for active jobs
            if (job?.status === 'training' || job?.status === 'queued') {
                return POLLING_INTERVALS.parsing
            }
            return false
        },
    })
}

// Fetch job logs
export function useJobLogs(jobId: string | undefined, tail = 100) {
    return useQuery({
        queryKey: ['jobs', jobId, 'logs', tail],
        queryFn: () => api.get<JobLog[]>(`/jobs/${jobId}/logs?tail=${tail}`),
        enabled: !!jobId,
        refetchInterval: 3000,
    })
}

// Start new fine-tuning job
export function useStartJob() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (request: NewJobRequest) =>
            api.post<{ jobId: string }>('/jobs', request),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobs'] })
            queryClient.invalidateQueries({ queryKey: ['dashboard'] })
        },
    })
}

// Cancel running job
export function useCancelJob() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (jobId: string) => api.delete(`/jobs/${jobId}`),
        onSuccess: (_, jobId) => {
            queryClient.invalidateQueries({ queryKey: ['jobs', jobId] })
            queryClient.invalidateQueries({ queryKey: ['jobs'] })
        },
    })
}

// Re-run a job with same or modified config
export function useRerunJob() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: ({ jobId, overrides }: { jobId: string; overrides?: Partial<NewJobRequest> }) =>
            api.post<{ jobId: string }>(`/jobs/${jobId}/rerun`, overrides || {}),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobs'] })
        },
    })
}

export type { JobLog, NewJobRequest }
