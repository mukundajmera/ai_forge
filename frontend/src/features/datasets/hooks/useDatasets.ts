import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { TrainingDataset, GenerateDatasetRequest, DatasetPreview, GenerationJob } from '@/types'
import { POLLING_INTERVALS } from '@/utils/constants'

// Fetch all datasets
export function useDatasets() {
    return useQuery({
        queryKey: ['datasets'],
        queryFn: () => api.get<TrainingDataset[]>('/datasets'),
    })
}

// Fetch single dataset
export function useDataset(id: string) {
    return useQuery({
        queryKey: ['datasets', id],
        queryFn: () => api.get<TrainingDataset>(`/datasets/${id}`),
        enabled: !!id,
    })
}

// Fetch dataset preview (sample examples)
export function useDatasetPreview(id: string) {
    return useQuery({
        queryKey: ['datasets', id, 'preview'],
        queryFn: () => api.get<DatasetPreview>(`/datasets/${id}/preview`),
        enabled: !!id,
    })
}

// Generate new dataset
export function useGenerateDataset() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (request: GenerateDatasetRequest) =>
            api.post<{ jobId: string; datasetId: string }>('/datasets/generate', request),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['datasets'] })
        },
    })
}

// Poll generation status
export function useGenerationStatus(
    jobId: string | null,
    options: { enabled?: boolean; onComplete?: (job: GenerationJob) => void } = {}
) {
    const { enabled = true, onComplete } = options

    return useQuery({
        queryKey: ['generation', jobId],
        queryFn: async () => {
            if (!jobId) throw new Error('No job ID');
            const job = await api.get<GenerationJob>(`/datasets/generation/${jobId}`);

            if (job.status === 'complete') {
                onComplete?.(job);
            }

            return job;
        },
        enabled: enabled && !!jobId,
        refetchInterval: (query) => {
            const data = query.state.data;
            if (data?.status === 'complete' || data?.status === 'failed') {
                return false;
            }
            return POLLING_INTERVALS.generation;
        },
    })
}

// Delete dataset
export function useDeleteDataset() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (datasetId: string) => api.delete(`/datasets/${datasetId}`),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['datasets'] })
        },
    })
}
