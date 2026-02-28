import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'

export const experimentQueryKeys = {
    all: ['experiments'] as const,
    list: (filters?: { status?: string; tag?: string }) =>
        ['experiments', 'list', filters] as const,
    detail: (id: string) => ['experiments', 'detail', id] as const,
    comparison: (ids: string[]) => ['experiments', 'compare', ids] as const,
}

export function useExperiments(filters?: { status?: string; tag?: string }) {
    return useQuery({
        queryKey: experimentQueryKeys.list(filters),
        queryFn: () => apiClient.getExperiments(filters),
        staleTime: 5000,
    })
}

export function useExperiment(id: string | undefined) {
    return useQuery({
        queryKey: experimentQueryKeys.detail(id!),
        queryFn: () => apiClient.getExperiment(id!),
        enabled: !!id,
        staleTime: 5000,
    })
}

export function useCompareExperiments(ids: string[]) {
    return useQuery({
        queryKey: experimentQueryKeys.comparison(ids),
        queryFn: () => apiClient.compareExperiments(ids),
        enabled: ids.length >= 2,
        staleTime: 10000,
    })
}

export function useCreateExperiment() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (data: {
            name: string
            description?: string
            base_model: string
            dataset_id?: string
            recipe_id?: string
            hyperparameters?: Record<string, unknown>
            tags?: string[]
        }) => apiClient.createExperiment(data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: experimentQueryKeys.all })
        },
    })
}

export function useDeleteExperiment() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (id: string) => apiClient.deleteExperiment(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: experimentQueryKeys.all })
        },
    })
}
