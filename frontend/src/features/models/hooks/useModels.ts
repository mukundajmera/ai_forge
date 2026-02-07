// =============================================================================
// Models Feature Hooks - React Query hooks for model operations
// =============================================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, APIError } from '@/lib/api'
import { toast } from '@/components/ui/Toast'

// =============================================================================
// Types
// =============================================================================

export interface Model {
    id: string
    name: string
    version: string
    baseModel: string
    status: 'ready' | 'deploying' | 'active' | 'error'
    isActive: boolean
    jobId?: string
    createdAt: string
    deployedAt?: string
    metrics: {
        codeBleu: number
        humanEval: number
        perplexity: number
        avgLatency: number
    }
    size: number // bytes
    quantization: string
    ollamaName?: string
}

export interface DeployRequest {
    jobId: string
    modelName: string
    quantization: string
    systemPrompt?: string
}

// =============================================================================
// Query Keys
// =============================================================================

export const modelQueryKeys = {
    all: ['models'] as const,
    list: () => ['models', 'list'] as const,
    detail: (id: string) => ['models', 'detail', id] as const,
    active: () => ['models', 'active'] as const,
}

// =============================================================================
// Queries
// =============================================================================

/**
 * List all models with 10s polling
 */
export function useModels() {
    return useQuery({
        queryKey: modelQueryKeys.list(),
        queryFn: () => apiClient.getModels() as Promise<Model[]>,
        refetchInterval: 10000,
        staleTime: 5000,
    })
}

/**
 * Get single model by ID
 */
export function useModel(modelId: string | undefined) {
    return useQuery({
        queryKey: modelQueryKeys.detail(modelId!),
        queryFn: async () => {
            const models = await apiClient.getModels() as Model[]
            const model = models.find(m => m.id === modelId)
            if (!model) {
                throw new APIError(404, 'Model not found')
            }
            return model
        },
        enabled: !!modelId,
    })
}

/**
 * Get active model
 */
export function useActiveModel() {
    return useQuery({
        queryKey: modelQueryKeys.active(),
        queryFn: () => apiClient.getActiveModel() as Promise<Model | null>,
        staleTime: 5000,
    })
}

// =============================================================================
// Mutations
// =============================================================================

/**
 * Deploy model to Ollama
 */
export function useDeployModel() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (modelId: string) => apiClient.deployModel(modelId),
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: modelQueryKeys.all })
            toast.success('Model deployed', `Available as ${data.ollamaName}`)
        },
        onError: (error: APIError) => {
            toast.error('Deployment failed', error.message)
        },
    })
}

/**
 * Activate model (set as default) with optimistic update
 */
export function useActivateModel() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (modelId: string) => apiClient.activateModel(modelId),
        onMutate: async (modelId) => {
            // Cancel outgoing refetches
            await queryClient.cancelQueries({ queryKey: modelQueryKeys.list() })
            await queryClient.cancelQueries({ queryKey: modelQueryKeys.active() })

            // Snapshot current state
            const previousModels = queryClient.getQueryData(modelQueryKeys.list())
            const previousActive = queryClient.getQueryData(modelQueryKeys.active())

            // Optimistically update
            queryClient.setQueryData(modelQueryKeys.list(), (old: Model[] | undefined) => {
                if (!old) return old
                return old.map(model => ({
                    ...model,
                    isActive: model.id === modelId,
                    status: model.id === modelId ? 'active' as const :
                        (model.status === 'active' ? 'ready' as const : model.status),
                }))
            })

            return { previousModels, previousActive }
        },
        onError: (error: APIError, _, context) => {
            // Rollback
            if (context?.previousModels) {
                queryClient.setQueryData(modelQueryKeys.list(), context.previousModels)
            }
            if (context?.previousActive) {
                queryClient.setQueryData(modelQueryKeys.active(), context.previousActive)
            }
            toast.error('Activation failed', error.message)
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: modelQueryKeys.all })
            toast.success('Model activated')
        },
    })
}

/**
 * Rollback to a previous model version
 */
export function useRollbackModel() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (modelId: string) => apiClient.rollbackModel(modelId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: modelQueryKeys.all })
            toast.success('Model rolled back')
        },
        onError: (error: APIError) => {
            toast.error('Rollback failed', error.message)
        },
    })
}

/**
 * Delete model (unload from Ollama)
 */
export function useDeleteModel() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: async (_modelId: string) => {
            // Note: API doesn't have a delete endpoint yet, 
            // this would be implemented when backend supports it
            throw new APIError(501, 'Model deletion not yet implemented')
        },
        onMutate: async (modelId) => {
            await queryClient.cancelQueries({ queryKey: modelQueryKeys.list() })
            const previousModels = queryClient.getQueryData(modelQueryKeys.list())

            queryClient.setQueryData(modelQueryKeys.list(), (old: Model[] | undefined) => {
                if (!old) return old
                return old.filter(m => m.id !== modelId)
            })

            return { previousModels }
        },
        onError: (error: APIError, _, context) => {
            if (context?.previousModels) {
                queryClient.setQueryData(modelQueryKeys.list(), context.previousModels)
            }
            toast.error('Delete failed', error.message)
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: modelQueryKeys.all })
            toast.success('Model deleted')
        },
    })
}
