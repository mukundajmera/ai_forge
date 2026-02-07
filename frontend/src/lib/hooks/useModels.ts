// =============================================================================
// Models Hooks - React Query hooks for model operations
// =============================================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, APIError } from '@/lib/api';
import { toast } from '@/components/ui/Toast';

// =============================================================================
// Query Keys
// =============================================================================

export const modelKeys = {
    all: ['models'] as const,
    lists: () => [...modelKeys.all, 'list'] as const,
    list: () => modelKeys.lists(),
    active: () => [...modelKeys.all, 'active'] as const,
    detail: (id: string) => [...modelKeys.all, 'detail', id] as const,
};

// =============================================================================
// Queries
// =============================================================================

/**
 * Fetch all models with 10s polling
 */
export function useModels() {
    return useQuery({
        queryKey: modelKeys.list(),
        queryFn: () => apiClient.getModels(),
        refetchInterval: 10000,
        staleTime: 5000,
    });
}

/**
 * Fetch currently active model
 */
export function useActiveModel() {
    return useQuery({
        queryKey: modelKeys.active(),
        queryFn: () => apiClient.getActiveModel(),
        staleTime: 5000,
    });
}

// =============================================================================
// Mutations
// =============================================================================

/**
 * Deploy model to Ollama
 */
export function useDeployModel() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (modelId: string) => apiClient.deployModel(modelId),
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: modelKeys.all });
            toast.success('Model deployed', `Available as ${data.ollamaName}`);
        },
        onError: (error: APIError) => {
            toast.error('Deployment failed', error.message);
        },
    });
}

/**
 * Set model as active
 */
export function useActivateModel() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (modelId: string) => apiClient.activateModel(modelId),
        onMutate: async (modelId) => {
            // Cancel any outgoing refetches
            await queryClient.cancelQueries({ queryKey: modelKeys.list() });
            await queryClient.cancelQueries({ queryKey: modelKeys.active() });

            // Snapshot current state
            const previousModels = queryClient.getQueryData(modelKeys.list());
            const previousActive = queryClient.getQueryData(modelKeys.active());

            // Optimistically update models list
            queryClient.setQueryData(modelKeys.list(), (old: unknown) => {
                if (!Array.isArray(old)) return old;
                return old.map((model) => ({
                    ...model,
                    isActive: model.id === modelId,
                    status: model.id === modelId ? 'active' : (model.status === 'active' ? 'ready' : model.status),
                }));
            });

            return { previousModels, previousActive };
        },
        onError: (error: APIError, _, context) => {
            // Rollback
            if (context?.previousModels) {
                queryClient.setQueryData(modelKeys.list(), context.previousModels);
            }
            if (context?.previousActive) {
                queryClient.setQueryData(modelKeys.active(), context.previousActive);
            }
            toast.error('Activation failed', error.message);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: modelKeys.all });
            toast.success('Model activated');
        },
    });
}

/**
 * Rollback to a previous model version
 */
export function useRollbackModel() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (modelId: string) => apiClient.rollbackModel(modelId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: modelKeys.all });
            toast.success('Model rolled back');
        },
        onError: (error: APIError) => {
            toast.error('Rollback failed', error.message);
        },
    });
}
