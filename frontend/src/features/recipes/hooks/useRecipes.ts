import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'

export const recipeQueryKeys = {
    all: ['recipes'] as const,
    list: (filters?: { task_type?: string; tag?: string }) =>
        ['recipes', 'list', filters] as const,
    detail: (id: string, hardware?: string) =>
        ['recipes', 'detail', id, hardware] as const,
}

export function useRecipes(filters?: { task_type?: string; tag?: string }) {
    return useQuery({
        queryKey: recipeQueryKeys.list(filters),
        queryFn: () => apiClient.getRecipes(filters),
        staleTime: 30000,
    })
}

export function useRecipe(id: string | undefined, hardware?: string) {
    return useQuery({
        queryKey: recipeQueryKeys.detail(id!, hardware),
        queryFn: () => apiClient.getRecipe(id!, hardware),
        enabled: !!id,
        staleTime: 30000,
    })
}
