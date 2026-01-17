import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { DataSource, AddDataSourceRequest, ParsedFile } from '@/types'

// Fetch all data sources
export function useDataSources() {
    return useQuery({
        queryKey: ['data-sources'],
        queryFn: () => api.get<DataSource[]>('/data-sources'),
    })
}

// Fetch single data source
export function useDataSource(id: string) {
    return useQuery({
        queryKey: ['data-sources', id],
        queryFn: () => api.get<DataSource>(`/data-sources/${id}`),
        enabled: !!id,
    })
}

// Fetch files for a data source
export function useDataSourceFiles(sourceId: string) {
    return useQuery({
        queryKey: ['data-sources', sourceId, 'files'],
        queryFn: () => api.get<ParsedFile[]>(`/data-sources/${sourceId}/files`),
        enabled: !!sourceId,
    })
}

// Add new data source
export function useAddDataSource() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (request: AddDataSourceRequest) =>
            api.post<DataSource>('/data-sources', request),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['data-sources'] })
        },
    })
}

// Sync/refresh data source
export function useSyncDataSource() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (sourceId: string) =>
            api.post<{ jobId: string }>(`/data-sources/${sourceId}/sync`),
        onSuccess: (_, sourceId) => {
            queryClient.invalidateQueries({ queryKey: ['data-sources', sourceId] })
            queryClient.invalidateQueries({ queryKey: ['data-sources'] })
        },
    })
}

// Delete data source
export function useDeleteDataSource() {
    const queryClient = useQueryClient()

    return useMutation({
        mutationFn: (sourceId: string) =>
            api.delete(`/data-sources/${sourceId}`),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['data-sources'] })
        },
    })
}
