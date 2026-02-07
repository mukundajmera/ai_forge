// =============================================================================
// System Hooks - React Query hooks for system status and health checks
// =============================================================================

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api';

// =============================================================================
// Query Keys
// =============================================================================

export const systemKeys = {
    all: ['system'] as const,
    status: () => [...systemKeys.all, 'status'] as const,
    health: () => [...systemKeys.all, 'health'] as const,
};

// =============================================================================
// Queries
// =============================================================================

/**
 * Fetch system status including GPU, CPU, memory, and Ollama info
 */
export function useSystemStatus() {
    return useQuery({
        queryKey: systemKeys.status(),
        queryFn: () => apiClient.getSystemStatus(),
        refetchInterval: 10000, // Poll every 10s
        staleTime: 5000,
    });
}

/**
 * Health check for connection status
 */
export function useHealthCheck() {
    return useQuery({
        queryKey: systemKeys.health(),
        queryFn: () => apiClient.healthCheck(),
        refetchInterval: 30000, // Poll every 30s
        staleTime: 15000,
        retry: 1,
    });
}

/**
 * Check if backend is reachable
 */
export function useBackendConnection() {
    const { data, isLoading, isError, error, refetch } = useHealthCheck();

    return {
        isConnected: !!data && data.status === 'healthy',
        isLoading,
        isError,
        error,
        refetch,
        status: data?.status || 'unknown',
        checks: data?.checks,
    };
}
