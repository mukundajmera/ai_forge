// =============================================================================
// Central Hooks Re-exports for lib/hooks
// =============================================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useState } from 'react';

// Re-export from feature modules
export {
    useJobs,
    useJob,
    useJobMetrics,
    useJobLogs,
    useValidation,
    useStartJob,
    useCancelJob,
    useRerunJob,
    useExportModel,
    jobQueryKeys,
} from '@/features/jobs/hooks/useJobs';

// =============================================================================
// Dataset Hooks
// =============================================================================

import { apiClient } from '@/api/client';
import type { Model } from '@/types';

export function useDatasets() {
    return useQuery({
        queryKey: ['datasets'],
        queryFn: () => apiClient.getDatasets(),
    });
}

export function useStartFineTune() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: apiClient.startFineTune.bind(apiClient),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobs'] });
        },
    });
}

// =============================================================================
// Model Hooks
// =============================================================================

export function useModels() {
    return useQuery({
        queryKey: ['models'],
        queryFn: () => apiClient.getModels() as Promise<Model[]>,
    });
}

export function useActiveModel() {
    const { data: models, ...rest } = useModels();
    const activeModel = models?.find((m) => m.isActive || m.status === 'active');
    return { data: activeModel, models, ...rest };
}

// =============================================================================
// System Hooks
// =============================================================================

export function useSystemStatus() {
    return useQuery({
        queryKey: ['system', 'status'],
        queryFn: () => apiClient.getSystemStatus(),
        refetchInterval: 30000,
    });
}

// =============================================================================
// Mission Hooks
// =============================================================================

import type { Mission } from '@/lib/types/mission.types';

export function useMissions() {
    return useQuery({
        queryKey: ['missions'],
        queryFn: async () => {
            // Return mock data for now - missions API may not be implemented
            return [] as Mission[];
        },
    });
}

export function useMission(missionId: string | undefined) {
    return useQuery({
        queryKey: ['missions', missionId],
        queryFn: async () => {
            // Return mock data for now
            return null as Mission | null;
        },
        enabled: !!missionId,
    });
}

export function useApproveMission() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (_missionId: string) => {
            // Mock implementation
            return { success: true };
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['missions'] });
        },
    });
}

export function useRejectMission() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (_missionId: string) => {
            // Mock implementation
            return { success: true };
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['missions'] });
        },
    });
}

// =============================================================================
// Utility Hooks
// =============================================================================

export function useKeyboardShortcut(key: string, callback: () => void, deps: unknown[] = []) {
    useEffect(() => {
        const handler = (event: KeyboardEvent) => {
            if (event.key === key && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                callback();
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [key, callback, ...deps]);
}

// =============================================================================
// Log Stream Hook
// =============================================================================

export interface LogEntry {
    timestamp: string;
    level: 'info' | 'warning' | 'error' | 'debug';
    message: string;
}

export function useLogStream(jobId: string | undefined) {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const connect = useCallback(() => {
        if (!jobId) return;

        // For now, use polling instead of SSE
        const interval = setInterval(async () => {
            try {
                const result = await apiClient.getJobLogs(jobId);
                setLogs(
                    result.logs.map((msg) => ({
                        timestamp: result.lastUpdated,
                        level: 'info' as const,
                        message: msg,
                    }))
                );
                setIsConnected(true);
            } catch (err) {
                setError(err as Error);
                setIsConnected(false);
            }
        }, 3000);

        return () => clearInterval(interval);
    }, [jobId]);

    useEffect(() => {
        const cleanup = connect();
        return cleanup;
    }, [connect]);

    return { logs, isConnected, error, connect };
}
