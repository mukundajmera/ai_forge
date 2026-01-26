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

import type { Mission, MissionsResponse } from '@/lib/types/mission.types';

export function useMissions(filter?: { status?: string; type?: string }) {
    return useQuery({
        queryKey: ['missions', filter],
        queryFn: () => apiClient.getMissions(filter) as Promise<MissionsResponse>,
    });
}

export function useMission(missionId: string | undefined) {
    return useQuery({
        queryKey: ['missions', missionId],
        queryFn: () => apiClient.getMission(missionId!) as Promise<Mission>,
        enabled: !!missionId,
    });
}

export function useApproveMission() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (missionId: string) => apiClient.approveMission(missionId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['missions'] });
        },
    });
}

export function useRejectMission() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({ missionId, reason }: { missionId: string; reason?: string }) => 
            apiClient.rejectMission(missionId, reason),
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
