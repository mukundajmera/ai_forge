// =============================================================================
// Missions React Query Hooks - Antigravity/Repo Guardian Integration
// =============================================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, APIError } from '@/lib/api/client';
import { toast } from '@/components/ui/Toast';
import type { MissionStatus, MissionType, Mission, MissionsResponse } from '@/lib/types';

// =============================================================================
// Query Keys
// =============================================================================

export const missionKeys = {
    all: ['missions'] as const,
    list: (filter?: { status?: MissionStatus; type?: MissionType }) =>
        [...missionKeys.all, 'list', filter] as const,
    detail: (id: string) => [...missionKeys.all, 'detail', id] as const,
    pending: () => [...missionKeys.all, 'pending'] as const,
};

export const artifactKeys = {
    all: ['artifacts'] as const,
    detail: (id: string) => [...artifactKeys.all, 'detail', id] as const,
    content: (id: string) => [...artifactKeys.all, 'content', id] as const,
};

// =============================================================================
// Mission Queries
// =============================================================================

/**
 * Fetch list of missions with optional filters
 * Polls every 10 seconds for new missions
 */
export function useMissions(filter?: { status?: MissionStatus; type?: MissionType }) {
    return useQuery({
        queryKey: missionKeys.list(filter),
        queryFn: () => apiClient.getMissions(filter),
        refetchInterval: 10000, // Poll for new missions every 10s
        staleTime: 5000,
    });
}

/**
 * Fetch pending missions only (for dashboard/topbar indicators)
 */
export function usePendingMissions() {
    return useQuery({
        queryKey: missionKeys.pending(),
        queryFn: () => apiClient.getMissions({ status: 'pending_approval' }),
        refetchInterval: 10000,
        staleTime: 5000,
        select: (data: MissionsResponse) => ({
            missions: data.missions,
            count: data.pending,
        }),
    });
}

/**
 * Fetch single mission detail
 */
export function useMission(id: string | undefined) {
    return useQuery({
        queryKey: missionKeys.detail(id!),
        queryFn: () => apiClient.getMission(id!),
        enabled: !!id,
        staleTime: 2000,
    });
}

// =============================================================================
// Mission Mutations
// =============================================================================

/**
 * Approve a pending mission
 */
export function useApproveMission() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ id, comment }: { id: string; comment?: string }) =>
            apiClient.approveMission(id, comment),
        onMutate: async ({ id }) => {
            // Cancel any outgoing refetches
            await queryClient.cancelQueries({ queryKey: missionKeys.detail(id) });
            await queryClient.cancelQueries({ queryKey: missionKeys.list() });

            // Snapshot previous value
            const previousMission = queryClient.getQueryData(missionKeys.detail(id));

            // Optimistically update
            queryClient.setQueryData(missionKeys.detail(id), (old: Mission | undefined) => {
                if (!old) return old;
                return {
                    ...old,
                    status: 'approved' as const,
                    approval: {
                        ...old.approval,
                        approvedAt: new Date().toISOString(),
                    },
                };
            });

            return { previousMission };
        },
        onError: (error: APIError, { id }, context) => {
            // Rollback on error
            if (context?.previousMission) {
                queryClient.setQueryData(missionKeys.detail(id), context.previousMission);
            }
            toast.error('Failed to approve mission', error.message);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: missionKeys.all });
            toast.success('Mission approved', 'The agent will begin execution.');
        },
    });
}

/**
 * Reject a pending mission
 */
export function useRejectMission() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ id, reason }: { id: string; reason: string }) =>
            apiClient.rejectMission(id, reason),
        onMutate: async ({ id, reason }) => {
            await queryClient.cancelQueries({ queryKey: missionKeys.detail(id) });

            const previousMission = queryClient.getQueryData(missionKeys.detail(id));

            queryClient.setQueryData(missionKeys.detail(id), (old: Mission | undefined) => {
                if (!old) return old;
                return {
                    ...old,
                    status: 'rejected' as const,
                    approval: {
                        ...old.approval,
                        rejectedAt: new Date().toISOString(),
                        rejectionReason: reason,
                    },
                };
            });

            return { previousMission };
        },
        onError: (error: APIError, { id }, context) => {
            if (context?.previousMission) {
                queryClient.setQueryData(missionKeys.detail(id), context.previousMission);
            }
            toast.error('Failed to reject mission', error.message);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: missionKeys.all });
            toast.success('Mission rejected', 'Your feedback helps improve future suggestions.');
        },
    });
}

// =============================================================================
// Artifact Queries
// =============================================================================

/**
 * Fetch artifact metadata
 */
export function useArtifact(id: string | undefined) {
    return useQuery({
        queryKey: artifactKeys.detail(id!),
        queryFn: () => apiClient.getArtifact(id!),
        enabled: !!id,
    });
}

/**
 * Fetch artifact content (for inline rendering)
 */
export function useArtifactContent(id: string | undefined) {
    return useQuery({
        queryKey: artifactKeys.content(id!),
        queryFn: () => apiClient.getArtifactContent(id!),
        enabled: !!id,
    });
}
