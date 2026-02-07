// =============================================================================
// useMissions Hook Tests
// =============================================================================

import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { server } from '@/mocks/server';
import {
    useMissions,
    usePendingMissions,
    useMission,
    useApproveMission,
    useRejectMission,
    useArtifact,
} from './useMissions';
import React from 'react';

// =============================================================================
// Test Setup
// =============================================================================

const createTestQueryClient = () =>
    new QueryClient({
        defaultOptions: {
            queries: { retry: false, gcTime: 0 },
            mutations: { retry: false },
        },
    });

const createWrapper = () => {
    const queryClient = createTestQueryClient();
    return function Wrapper({ children }: { children: React.ReactNode }) {
        return (
            <QueryClientProvider client={queryClient}>
                {children}
            </QueryClientProvider>
        );
    };
};

// =============================================================================
// Tests
// =============================================================================

describe('useMissions', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('fetches missions successfully', async () => {
        const { result } = renderHook(() => useMissions(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(result.current.data?.missions).toBeDefined();
    });

    it('returns missions with expected structure', async () => {
        const { result } = renderHook(() => useMissions(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        const data = result.current.data;
        expect(data?.missions).toBeDefined();
        expect(Array.isArray(data?.missions)).toBe(true);
        expect(data?.pending).toBeDefined();
    });

    it('applies status filter correctly', async () => {
        const { result } = renderHook(
            () => useMissions({ status: 'pending_approval' }),
            { wrapper: createWrapper() }
        );

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data?.missions).toBeDefined();
    });

    it('handles API errors gracefully', async () => {
        server.use(
            http.get('http://localhost:8000/missions', () => {
                return HttpResponse.json({ message: 'Server error' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useMissions(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('usePendingMissions', () => {
    it('fetches pending missions', async () => {
        const { result } = renderHook(() => usePendingMissions(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(result.current.data?.count).toBeDefined();
    });

    it('returns count of pending missions', async () => {
        const { result } = renderHook(() => usePendingMissions(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(typeof result.current.data?.count).toBe('number');
    });
});

describe('useMission', () => {
    it('fetches single mission by ID', async () => {
        const { result } = renderHook(() => useMission('mission-1'), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(result.current.data?.id).toBe('mission-1');
    });

    it('is disabled when id is undefined', () => {
        const { result } = renderHook(() => useMission(undefined), { wrapper: createWrapper() });

        expect(result.current.isFetching).toBe(false);
    });

    it('handles 404 for non-existent mission', async () => {
        server.use(
            http.get('http://localhost:8000/missions/non-existent', () => {
                return HttpResponse.json({ message: 'Not found' }, { status: 404 });
            })
        );

        const { result } = renderHook(() => useMission('non-existent'), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useApproveMission', () => {
    it('approves mission successfully', async () => {
        const { result } = renderHook(() => useApproveMission(), { wrapper: createWrapper() });

        await result.current.mutateAsync({ id: 'mission-1' });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('approves mission with comment', async () => {
        const { result } = renderHook(() => useApproveMission(), { wrapper: createWrapper() });

        await result.current.mutateAsync({ id: 'mission-1', comment: 'Looks good!' });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('handles approval error', async () => {
        server.use(
            http.post('http://localhost:8000/missions/mission-1/approve', () => {
                return HttpResponse.json({ message: 'Cannot approve' }, { status: 400 });
            })
        );

        const { result } = renderHook(() => useApproveMission(), { wrapper: createWrapper() });

        try {
            await result.current.mutateAsync({ id: 'mission-1' });
        } catch (error) {
            // Expected to throw
        }

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useRejectMission', () => {
    it('rejects mission successfully', async () => {
        const { result } = renderHook(() => useRejectMission(), { wrapper: createWrapper() });

        await result.current.mutateAsync({ id: 'mission-1', reason: 'Not enough data' });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('handles rejection error', async () => {
        server.use(
            http.post('http://localhost:8000/missions/mission-1/reject', () => {
                return HttpResponse.json({ message: 'Cannot reject' }, { status: 400 });
            })
        );

        const { result } = renderHook(() => useRejectMission(), { wrapper: createWrapper() });

        try {
            await result.current.mutateAsync({ id: 'mission-1', reason: 'Reason' });
        } catch (error) {
            // Expected to throw
        }

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useArtifact', () => {
    it('is disabled when id is undefined', () => {
        const { result } = renderHook(() => useArtifact(undefined), { wrapper: createWrapper() });

        expect(result.current.isFetching).toBe(false);
    });
});
