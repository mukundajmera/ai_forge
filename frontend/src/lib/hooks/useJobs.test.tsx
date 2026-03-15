// =============================================================================
// useJobs Hook Tests
// =============================================================================

import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { server } from '@/mocks/server';
import {
    useJobs,
    useJob,
    useJobMetrics,
    useJobLogs,
    useStartFineTune,
    useCancelJob,
    useExportModel,
} from './useJobs';
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

describe('useJobs', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('fetches jobs successfully', async () => {
        const { result } = renderHook(() => useJobs(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(Array.isArray(result.current.data)).toBe(true);
    });

    it('returns jobs data with expected structure', async () => {
        const { result } = renderHook(() => useJobs(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        const jobs = result.current.data;
        expect(jobs?.length).toBeGreaterThan(0);
        expect(jobs?.[0]).toHaveProperty('id');
        expect(jobs?.[0]).toHaveProperty('status');
    });

    it('handles API errors gracefully', async () => {
        server.use(
            http.get('http://localhost:8000/jobs', () => {
                return HttpResponse.json({ message: 'Server error' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useJobs(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useJob', () => {
    it('fetches single job by ID', async () => {
        const { result } = renderHook(() => useJob('job-1'), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(result.current.data?.id).toBe('job-1');
    });

    it('is disabled when jobId is undefined', () => {
        const { result } = renderHook(() => useJob(undefined), { wrapper: createWrapper() });

        expect(result.current.isFetching).toBe(false);
    });

    it('handles 404 for non-existent job', async () => {
        server.use(
            http.get('http://localhost:8000/jobs/non-existent', () => {
                return HttpResponse.json({ message: 'Not found' }, { status: 404 });
            })
        );

        const { result } = renderHook(() => useJob('non-existent'), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useJobMetrics', () => {
    it('fetches job metrics', async () => {
        const { result } = renderHook(() => useJobMetrics('job-1'), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toHaveProperty('steps');
        expect(result.current.data).toHaveProperty('losses');
    });

    it('is disabled when jobId is undefined', () => {
        const { result } = renderHook(() => useJobMetrics(undefined), { wrapper: createWrapper() });

        expect(result.current.isFetching).toBe(false);
    });
});

describe('useJobLogs', () => {
    it('fetches job logs', async () => {
        const { result } = renderHook(() => useJobLogs('job-1'), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toHaveProperty('logs');
    });

    it('is disabled when enabled is false', () => {
        const { result } = renderHook(() => useJobLogs('job-1', false), { wrapper: createWrapper() });

        expect(result.current.isFetching).toBe(false);
    });
});

describe('useStartFineTune', () => {
    it('starts fine-tune successfully', async () => {
        const { result } = renderHook(() => useStartFineTune(), { wrapper: createWrapper() });

        await result.current.mutateAsync({
            project_name: 'test-project',
            dataset_id: 'ds-1',
            base_model: 'Llama-3.2-3B',
            epochs: 3,
            learning_rate: 0.0001,
            rank: 64,
            batch_size: 4,
            use_pissa: true,
        });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('returns job ID on success', async () => {
        const { result } = renderHook(() => useStartFineTune(), { wrapper: createWrapper() });

        const response = await result.current.mutateAsync({
            project_name: 'test-project',
            dataset_id: 'ds-1',
            base_model: 'Llama-3.2-3B',
            epochs: 3,
            learning_rate: 0.0001,
            rank: 64,
            batch_size: 4,
            use_pissa: true,
        });

        expect(response).toHaveProperty('jobId');
    });
});

describe('useCancelJob', () => {
    it('cancels job successfully', async () => {
        const { result } = renderHook(() => useCancelJob(), { wrapper: createWrapper() });

        await result.current.mutateAsync('job-1');

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('handles cancel error', async () => {
        server.use(
            http.delete('http://localhost:8000/jobs/job-1', () => {
                return HttpResponse.json({ message: 'Cannot cancel completed job' }, { status: 400 });
            })
        );

        const { result } = renderHook(() => useCancelJob(), { wrapper: createWrapper() });

        try {
            await result.current.mutateAsync('job-1');
        } catch (error) {
            // Expected to throw
        }

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useExportModel', () => {
    it('exports model successfully', async () => {
        const { result } = renderHook(() => useExportModel(), { wrapper: createWrapper() });

        await result.current.mutateAsync('job-1');

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });
});
