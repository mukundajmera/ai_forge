// =============================================================================
// useDatasets Hook Tests
// =============================================================================

import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { server } from '@/mocks/server';
import {
    useDataSources,
    useDatasets,
    useDataset,
    useGenerateDataset,
    useDeleteDataSource,
} from './useDatasets';
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

describe('useDataSources', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('fetches data sources successfully', async () => {
        const { result } = renderHook(() => useDataSources(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(Array.isArray(result.current.data)).toBe(true);
    });

    it('returns data sources with expected structure', async () => {
        const { result } = renderHook(() => useDataSources(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        const sources = result.current.data;
        expect(sources?.length).toBeGreaterThan(0);
        expect(sources?.[0]).toHaveProperty('id');
        expect(sources?.[0]).toHaveProperty('name');
        expect(sources?.[0]).toHaveProperty('type');
    });

    it('handles API errors gracefully', async () => {
        server.use(
            http.get('http://localhost:8000/api/data-sources', () => {
                return HttpResponse.json({ message: 'Server error' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useDataSources(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useDatasets', () => {
    it('fetches datasets successfully', async () => {
        const { result } = renderHook(() => useDatasets(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(Array.isArray(result.current.data)).toBe(true);
    });

    it('returns datasets with expected structure', async () => {
        const { result } = renderHook(() => useDatasets(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        const datasets = result.current.data;
        expect(datasets?.length).toBeGreaterThan(0);
        expect(datasets?.[0]).toHaveProperty('id');
        expect(datasets?.[0]).toHaveProperty('name');
        expect(datasets?.[0]).toHaveProperty('exampleCount');
    });
});

describe('useDataset', () => {
    it('fetches single dataset by ID', async () => {
        const { result } = renderHook(() => useDataset('ds-1'), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(result.current.data?.id).toBe('ds-1');
    });

    it('is disabled when datasetId is undefined', () => {
        const { result } = renderHook(() => useDataset(undefined), { wrapper: createWrapper() });

        expect(result.current.isFetching).toBe(false);
    });

    it('handles 404 for non-existent dataset', async () => {
        server.use(
            http.get('http://localhost:8000/api/datasets/non-existent', () => {
                return HttpResponse.json({ message: 'Not found' }, { status: 404 });
            })
        );

        const { result } = renderHook(() => useDataset('non-existent'), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useGenerateDataset', () => {
    it('generates dataset successfully', async () => {
        const { result } = renderHook(() => useGenerateDataset(), { wrapper: createWrapper() });

        await result.current.mutateAsync({
            name: 'test-dataset',
            sourceIds: ['src-1'],
            format: 'alpaca',
        });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('returns dataset ID on success', async () => {
        const { result } = renderHook(() => useGenerateDataset(), { wrapper: createWrapper() });

        const response = await result.current.mutateAsync({
            name: 'test-dataset',
            sourceIds: ['src-1'],
            format: 'alpaca',
        });

        expect(response).toHaveProperty('datasetId');
    });

    it('handles generation error', async () => {
        server.use(
            http.post('http://localhost:8000/api/datasets/generate', () => {
                return HttpResponse.json({ message: 'Generation failed' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useGenerateDataset(), { wrapper: createWrapper() });

        try {
            await result.current.mutateAsync({
                name: 'test-dataset',
                sourceIds: ['src-1'],
                format: 'alpaca',
            });
        } catch (error) {
            // Expected to throw
        }

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useDeleteDataSource', () => {
    it('deletes data source successfully', async () => {
        const { result } = renderHook(() => useDeleteDataSource(), { wrapper: createWrapper() });

        await result.current.mutateAsync('src-1');

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('handles delete error', async () => {
        server.use(
            http.delete('http://localhost:8000/api/data-sources/src-1', () => {
                return HttpResponse.json({ message: 'Cannot delete' }, { status: 400 });
            })
        );

        const { result } = renderHook(() => useDeleteDataSource(), { wrapper: createWrapper() });

        try {
            await result.current.mutateAsync('src-1');
        } catch (error) {
            // Expected to throw
        }

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});
