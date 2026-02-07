// =============================================================================
// useModels Hook Tests
// =============================================================================

import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { server } from '@/mocks/server';
import {
    useModels,
    useActiveModel,
    useDeployModel,
    useActivateModel,
    useRollbackModel,
} from './useModels';
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

describe('useModels', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('fetches models successfully', async () => {
        const { result } = renderHook(() => useModels(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
        expect(Array.isArray(result.current.data)).toBe(true);
    });

    it('returns models with expected structure', async () => {
        const { result } = renderHook(() => useModels(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        const models = result.current.data;
        expect(models?.length).toBeGreaterThan(0);
        expect(models?.[0]).toHaveProperty('id');
        expect(models?.[0]).toHaveProperty('name');
        expect(models?.[0]).toHaveProperty('status');
    });

    it('handles API errors gracefully', async () => {
        server.use(
            http.get('http://localhost:8000/models', () => {
                return HttpResponse.json({ message: 'Server error' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useModels(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useActiveModel', () => {
    it('fetches active model successfully', async () => {
        const { result } = renderHook(() => useActiveModel(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
    });

    it('returns active model with status active', async () => {
        const { result } = renderHook(() => useActiveModel(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data?.status).toBe('active');
    });

    it.skip('handles no active model gracefully', async () => {
        server.use(
            http.get('http://localhost:8000/models/active', () => {
                return HttpResponse.json({ message: 'Not found' }, { status: 404 });
            })
        );

        const { result } = renderHook(() => useActiveModel(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isLoading).toBe(false), { timeout: 5000 });
        expect(result.current.isError).toBe(true);
    });
});

describe('useDeployModel', () => {
    it('deploys model successfully', async () => {
        const { result } = renderHook(() => useDeployModel(), { wrapper: createWrapper() });

        await result.current.mutateAsync('model-1');

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('handles deployment error', async () => {
        server.use(
            http.post('http://localhost:8000/models/model-1/deploy', () => {
                return HttpResponse.json({ message: 'Deployment failed' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useDeployModel(), { wrapper: createWrapper() });

        try {
            await result.current.mutateAsync('model-1');
        } catch (error) {
            // Expected to throw
        }

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useActivateModel', () => {
    it('activates model successfully', async () => {
        const { result } = renderHook(() => useActivateModel(), { wrapper: createWrapper() });

        await result.current.mutateAsync('model-2');

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('handles activation error', async () => {
        server.use(
            http.post('http://localhost:8000/models/model-1/activate', () => {
                return HttpResponse.json({ message: 'Cannot activate' }, { status: 400 });
            })
        );

        const { result } = renderHook(() => useActivateModel(), { wrapper: createWrapper() });

        try {
            await result.current.mutateAsync('model-1');
        } catch (error) {
            // Expected to throw
        }

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useRollbackModel', () => {
    it('rolls back model successfully', async () => {
        const { result } = renderHook(() => useRollbackModel(), { wrapper: createWrapper() });

        await result.current.mutateAsync('model-1');

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
    });

    it('handles rollback error', async () => {
        server.use(
            http.post('http://localhost:8000/models/model-1/rollback', () => {
                return HttpResponse.json({ message: 'Cannot rollback' }, { status: 400 });
            })
        );

        const { result } = renderHook(() => useRollbackModel(), { wrapper: createWrapper() });

        try {
            await result.current.mutateAsync('model-1');
        } catch (error) {
            // Expected to throw
        }

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});
