// =============================================================================
// useSystem Hook Tests
// =============================================================================

import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { server } from '@/mocks/server';
import {
    useSystemStatus,
    useHealthCheck,
    useBackendConnection,
} from './useSystem';
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

describe('useSystemStatus', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('fetches system status successfully', async () => {
        const { result } = renderHook(() => useSystemStatus(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
    });

    it('returns system status with expected structure', async () => {
        const { result } = renderHook(() => useSystemStatus(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        const status = result.current.data;
        expect(status).toHaveProperty('healthy');
        expect(status).toHaveProperty('cpu');
        expect(status).toHaveProperty('memory');
        expect(status).toHaveProperty('gpu');
    });

    it('returns GPU information', async () => {
        const { result } = renderHook(() => useSystemStatus(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        const status = result.current.data;
        expect(status?.gpu).toHaveProperty('name');
        expect(status?.gpu).toHaveProperty('memoryUsed');
        expect(status?.gpu).toHaveProperty('memoryTotal');
    });

    it('returns Ollama status', async () => {
        const { result } = renderHook(() => useSystemStatus(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        const status = result.current.data;
        expect(status?.ollama).toHaveProperty('status');
    });

    it('handles API errors gracefully', async () => {
        server.use(
            http.get('http://localhost:8000/status', () => {
                return HttpResponse.json({ message: 'Server error' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useSystemStatus(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isError).toBe(true));
    });
});

describe('useHealthCheck', () => {
    it('fetches health check successfully', async () => {
        const { result } = renderHook(() => useHealthCheck(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));
        expect(result.current.data).toBeDefined();
    });

    it('returns health status', async () => {
        const { result } = renderHook(() => useHealthCheck(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        expect(result.current.data?.status).toBe('healthy');
    });

    it('returns health checks', async () => {
        const { result } = renderHook(() => useHealthCheck(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        expect(result.current.data?.checks).toBeDefined();
        expect(result.current.data?.checks).toHaveProperty('database');
        expect(result.current.data?.checks).toHaveProperty('ollama');
    });

    it('handles network errors', async () => {
        server.use(
            http.get('http://localhost:8000/health', () => {
                return HttpResponse.json({ message: 'Server error' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useHealthCheck(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isLoading).toBe(false), { timeout: 5000 });
        expect(result.current.isError).toBe(true);
    });
});

describe('useBackendConnection', () => {
    it('returns isConnected true when healthy', async () => {
        const { result } = renderHook(() => useBackendConnection(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isConnected).toBe(true));
    });

    it('returns isConnected false when unhealthy', async () => {
        server.use(
            http.get('http://localhost:8000/health', () => {
                return HttpResponse.json({ status: 'unhealthy' });
            })
        );

        const { result } = renderHook(() => useBackendConnection(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(result.current.isConnected).toBe(false);
    });

    it('returns isError true on network error', async () => {
        server.use(
            http.get('http://localhost:8000/health', () => {
                return HttpResponse.json({ message: 'Server error' }, { status: 500 });
            })
        );

        const { result } = renderHook(() => useBackendConnection(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isLoading).toBe(false), { timeout: 5000 });
        expect(result.current.isError).toBe(true);
    });

    it('returns checks object', async () => {
        const { result } = renderHook(() => useBackendConnection(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isConnected).toBe(true));
        expect(result.current.checks).toBeDefined();
    });

    it('provides refetch function', async () => {
        const { result } = renderHook(() => useBackendConnection(), { wrapper: createWrapper() });

        await waitFor(() => expect(result.current.isConnected).toBe(true));
        expect(typeof result.current.refetch).toBe('function');
    });
});
