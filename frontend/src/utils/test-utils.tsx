// =============================================================================
// Test Utilities - Custom render with providers for testing
// =============================================================================

import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';

// =============================================================================
// Query Client Factory
// =============================================================================

/**
 * Create a fresh QueryClient for each test to ensure isolation
 */
export function createTestQueryClient() {
    return new QueryClient({
        defaultOptions: {
            queries: {
                retry: false,
                gcTime: 0,
                staleTime: 0,
            },
            mutations: {
                retry: false,
            },
        },
    });
}

// =============================================================================
// Provider Wrapper
// =============================================================================

interface WrapperProps {
    children: React.ReactNode;
}

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
    initialEntries?: string[];
    queryClient?: QueryClient;
}

/**
 * Custom render function that wraps components with all necessary providers
 */
export function renderWithProviders(
    ui: ReactElement,
    {
        initialEntries = ['/'],
        queryClient = createTestQueryClient(),
        ...renderOptions
    }: CustomRenderOptions = {}
) {
    function Wrapper({ children }: WrapperProps) {
        return (
            <QueryClientProvider client={queryClient}>
                <MemoryRouter initialEntries={initialEntries}>
                    {children}
                </MemoryRouter>
            </QueryClientProvider>
        );
    }

    return {
        ...render(ui, { wrapper: Wrapper, ...renderOptions }),
        queryClient,
    };
}

/**
 * Wrapper for components that use BrowserRouter (for testing hooks)
 */
export function createWrapper(queryClient?: QueryClient) {
    const client = queryClient || createTestQueryClient();

    return function Wrapper({ children }: WrapperProps) {
        return (
            <QueryClientProvider client={client}>
                <BrowserRouter>{children}</BrowserRouter>
            </QueryClientProvider>
        );
    };
}

// =============================================================================
// Test Helpers
// =============================================================================

/**
 * Wait for a specific amount of time (use sparingly)
 */
export function wait(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Create a mock function that resolves after a delay
 */
export function createDelayedMock<T>(value: T, delay = 100) {
    return async () => {
        await wait(delay);
        return value;
    };
}

/**
 * Create a mock function that rejects after a delay
 */
export function createDelayedRejectMock(error: Error, delay = 100) {
    return async () => {
        await wait(delay);
        throw error;
    };
}

// Re-export everything from testing-library
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
