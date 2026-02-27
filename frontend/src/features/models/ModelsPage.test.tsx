// =============================================================================
// ModelsPage Tests - Button props and rendering
// =============================================================================

import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { server } from '@/mocks/server';
import { ModelsPage } from './ModelsPage';

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

const renderWithProviders = (ui: React.ReactElement) => {
    const queryClient = createTestQueryClient();
    return {
        ...render(
            <QueryClientProvider client={queryClient}>
                <MemoryRouter>
                    {ui}
                </MemoryRouter>
            </QueryClientProvider>
        ),
        queryClient,
    };
};

// =============================================================================
// Tests
// =============================================================================

describe('ModelsPage', () => {
    it('shows loading state initially', () => {
        renderWithProviders(<ModelsPage />);
        expect(screen.getByText('Loading models...')).toBeInTheDocument();
    });

    it('renders empty state with Start Training link when no models', async () => {
        // Set up handler that returns empty model list
        server.use(
            http.get('http://localhost:8000/v1/models', () => {
                return HttpResponse.json({ data: [] });
            })
        );

        renderWithProviders(<ModelsPage />);

        const startTrainingLink = await screen.findByText('Start Training');
        expect(startTrainingLink).toBeInTheDocument();
        // Should be wrapped in a Link to /jobs/new
        const link = startTrainingLink.closest('a');
        expect(link).toHaveAttribute('href', '/jobs/new');
    });

    it('renders Deploy to Ollama button when models load', async () => {
        server.use(
            http.get('http://localhost:8000/v1/models', () => {
                return HttpResponse.json({
                    data: [{
                        id: 'model-1',
                        object: 'model',
                        owned_by: 'local',
                        created: Date.now() / 1000,
                    }]
                });
            })
        );

        renderWithProviders(<ModelsPage />);

        await waitFor(() => {
            expect(screen.getByText('Deploy to Ollama')).toBeInTheDocument();
        });

        // Button should use intent prop correctly (renders as a button)
        const deployButton = screen.getByText('Deploy to Ollama').closest('button');
        expect(deployButton).toBeInTheDocument();
    });

    it('shows error state when API fails', async () => {
        server.use(
            http.get('http://localhost:8000/v1/models', () => {
                return HttpResponse.json({ error: 'Server error' }, { status: 500 });
            })
        );

        renderWithProviders(<ModelsPage />);

        await waitFor(() => {
            expect(screen.getByText('Failed to load models')).toBeInTheDocument();
        });
    });
});

