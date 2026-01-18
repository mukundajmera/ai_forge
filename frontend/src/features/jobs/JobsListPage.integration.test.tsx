// =============================================================================
// JobsListPage Integration Tests
// =============================================================================

import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { JobsListPage } from './JobsListPage';

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

const renderWithProviders = (ui: React.ReactElement, { route = '/' } = {}) => {
    const queryClient = createTestQueryClient();
    return {
        ...render(
            <QueryClientProvider client={queryClient}>
                <MemoryRouter initialEntries={[route]}>
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

describe('JobsListPage Integration', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders page header correctly', async () => {
        renderWithProviders(<JobsListPage />);

        expect(screen.getByText('Training Jobs')).toBeInTheDocument();
        expect(screen.getByText(/All fine-tuning runs/i)).toBeInTheDocument();
    });

    it('renders new fine-tune button', async () => {
        renderWithProviders(<JobsListPage />);

        expect(screen.getByRole('link', { name: /New Fine-Tune/i })).toBeInTheDocument();
    });

    // =========================================================================
    // Jobs List Display
    // =========================================================================

    it('renders jobs list with data', async () => {
        renderWithProviders(<JobsListPage />);

        // Page uses mock data by default, check for expected job project names
        await waitFor(() => {
            expect(screen.getByText('my-project')).toBeInTheDocument();
        });
    });

    it.skip('displays job status badges', async () => {
        renderWithProviders(<JobsListPage />);

        await waitFor(() => {
            expect(screen.getByText(/Training/i)).toBeInTheDocument();
        });
    });

    it.skip('displays job base model', async () => {
        renderWithProviders(<JobsListPage />);

        await waitFor(() => {
            expect(screen.getByText('Llama-3.2-3B')).toBeInTheDocument();
        });
    });

    // =========================================================================
    // Filter Functionality
    // =========================================================================

    it.skip('renders filter tabs', async () => {
        renderWithProviders(<JobsListPage />);

        expect(screen.getByText('All')).toBeInTheDocument();
        expect(screen.getByText('Running')).toBeInTheDocument();
        expect(screen.getByText('Completed')).toBeInTheDocument();
        expect(screen.getByText('Failed')).toBeInTheDocument();
    });

    it.skip('filters jobs by status when tab is clicked', async () => {
        renderWithProviders(<JobsListPage />);

        // Click on Completed filter
        const completedTab = screen.getByText('Completed');
        fireEvent.click(completedTab);

        // Should show completed jobs only
        await waitFor(() => {
            expect(completedTab).toHaveClass('active');
        });
    });

    // =========================================================================
    // Search Functionality
    // =========================================================================

    it('renders search input', async () => {
        renderWithProviders(<JobsListPage />);

        expect(screen.getByPlaceholderText(/Search projects/i)).toBeInTheDocument();
    });

    it('filters jobs by search query', async () => {
        renderWithProviders(<JobsListPage />);

        const searchInput = screen.getByPlaceholderText(/Search projects/i);
        fireEvent.change(searchInput, { target: { value: 'api' } });

        await waitFor(() => {
            expect(screen.getByText('api-helper')).toBeInTheDocument();
        });
    });

    // =========================================================================
    // Empty State
    // =========================================================================

    it('shows empty state when search returns no results', async () => {
        renderWithProviders(<JobsListPage />);

        const searchInput = screen.getByPlaceholderText(/Search projects/i);
        fireEvent.change(searchInput, { target: { value: 'xyz-nonexistent-project' } });

        await waitFor(() => {
            expect(screen.getByText(/No jobs match your search/i)).toBeInTheDocument();
        });
    });

    // =========================================================================
    // Progress Display
    // =========================================================================

    it('shows progress bar for training jobs', async () => {
        renderWithProviders(<JobsListPage />);

        // Training jobs should show a progress bar
        await waitFor(() => {
            expect(screen.getByText('78%')).toBeInTheDocument();
        });
    });

    // =========================================================================
    // Job Details Link
    // =========================================================================

    it('job row links to job detail page', async () => {
        renderWithProviders(<JobsListPage />);

        await waitFor(() => {
            const jobLink = screen.getAllByRole('link').find(
                link => link.getAttribute('href')?.includes('/jobs/')
            );
            expect(jobLink).toBeInTheDocument();
        });
    });
});
