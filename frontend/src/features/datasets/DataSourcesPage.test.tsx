// =============================================================================
// DataSourcesPage Tests - Navigation and rendering
// =============================================================================

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DataSourcesPage } from './DataSourcesPage';

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

// Track navigation
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return {
        ...actual,
        useNavigate: () => mockNavigate,
    };
});

const renderWithProviders = (ui: React.ReactElement) => {
    const queryClient = createTestQueryClient();
    return {
        ...render(
            <QueryClientProvider client={queryClient}>
                <MemoryRouter initialEntries={['/datasets']}>
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

describe('DataSourcesPage', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders page header correctly', () => {
        renderWithProviders(<DataSourcesPage />);

        expect(screen.getByText('Data Sources')).toBeInTheDocument();
        expect(screen.getByText(/Manage your training data sources/i)).toBeInTheDocument();
    });

    it('renders View Datasets button', () => {
        renderWithProviders(<DataSourcesPage />);

        expect(screen.getByText('View Datasets')).toBeInTheDocument();
    });

    it('navigates via React Router instead of window.location for View Datasets', async () => {
        renderWithProviders(<DataSourcesPage />);

        const viewDatasetsBtn = screen.getByText('View Datasets');
        fireEvent.click(viewDatasetsBtn);

        expect(mockNavigate).toHaveBeenCalledWith('/datasets/generated');
    });

    it('renders Add Data Source button', () => {
        renderWithProviders(<DataSourcesPage />);

        expect(screen.getByText('Add Data Source')).toBeInTheDocument();
    });
});
