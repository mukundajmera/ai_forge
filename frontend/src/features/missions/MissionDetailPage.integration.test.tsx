// =============================================================================
// MissionDetailPage Integration Tests
// =============================================================================

import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { server } from '@/mocks/server';
import { mockMissions } from '@/mocks/handlers';
import { MissionDetailPage } from './MissionDetailPage';

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

const renderWithProviders = (missionId: string = 'mission-1') => {
    const queryClient = createTestQueryClient();
    return {
        ...render(
            <QueryClientProvider client={queryClient}>
                <MemoryRouter initialEntries={[`/missions/${missionId}`]}>
                    <Routes>
                        <Route path="/missions/:missionId" element={<MissionDetailPage />} />
                        <Route path="/missions" element={<div>Missions List</div>} />
                    </Routes>
                </MemoryRouter>
            </QueryClientProvider>
        ),
        queryClient,
    };
};

// =============================================================================
// Tests
// =============================================================================

describe('MissionDetailPage Integration', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders mission title', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Recommend retrain for myproject')).toBeInTheDocument();
        });
    });

    it('renders back link', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Back to Missions')).toBeInTheDocument();
        });
    });

    it('renders mission description', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText(/New commits detected/i)).toBeInTheDocument();
        });
    });

    // =========================================================================
    // Agent Analysis Section
    // =========================================================================

    it('renders agent analysis section', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Agent Analysis')).toBeInTheDocument();
        });
    });

    it('displays trigger information', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Trigger')).toBeInTheDocument();
            expect(screen.getByText(/New commits detected/i)).toBeInTheDocument();
        });
    });

    it('displays analysis information', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Analysis')).toBeInTheDocument();
        });
    });

    it('displays expected outcome', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Expected Outcome')).toBeInTheDocument();
        });
    });

    // =========================================================================
    // Status and Details
    // =========================================================================

    it('displays mission status', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Status')).toBeInTheDocument();
            expect(screen.getByText(/pending/i)).toBeInTheDocument();
        });
    });

    it('displays mission type', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Type')).toBeInTheDocument();
        });
    });

    it('displays confidence score', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByText('Confidence')).toBeInTheDocument();
            expect(screen.getByText('85%')).toBeInTheDocument();
        });
    });

    // =========================================================================
    // Approve/Reject Actions
    // =========================================================================

    it('renders approve button for pending mission', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Approve/i })).toBeInTheDocument();
        });
    });

    it('renders reject button for pending mission', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Reject/i })).toBeInTheDocument();
        });
    });

    it('opens reject dialog when reject button is clicked', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Reject/i })).toBeInTheDocument();
        });

        fireEvent.click(screen.getByRole('button', { name: /Reject/i }));

        await waitFor(() => {
            expect(screen.getByText('Reject Mission')).toBeInTheDocument();
            expect(screen.getByPlaceholderText(/Not enough new commits/i)).toBeInTheDocument();
        });
    });

    it('reject dialog requires reason before submitting', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Reject/i })).toBeInTheDocument();
        });

        fireEvent.click(screen.getByRole('button', { name: /^Reject$/i }));

        await waitFor(() => {
            const rejectMissionBtn = screen.getByRole('button', { name: 'Reject Mission' });
            expect(rejectMissionBtn).toBeDisabled();
        });
    });

    it('enables reject button when reason is provided', async () => {
        renderWithProviders();

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Reject/i })).toBeInTheDocument();
        });

        fireEvent.click(screen.getByRole('button', { name: /^Reject$/i }));

        await waitFor(() => {
            expect(screen.getByPlaceholderText(/Not enough new commits/i)).toBeInTheDocument();
        });

        const textarea = screen.getByPlaceholderText(/Not enough new commits/i);
        fireEvent.change(textarea, { target: { value: 'Test rejection reason' } });

        await waitFor(() => {
            const rejectMissionBtn = screen.getByRole('button', { name: 'Reject Mission' });
            expect(rejectMissionBtn).not.toBeDisabled();
        });
    });

    // =========================================================================
    // Error States
    // =========================================================================

    it('shows error state for non-existent mission', async () => {
        server.use(
            http.get('http://localhost:8000/missions/non-existent', () => {
                return HttpResponse.json({ message: 'Not found' }, { status: 404 });
            })
        );

        renderWithProviders('non-existent');

        await waitFor(() => {
            expect(screen.getByText('Mission Not Found')).toBeInTheDocument();
        });
    });

    it('shows back to missions button on error', async () => {
        server.use(
            http.get('http://localhost:8000/missions/non-existent', () => {
                return HttpResponse.json({ message: 'Not found' }, { status: 404 });
            })
        );

        renderWithProviders('non-existent');

        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'Back to Missions' })).toBeInTheDocument();
        });
    });

    // =========================================================================
    // Loading State
    // =========================================================================

    it('shows loading skeleton initially', () => {
        renderWithProviders();

        // Should show skeleton elements during loading
        expect(document.querySelector('.skeleton-title, .skeleton')).toBeInTheDocument();
    });
});
