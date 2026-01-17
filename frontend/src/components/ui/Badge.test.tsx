// =============================================================================
// Badge Component Tests
// =============================================================================

import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { Badge, StatusBadge, ModelStatusBadge, QualityBadge, CountBadge } from './Badge';

describe('Badge', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders children correctly', () => {
        render(<Badge>Test Badge</Badge>);
        expect(screen.getByText('Test Badge')).toBeInTheDocument();
    });

    it('renders as a span element', () => {
        render(<Badge>Test</Badge>);
        const badge = screen.getByText('Test');
        expect(badge.tagName).toBe('SPAN');
    });

    // =========================================================================
    // Variant Styles
    // =========================================================================

    it('applies muted variant class by default', () => {
        render(<Badge>Muted</Badge>);
        expect(screen.getByText('Muted')).toHaveClass('badge-muted');
    });

    it('applies success variant class', () => {
        render(<Badge variant="success">Success</Badge>);
        expect(screen.getByText('Success')).toHaveClass('badge-success');
    });

    it('applies warning variant class', () => {
        render(<Badge variant="warning">Warning</Badge>);
        expect(screen.getByText('Warning')).toHaveClass('badge-warning');
    });

    it('applies danger variant class', () => {
        render(<Badge variant="danger">Danger</Badge>);
        expect(screen.getByText('Danger')).toHaveClass('badge-danger');
    });

    it('applies info variant class', () => {
        render(<Badge variant="info">Info</Badge>);
        expect(screen.getByText('Info')).toHaveClass('badge-info');
    });

    it('applies primary variant class', () => {
        render(<Badge variant="primary">Primary</Badge>);
        expect(screen.getByText('Primary')).toHaveClass('badge-primary');
    });

    // =========================================================================
    // Dot Indicator
    // =========================================================================

    it('shows dot indicator when dot prop is true', () => {
        render(<Badge dot>With Dot</Badge>);
        const badge = screen.getByText('With Dot');
        expect(badge.querySelector('.badge-dot')).toBeInTheDocument();
    });

    it('does not show dot indicator by default', () => {
        render(<Badge>No Dot</Badge>);
        const badge = screen.getByText('No Dot');
        expect(badge.querySelector('.badge-dot')).not.toBeInTheDocument();
    });

    // =========================================================================
    // Size Variations
    // =========================================================================

    it('applies md size classes by default', () => {
        render(<Badge>Medium</Badge>);
        expect(screen.getByText('Medium')).toHaveClass('px-2', 'py-1');
    });

    it('applies sm size classes', () => {
        render(<Badge size="sm">Small</Badge>);
        expect(screen.getByText('Small')).toHaveClass('px-1.5', 'py-0.5');
    });
});

describe('StatusBadge', () => {
    // =========================================================================
    // Status Rendering
    // =========================================================================

    it('renders queued status correctly', () => {
        render(<StatusBadge status="queued" />);
        expect(screen.getByText('Queued')).toBeInTheDocument();
    });

    it('renders running status correctly', () => {
        render(<StatusBadge status="running" />);
        expect(screen.getByText('Running')).toBeInTheDocument();
    });

    it('renders completed status correctly', () => {
        render(<StatusBadge status="completed" />);
        expect(screen.getByText('Completed')).toBeInTheDocument();
    });

    it('renders failed status correctly', () => {
        render(<StatusBadge status="failed" />);
        expect(screen.getByText('Failed')).toBeInTheDocument();
    });

    it('renders cancelled status correctly', () => {
        render(<StatusBadge status="cancelled" />);
        expect(screen.getByText('Cancelled')).toBeInTheDocument();
    });

    // =========================================================================
    // Icon and Label Control
    // =========================================================================

    it('shows icon by default', () => {
        const { container } = render(<StatusBadge status="completed" />);
        expect(container.querySelector('svg')).toBeInTheDocument();
    });

    it('hides icon when showIcon is false', () => {
        const { container } = render(<StatusBadge status="completed" showIcon={false} />);
        expect(container.querySelector('svg')).not.toBeInTheDocument();
    });

    it('shows label by default', () => {
        render(<StatusBadge status="completed" />);
        expect(screen.getByText('Completed')).toBeInTheDocument();
    });

    it('hides label when showLabel is false', () => {
        render(<StatusBadge status="completed" showLabel={false} />);
        expect(screen.queryByText('Completed')).not.toBeInTheDocument();
    });

    // =========================================================================
    // Running Animation
    // =========================================================================

    it('applies animate-spin class for running status', () => {
        const { container } = render(<StatusBadge status="running" />);
        expect(container.querySelector('.animate-spin')).toBeInTheDocument();
    });

    it('does not apply animate-spin for completed status', () => {
        const { container } = render(<StatusBadge status="completed" />);
        expect(container.querySelector('.animate-spin')).not.toBeInTheDocument();
    });
});

describe('ModelStatusBadge', () => {
    it('renders active status correctly', () => {
        render(<ModelStatusBadge status="active" />);
        expect(screen.getByText('Active')).toBeInTheDocument();
    });

    it('renders candidate status correctly', () => {
        render(<ModelStatusBadge status="candidate" />);
        expect(screen.getByText('Candidate')).toBeInTheDocument();
    });

    it('renders deprecated status correctly', () => {
        render(<ModelStatusBadge status="deprecated" />);
        expect(screen.getByText('Deprecated')).toBeInTheDocument();
    });

    it('renders exporting status correctly', () => {
        render(<ModelStatusBadge status="exporting" />);
        expect(screen.getByText('Exporting')).toBeInTheDocument();
    });

    it('shows check icon for active status', () => {
        const { container } = render(<ModelStatusBadge status="active" />);
        expect(container.querySelector('svg')).toBeInTheDocument();
    });
});

describe('QualityBadge', () => {
    it('shows success variant for high scores (>= 80%)', () => {
        render(<QualityBadge score={0.85} />);
        expect(screen.getByText('85%')).toHaveClass('badge-success');
    });

    it('shows warning variant for medium scores (>= 60%)', () => {
        render(<QualityBadge score={0.65} />);
        expect(screen.getByText('65%')).toHaveClass('badge-warning');
    });

    it('shows danger variant for low scores (< 60%)', () => {
        render(<QualityBadge score={0.45} />);
        expect(screen.getByText('45%')).toHaveClass('badge-danger');
    });

    it('formats percentage correctly', () => {
        render(<QualityBadge score={0.789} />);
        expect(screen.getByText('79%')).toBeInTheDocument();
    });
});

describe('CountBadge', () => {
    it('renders count correctly', () => {
        render(<CountBadge count={5} />);
        expect(screen.getByText('5')).toBeInTheDocument();
    });

    it('returns null for zero count', () => {
        const { container } = render(<CountBadge count={0} />);
        expect(container.firstChild).toBeNull();
    });

    it('returns null for negative count', () => {
        const { container } = render(<CountBadge count={-1} />);
        expect(container.firstChild).toBeNull();
    });

    it('shows max+ when count exceeds default max', () => {
        render(<CountBadge count={150} />);
        expect(screen.getByText('99+')).toBeInTheDocument();
    });

    it('respects custom max value', () => {
        render(<CountBadge count={50} max={30} />);
        expect(screen.getByText('30+')).toBeInTheDocument();
    });

    it('shows exact count when below max', () => {
        render(<CountBadge count={25} max={30} />);
        expect(screen.getByText('25')).toBeInTheDocument();
    });
});
