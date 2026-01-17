// =============================================================================
// Card Component Tests
// =============================================================================

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Card, StatCard, MetricCard } from './Card';

describe('Card', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders children correctly', () => {
        render(<Card>Card Content</Card>);
        expect(screen.getByText('Card Content')).toBeInTheDocument();
    });

    it('renders as a div element', () => {
        render(<Card>Test</Card>);
        const card = screen.getByText('Test').closest('div');
        expect(card).toBeInTheDocument();
    });

    it('applies card class', () => {
        render(<Card>Test</Card>);
        const card = screen.getByText('Test').closest('.card');
        expect(card).toBeInTheDocument();
    });

    // =========================================================================
    // Elevated Style
    // =========================================================================

    it('applies elevated class when elevated prop is true', () => {
        render(<Card elevated>Elevated</Card>);
        const card = screen.getByText('Elevated').closest('.card');
        expect(card).toHaveClass('card-elevated');
    });

    it('does not apply elevated class by default', () => {
        render(<Card>Normal</Card>);
        const card = screen.getByText('Normal').closest('.card');
        expect(card).not.toHaveClass('card-elevated');
    });

    // =========================================================================
    // Padding Variants
    // =========================================================================

    it('applies md padding class by default', () => {
        render(<Card>Medium Padding</Card>);
        const card = screen.getByText('Medium Padding').closest('.card');
        expect(card).toHaveClass('p-5');
    });

    it('applies no padding class for none', () => {
        render(<Card padding="none">No Padding</Card>);
        const card = screen.getByText('No Padding').closest('.card');
        expect(card).not.toHaveClass('p-3', 'p-5', 'p-6');
    });

    it('applies sm padding class', () => {
        render(<Card padding="sm">Small Padding</Card>);
        const card = screen.getByText('Small Padding').closest('.card');
        expect(card).toHaveClass('p-3');
    });

    it('applies lg padding class', () => {
        render(<Card padding="lg">Large Padding</Card>);
        const card = screen.getByText('Large Padding').closest('.card');
        expect(card).toHaveClass('p-6');
    });

    // =========================================================================
    // Interactive Mode
    // =========================================================================

    it('handles onClick when provided', () => {
        const handleClick = vi.fn();
        render(<Card onClick={handleClick}>Clickable</Card>);
        fireEvent.click(screen.getByText('Clickable'));
        expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('applies button role when onClick is provided', () => {
        const handleClick = vi.fn();
        render(<Card onClick={handleClick}>Clickable</Card>);
        expect(screen.getByRole('button')).toBeInTheDocument();
    });

    it('applies cursor-pointer class when onClick is provided', () => {
        const handleClick = vi.fn();
        render(<Card onClick={handleClick}>Clickable</Card>);
        const card = screen.getByText('Clickable').closest('.card');
        expect(card).toHaveClass('cursor-pointer');
    });

    // =========================================================================
    // Custom Classes
    // =========================================================================

    it('applies custom className', () => {
        render(<Card className="custom-class">Custom</Card>);
        const card = screen.getByText('Custom').closest('.card');
        expect(card).toHaveClass('custom-class');
    });
});

describe('StatCard', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders label and value correctly', () => {
        render(<StatCard label="Total Jobs" value={42} />);
        expect(screen.getByText('Total Jobs')).toBeInTheDocument();
        expect(screen.getByText('42')).toBeInTheDocument();
    });

    it('renders string value correctly', () => {
        render(<StatCard label="Status" value="Active" />);
        expect(screen.getByText('Active')).toBeInTheDocument();
    });

    // =========================================================================
    // Icon Support
    // =========================================================================

    it('renders icon when provided', () => {
        render(<StatCard label="Test" value="100" icon={<span data-testid="icon">🔥</span>} />);
        expect(screen.getByTestId('icon')).toBeInTheDocument();
    });

    // =========================================================================
    // Trend Display
    // =========================================================================

    it('renders trend with up direction', () => {
        render(<StatCard label="Revenue" value="$1000" trend={{ direction: 'up', value: '+10%' }} />);
        expect(screen.getByText('+10%')).toBeInTheDocument();
    });

    it('renders trend with down direction', () => {
        render(<StatCard label="Errors" value="5" trend={{ direction: 'down', value: '-20%' }} />);
        expect(screen.getByText('-20%')).toBeInTheDocument();
    });

    it('renders trend with neutral direction', () => {
        render(<StatCard label="Users" value="100" trend={{ direction: 'neutral', value: '0%' }} />);
        expect(screen.getByText('0%')).toBeInTheDocument();
    });

    // =========================================================================
    // Loading State
    // =========================================================================

    it('shows skeleton loader when loading', () => {
        const { container } = render(<StatCard label="Test" value="100" loading />);
        expect(container.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
    });

    it('does not show value when loading', () => {
        render(<StatCard label="Test" value="100" loading />);
        expect(screen.queryByText('100')).not.toBeInTheDocument();
    });
});

describe('MetricCard', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders label and value correctly', () => {
        render(<MetricCard label="Accuracy" value="95.5" />);
        expect(screen.getByText('Accuracy')).toBeInTheDocument();
        expect(screen.getByText('95.5')).toBeInTheDocument();
    });

    it('renders unit when provided', () => {
        render(<MetricCard label="Speed" value="120" unit="ms" />);
        expect(screen.getByText('ms')).toBeInTheDocument();
    });

    it('renders description when provided', () => {
        render(<MetricCard label="Test" value="100" description="This is a test metric" />);
        expect(screen.getByText('This is a test metric')).toBeInTheDocument();
    });

    // =========================================================================
    // Compact Mode
    // =========================================================================

    it('renders compact mode correctly', () => {
        const { container } = render(<MetricCard label="Test" value="100" compact />);
        expect(container.querySelector('.metric-compact')).toBeInTheDocument();
    });

    it('renders compact mode with unit', () => {
        render(<MetricCard label="Speed" value="120" unit="ms" compact />);
        expect(screen.getByText('ms')).toBeInTheDocument();
    });

    // =========================================================================
    // Status Colors
    // =========================================================================

    it('applies default status color', () => {
        render(<MetricCard label="Test" value="100" />);
        const value = screen.getByText('100');
        expect(value).toHaveStyle({ color: 'var(--text-primary)' });
    });

    it('applies good status color', () => {
        render(<MetricCard label="Test" value="100" status="good" />);
        const value = screen.getByText('100');
        expect(value).toHaveStyle({ color: 'var(--status-success)' });
    });

    it('applies warning status color', () => {
        render(<MetricCard label="Test" value="100" status="warning" />);
        const value = screen.getByText('100');
        expect(value).toHaveStyle({ color: 'var(--status-warning)' });
    });

    it('applies critical status color', () => {
        render(<MetricCard label="Test" value="100" status="critical" />);
        const value = screen.getByText('100');
        expect(value).toHaveStyle({ color: 'var(--status-danger)' });
    });
});
