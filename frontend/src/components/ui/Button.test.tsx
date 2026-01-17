// =============================================================================
// Button Component Tests
// =============================================================================

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Button, PrimaryButton, SecondaryButton, GhostButton, DestructiveButton, IconButton } from './Button';
import { Plus } from 'lucide-react';

describe('Button', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders children correctly', () => {
        render(<Button>Click me</Button>);
        expect(screen.getByText('Click me')).toBeInTheDocument();
    });

    it('renders as a button element', () => {
        render(<Button>Test</Button>);
        expect(screen.getByRole('button')).toBeInTheDocument();
    });

    // =========================================================================
    // Click Handling
    // =========================================================================

    it('calls onClick when clicked', () => {
        const handleClick = vi.fn();
        render(<Button onClick={handleClick}>Click</Button>);
        fireEvent.click(screen.getByText('Click'));
        expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('does not call onClick when disabled', () => {
        const handleClick = vi.fn();
        render(<Button onClick={handleClick} disabled>Click</Button>);
        fireEvent.click(screen.getByText('Click'));
        expect(handleClick).not.toHaveBeenCalled();
    });

    // =========================================================================
    // Loading State
    // =========================================================================

    it('is disabled when loading', () => {
        render(<Button loading>Click</Button>);
        expect(screen.getByRole('button')).toBeDisabled();
    });

    it('shows spinner when loading', () => {
        const { container } = render(<Button loading>Click</Button>);
        expect(container.querySelector('.spinner')).toBeInTheDocument();
    });

    it('does not call onClick when loading', () => {
        const handleClick = vi.fn();
        render(<Button onClick={handleClick} loading>Click</Button>);
        fireEvent.click(screen.getByRole('button'));
        expect(handleClick).not.toHaveBeenCalled();
    });

    // =========================================================================
    // Intent Styles
    // =========================================================================

    it('applies primary intent class by default', () => {
        const { container } = render(<Button>Primary</Button>);
        expect(container.firstChild).toHaveClass('btn-primary');
    });

    it('applies secondary intent class', () => {
        const { container } = render(<Button intent="secondary">Secondary</Button>);
        expect(container.firstChild).toHaveClass('btn-secondary');
    });

    it('applies ghost intent class', () => {
        const { container } = render(<Button intent="ghost">Ghost</Button>);
        expect(container.firstChild).toHaveClass('btn-ghost');
    });

    it('applies destructive intent class', () => {
        const { container } = render(<Button intent="destructive">Destructive</Button>);
        expect(container.firstChild).toHaveClass('btn-destructive');
    });

    // =========================================================================
    // Size Styles
    // =========================================================================

    it('applies md size class by default', () => {
        const { container } = render(<Button>Medium</Button>);
        expect(container.firstChild).toHaveClass('btn-md');
    });

    it('applies sm size class', () => {
        const { container } = render(<Button size="sm">Small</Button>);
        expect(container.firstChild).toHaveClass('btn-sm');
    });

    it('applies lg size class', () => {
        const { container } = render(<Button size="lg">Large</Button>);
        expect(container.firstChild).toHaveClass('btn-lg');
    });

    // =========================================================================
    // Icon Support
    // =========================================================================

    it('renders icon on the left by default', () => {
        const { container } = render(
            <Button icon={<Plus data-testid="icon" />}>With Icon</Button>
        );
        const button = container.firstChild as HTMLElement;
        const icon = screen.getByTestId('icon');
        const text = screen.getByText('With Icon');

        // Icon should come before text in DOM order
        expect(button.firstElementChild).toBe(icon);
        expect(text).toBeInTheDocument();
    });

    it('renders icon on the right when specified', () => {
        const { container } = render(
            <Button icon={<Plus data-testid="icon" />} iconPosition="right">With Icon</Button>
        );
        const button = container.firstChild as HTMLElement;
        const icon = screen.getByTestId('icon');

        // Icon should come after text (last element is the icon)
        expect(button.lastElementChild).toBe(icon);
    });

    it('hides icon when loading', () => {
        render(
            <Button icon={<Plus data-testid="icon" />} loading>With Icon</Button>
        );
        expect(screen.queryByTestId('icon')).not.toBeInTheDocument();
    });

    // =========================================================================
    // Full Width
    // =========================================================================

    it('applies full width class when fullWidth is true', () => {
        const { container } = render(<Button fullWidth>Full Width</Button>);
        expect(container.firstChild).toHaveClass('w-full');
    });

    // =========================================================================
    // Custom Classes
    // =========================================================================

    it('applies custom className', () => {
        const { container } = render(<Button className="custom-class">Custom</Button>);
        expect(container.firstChild).toHaveClass('custom-class');
    });

    // =========================================================================
    // Convenience Components
    // =========================================================================

    it('PrimaryButton renders with primary intent', () => {
        const { container } = render(<PrimaryButton>Primary</PrimaryButton>);
        expect(container.firstChild).toHaveClass('btn-primary');
    });

    it('SecondaryButton renders with secondary intent', () => {
        const { container } = render(<SecondaryButton>Secondary</SecondaryButton>);
        expect(container.firstChild).toHaveClass('btn-secondary');
    });

    it('GhostButton renders with ghost intent', () => {
        const { container } = render(<GhostButton>Ghost</GhostButton>);
        expect(container.firstChild).toHaveClass('btn-ghost');
    });

    it('DestructiveButton renders with destructive intent', () => {
        const { container } = render(<DestructiveButton>Destructive</DestructiveButton>);
        expect(container.firstChild).toHaveClass('btn-destructive');
    });

    it('IconButton renders with aria-label', () => {
        render(
            <IconButton icon={<Plus data-testid="icon" />} aria-label="Add item" />
        );
        expect(screen.getByRole('button')).toHaveAttribute('aria-label', 'Add item');
    });

    it('IconButton applies btn-icon class', () => {
        const { container } = render(
            <IconButton icon={<Plus />} aria-label="Add" />
        );
        expect(container.firstChild).toHaveClass('btn-icon');
    });
});
