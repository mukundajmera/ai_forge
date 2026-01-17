// =============================================================================
// Input Component Tests
// =============================================================================

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Input, Textarea } from './Input';
import { Search, X } from 'lucide-react';

describe('Input', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders input element', () => {
        render(<Input placeholder="Enter text" />);
        expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
    });

    it('renders with value', () => {
        render(<Input value="test value" onChange={() => { }} />);
        expect(screen.getByDisplayValue('test value')).toBeInTheDocument();
    });

    // =========================================================================
    // Label
    // =========================================================================

    it('renders label when provided', () => {
        render(<Input label="Username" />);
        expect(screen.getByText('Username')).toBeInTheDocument();
    });

    it('associates label with input via htmlFor', () => {
        render(<Input label="Username" id="username-input" />);
        const label = screen.getByText('Username');
        expect(label).toHaveAttribute('for', 'username-input');
    });

    // =========================================================================
    // Value Change
    // =========================================================================

    it('calls onChange when value changes', () => {
        const handleChange = vi.fn();
        render(<Input onChange={handleChange} />);
        const input = screen.getByRole('textbox');
        fireEvent.change(input, { target: { value: 'new value' } });
        expect(handleChange).toHaveBeenCalled();
    });

    // =========================================================================
    // Icons
    // =========================================================================

    it('renders left icon when provided', () => {
        render(<Input icon={<Search data-testid="left-icon" />} />);
        expect(screen.getByTestId('left-icon')).toBeInTheDocument();
    });

    it('renders right icon when provided', () => {
        render(<Input rightIcon={<X data-testid="right-icon" />} />);
        expect(screen.getByTestId('right-icon')).toBeInTheDocument();
    });

    it('applies left icon padding class', () => {
        render(<Input icon={<Search />} />);
        const input = screen.getByRole('textbox');
        expect(input).toHaveClass('input-with-left-icon');
    });

    it('applies right icon padding class', () => {
        render(<Input rightIcon={<X />} />);
        const input = screen.getByRole('textbox');
        expect(input).toHaveClass('input-with-right-icon');
    });

    // =========================================================================
    // Error State
    // =========================================================================

    it('displays error message when error prop is provided', () => {
        render(<Input error="This field is required" />);
        expect(screen.getByText('This field is required')).toBeInTheDocument();
    });

    it('applies error class when error prop is provided', () => {
        render(<Input error="Error" />);
        const input = screen.getByRole('textbox');
        expect(input).toHaveClass('input-error');
    });

    it('error message has error color style', () => {
        render(<Input error="Error message" />);
        const errorMessage = screen.getByText('Error message');
        expect(errorMessage).toHaveStyle({ color: 'var(--error-500)' });
    });

    // =========================================================================
    // Hint
    // =========================================================================

    it('displays hint when provided', () => {
        render(<Input hint="This is a helpful hint" />);
        expect(screen.getByText('This is a helpful hint')).toBeInTheDocument();
    });

    it('hides hint when error is present', () => {
        render(<Input hint="Helpful hint" error="Error message" />);
        expect(screen.queryByText('Helpful hint')).not.toBeInTheDocument();
        expect(screen.getByText('Error message')).toBeInTheDocument();
    });

    // =========================================================================
    // Disabled State
    // =========================================================================

    it('is disabled when disabled prop is true', () => {
        render(<Input disabled />);
        expect(screen.getByRole('textbox')).toBeDisabled();
    });

    // =========================================================================
    // Custom Props
    // =========================================================================

    it('applies custom className', () => {
        render(<Input className="custom-input" />);
        const input = screen.getByRole('textbox');
        expect(input).toHaveClass('custom-input');
    });

    it('passes through additional props', () => {
        render(<Input type="email" maxLength={50} />);
        const input = screen.getByRole('textbox');
        expect(input).toHaveAttribute('type', 'email');
        expect(input).toHaveAttribute('maxLength', '50');
    });
});

describe('Textarea', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders textarea element', () => {
        render(<Textarea placeholder="Enter text" />);
        expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
    });

    it('renders textarea as a textarea element', () => {
        render(<Textarea placeholder="test" />);
        const textarea = screen.getByPlaceholderText('test');
        expect(textarea.tagName).toBe('TEXTAREA');
    });

    // =========================================================================
    // Label
    // =========================================================================

    it('renders label when provided', () => {
        render(<Textarea label="Description" />);
        expect(screen.getByText('Description')).toBeInTheDocument();
    });

    // =========================================================================
    // Value Change
    // =========================================================================

    it('calls onChange when value changes', () => {
        const handleChange = vi.fn();
        render(<Textarea onChange={handleChange} />);
        const textarea = screen.getByRole('textbox');
        fireEvent.change(textarea, { target: { value: 'new value' } });
        expect(handleChange).toHaveBeenCalled();
    });

    // =========================================================================
    // Error State
    // =========================================================================

    it('displays error message when error prop is provided', () => {
        render(<Textarea error="Description is required" />);
        expect(screen.getByText('Description is required')).toBeInTheDocument();
    });

    it('applies error class when error prop is provided', () => {
        render(<Textarea error="Error" />);
        const textarea = screen.getByRole('textbox');
        expect(textarea).toHaveClass('input-error');
    });

    // =========================================================================
    // Style
    // =========================================================================

    it('has minimum height style', () => {
        render(<Textarea />);
        const textarea = screen.getByRole('textbox');
        expect(textarea).toHaveStyle({ minHeight: '100px' });
    });

    it('has vertical resize style', () => {
        render(<Textarea />);
        const textarea = screen.getByRole('textbox');
        expect(textarea).toHaveStyle({ resize: 'vertical' });
    });
});
