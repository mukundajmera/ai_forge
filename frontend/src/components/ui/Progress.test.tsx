// =============================================================================
// Progress Component Tests
// =============================================================================

import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { Progress, QualityBar } from './Progress';

describe('Progress', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders progress bar', () => {
        const { container } = render(<Progress value={50} />);
        expect(container.querySelector('.progress')).toBeInTheDocument();
    });

    it('renders progress bar element', () => {
        const { container } = render(<Progress value={50} />);
        expect(container.querySelector('.progress-bar')).toBeInTheDocument();
    });

    // =========================================================================
    // Value Display
    // =========================================================================

    it('sets correct width based on value', () => {
        const { container } = render(<Progress value={75} />);
        const bar = container.querySelector('.progress-bar');
        expect(bar).toHaveStyle({ width: '75%' });
    });

    it('clamps value to 0 for negative values', () => {
        const { container } = render(<Progress value={-10} />);
        const bar = container.querySelector('.progress-bar');
        expect(bar).toHaveStyle({ width: '0%' });
    });

    it('clamps value to 100 for values over 100', () => {
        const { container } = render(<Progress value={150} />);
        const bar = container.querySelector('.progress-bar');
        expect(bar).toHaveStyle({ width: '100%' });
    });

    // =========================================================================
    // Label
    // =========================================================================

    it('does not show label by default', () => {
        render(<Progress value={50} />);
        expect(screen.queryByText('50%')).not.toBeInTheDocument();
    });

    it('shows label when showLabel is true', () => {
        render(<Progress value={50} showLabel />);
        expect(screen.getByText('50%')).toBeInTheDocument();
    });

    it('rounds label value', () => {
        render(<Progress value={33.7} showLabel />);
        expect(screen.getByText('34%')).toBeInTheDocument();
    });

    // =========================================================================
    // Size Variants
    // =========================================================================

    it('applies default md size (no additional class)', () => {
        const { container } = render(<Progress value={50} />);
        const progress = container.querySelector('.progress');
        expect(progress).not.toHaveClass('progress-sm');
    });

    it('applies sm size class', () => {
        const { container } = render(<Progress value={50} size="sm" />);
        const progress = container.querySelector('.progress');
        expect(progress).toHaveClass('progress-sm');
    });

    // =========================================================================
    // Variant Styles
    // =========================================================================

    it('applies default variant (no success class)', () => {
        const { container } = render(<Progress value={50} />);
        const progress = container.querySelector('.progress');
        expect(progress).not.toHaveClass('progress-success');
    });

    it('applies success variant class', () => {
        const { container } = render(<Progress value={50} variant="success" />);
        const progress = container.querySelector('.progress');
        expect(progress).toHaveClass('progress-success');
    });

    // =========================================================================
    // Custom ClassName
    // =========================================================================

    it('applies custom className', () => {
        const { container } = render(<Progress value={50} className="custom-progress" />);
        const wrapper = container.querySelector('.progress-wrapper');
        expect(wrapper).toHaveClass('custom-progress');
    });
});

describe('QualityBar', () => {
    // =========================================================================
    // Basic Rendering
    // =========================================================================

    it('renders quality bar', () => {
        const { container } = render(<QualityBar score={0.8} />);
        expect(container.querySelector('.quality-bar')).toBeInTheDocument();
    });

    it('renders correct number of segments', () => {
        const { container } = render(<QualityBar score={0.5} />);
        const segments = container.querySelectorAll('.quality-bar-segment');
        expect(segments).toHaveLength(5); // default segments
    });

    it('renders custom number of segments', () => {
        const { container } = render(<QualityBar score={0.5} segments={10} />);
        const segments = container.querySelectorAll('.quality-bar-segment');
        expect(segments).toHaveLength(10);
    });

    // =========================================================================
    // Filled Segments
    // =========================================================================

    it('fills correct segments based on score', () => {
        const { container } = render(<QualityBar score={0.6} segments={5} />);
        const filledSegments = container.querySelectorAll('.quality-bar-segment.filled');
        expect(filledSegments).toHaveLength(3); // 0.6 * 5 = 3
    });

    it('fills all segments for score of 1.0', () => {
        const { container } = render(<QualityBar score={1.0} segments={5} />);
        const filledSegments = container.querySelectorAll('.quality-bar-segment.filled');
        expect(filledSegments).toHaveLength(5);
    });

    it('fills no segments for score of 0', () => {
        const { container } = render(<QualityBar score={0} segments={5} />);
        const filledSegments = container.querySelectorAll('.quality-bar-segment.filled');
        expect(filledSegments).toHaveLength(0);
    });

    // =========================================================================
    // Quality Levels
    // =========================================================================

    it('applies high level class for score >= 0.7', () => {
        const { container } = render(<QualityBar score={0.8} />);
        expect(container.querySelector('.high')).toBeInTheDocument();
    });

    it('applies medium level class for score >= 0.5 and < 0.7', () => {
        const { container } = render(<QualityBar score={0.6} />);
        expect(container.querySelector('.medium')).toBeInTheDocument();
    });

    it('applies low level class for score < 0.5', () => {
        const { container } = render(<QualityBar score={0.3} />);
        expect(container.querySelector('.low')).toBeInTheDocument();
    });

    // =========================================================================
    // Value Display
    // =========================================================================

    it('shows value by default', () => {
        render(<QualityBar score={0.75} />);
        expect(screen.getByText('0.75')).toBeInTheDocument();
    });

    it('hides value when showValue is false', () => {
        render(<QualityBar score={0.75} showValue={false} />);
        expect(screen.queryByText('0.75')).not.toBeInTheDocument();
    });

    it('formats value to 2 decimal places', () => {
        render(<QualityBar score={0.333333} />);
        expect(screen.getByText('0.33')).toBeInTheDocument();
    });
});
