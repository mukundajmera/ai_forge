// =============================================================================
// Modal Component Tests
// =============================================================================

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Modal } from './Modal';

describe('Modal', () => {
    // =========================================================================
    // Open/Close State
    // =========================================================================

    it('renders when isOpen is true', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="Test Modal">
                Modal Content
            </Modal>
        );
        expect(screen.getByText('Test Modal')).toBeInTheDocument();
        expect(screen.getByText('Modal Content')).toBeInTheDocument();
    });

    it('does not render when isOpen is false', () => {
        render(
            <Modal isOpen={false} onClose={() => { }} title="Test Modal">
                Modal Content
            </Modal>
        );
        expect(screen.queryByText('Test Modal')).not.toBeInTheDocument();
        expect(screen.queryByText('Modal Content')).not.toBeInTheDocument();
    });

    // =========================================================================
    // Close Button
    // =========================================================================

    it('calls onClose when close button is clicked', () => {
        const handleClose = vi.fn();
        render(
            <Modal isOpen={true} onClose={handleClose} title="Test Modal">
                Content
            </Modal>
        );
        const closeButton = screen.getByRole('button', { name: 'Close modal' });
        fireEvent.click(closeButton);
        expect(handleClose).toHaveBeenCalledTimes(1);
    });

    // =========================================================================
    // Overlay Click
    // =========================================================================

    it('calls onClose when overlay is clicked', () => {
        const handleClose = vi.fn();
        render(
            <Modal isOpen={true} onClose={handleClose} title="Test Modal">
                Content
            </Modal>
        );
        const overlay = document.querySelector('.modal-overlay');
        fireEvent.click(overlay!);
        expect(handleClose).toHaveBeenCalledTimes(1);
    });

    it('does not call onClose when modal content is clicked', () => {
        const handleClose = vi.fn();
        render(
            <Modal isOpen={true} onClose={handleClose} title="Test Modal">
                <button>Inner Button</button>
            </Modal>
        );
        fireEvent.click(screen.getByText('Inner Button'));
        expect(handleClose).not.toHaveBeenCalled();
    });

    // =========================================================================
    // Escape Key
    // =========================================================================

    it('calls onClose when Escape key is pressed', () => {
        const handleClose = vi.fn();
        render(
            <Modal isOpen={true} onClose={handleClose} title="Test Modal">
                Content
            </Modal>
        );
        fireEvent.keyDown(document, { key: 'Escape' });
        expect(handleClose).toHaveBeenCalledTimes(1);
    });

    it('does not call onClose on Escape when modal is closed', () => {
        const handleClose = vi.fn();
        render(
            <Modal isOpen={false} onClose={handleClose} title="Test Modal">
                Content
            </Modal>
        );
        fireEvent.keyDown(document, { key: 'Escape' });
        expect(handleClose).not.toHaveBeenCalled();
    });

    // =========================================================================
    // Title
    // =========================================================================

    it('renders title correctly', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="My Modal Title">
                Content
            </Modal>
        );
        expect(screen.getByRole('heading', { name: 'My Modal Title' })).toBeInTheDocument();
    });

    // =========================================================================
    // Size Variants
    // =========================================================================

    it('applies default md size class', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="Test">
                Content
            </Modal>
        );
        expect(document.querySelector('.modal-md')).toBeInTheDocument();
    });

    it('applies sm size class', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="Test" size="sm">
                Content
            </Modal>
        );
        expect(document.querySelector('.modal-sm')).toBeInTheDocument();
    });

    it('applies lg size class', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="Test" size="lg">
                Content
            </Modal>
        );
        expect(document.querySelector('.modal-lg')).toBeInTheDocument();
    });

    it('applies xl size class', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="Test" size="xl">
                Content
            </Modal>
        );
        expect(document.querySelector('.modal-xl')).toBeInTheDocument();
    });

    // =========================================================================
    // Footer
    // =========================================================================

    it('renders footer when provided', () => {
        render(
            <Modal
                isOpen={true}
                onClose={() => { }}
                title="Test"
                footer={<button>Submit</button>}
            >
                Content
            </Modal>
        );
        expect(screen.getByText('Submit')).toBeInTheDocument();
    });

    it('does not render footer when not provided', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="Test">
                Content
            </Modal>
        );
        expect(document.querySelector('.modal-footer')).not.toBeInTheDocument();
    });

    // =========================================================================
    // Body Scroll Prevention
    // =========================================================================

    it('prevents body scroll when open', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="Test">
                Content
            </Modal>
        );
        expect(document.body.style.overflow).toBe('hidden');
    });

    it('restores body scroll when closed', () => {
        const { rerender } = render(
            <Modal isOpen={true} onClose={() => { }} title="Test">
                Content
            </Modal>
        );
        expect(document.body.style.overflow).toBe('hidden');

        rerender(
            <Modal isOpen={false} onClose={() => { }} title="Test">
                Content
            </Modal>
        );
        expect(document.body.style.overflow).toBe('');
    });

    // =========================================================================
    // Children Rendering
    // =========================================================================

    it('renders children content correctly', () => {
        render(
            <Modal isOpen={true} onClose={() => { }} title="Test">
                <div>
                    <p>Paragraph 1</p>
                    <p>Paragraph 2</p>
                </div>
            </Modal>
        );
        expect(screen.getByText('Paragraph 1')).toBeInTheDocument();
        expect(screen.getByText('Paragraph 2')).toBeInTheDocument();
    });
});
