import { ReactNode, useEffect } from 'react'
import { X } from 'lucide-react'
import clsx from 'clsx'

interface DialogProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    children: ReactNode;
    footer?: ReactNode;
    size?: 'md' | 'lg' | 'xl';
}

export function Dialog({
    isOpen,
    onClose,
    title,
    children,
    footer,
    size = 'md'
}: DialogProps) {
    // Handle escape key
    useEffect(() => {
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose();
        };

        if (isOpen) {
            document.addEventListener('keydown', handleEscape);
            document.body.style.overflow = 'hidden';
        }

        return () => {
            document.removeEventListener('keydown', handleEscape);
            document.body.style.overflow = '';
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div className="dialog-overlay" onClick={onClose}>
            <div
                className={clsx('dialog', size === 'lg' && 'dialog-lg', size === 'xl' && 'dialog-xl')}
                onClick={e => e.stopPropagation()}
            >
                <div className="dialog-header">
                    <h2 className="dialog-title">{title}</h2>
                    <button
                        className="btn btn-ghost btn-icon"
                        onClick={onClose}
                        aria-label="Close"
                    >
                        <X size={18} />
                    </button>
                </div>

                <div className="dialog-body">
                    {children}
                </div>

                {footer && (
                    <div className="dialog-footer">
                        {footer}
                    </div>
                )}
            </div>
        </div>
    )
}
