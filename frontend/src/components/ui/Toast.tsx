import { useState, useEffect, createContext, useContext, useCallback } from 'react'
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react'

type ToastType = 'success' | 'error' | 'warning' | 'info'

interface Toast {
    id: string
    type: ToastType
    message: string
    description?: string
    duration?: number
}

interface ToastContextType {
    toasts: Toast[]
    addToast: (toast: Omit<Toast, 'id'>) => void
    removeToast: (id: string) => void
}

const ToastContext = createContext<ToastContextType | null>(null)

export function useToast() {
    const context = useContext(ToastContext)
    if (!context) {
        throw new Error('useToast must be used within ToastProvider')
    }
    return context
}

// Convenience functions
export function useToastHelpers() {
    const { addToast } = useToast()

    return {
        success: (message: string, description?: string) =>
            addToast({ type: 'success', message, description }),
        error: (message: string, description?: string) =>
            addToast({ type: 'error', message, description, duration: 8000 }),
        warning: (message: string, description?: string) =>
            addToast({ type: 'warning', message, description }),
        info: (message: string, description?: string) =>
            addToast({ type: 'info', message, description }),
    }
}

interface ToastProviderProps {
    children: React.ReactNode
}

export function ToastProvider({ children }: ToastProviderProps) {
    const [toasts, setToasts] = useState<Toast[]>([])

    const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
        const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`
        setToasts(prev => [...prev, { ...toast, id }])
    }, [])

    const removeToast = useCallback((id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id))
    }, [])

    return (
        <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
            {children}
            <ToastContainer toasts={toasts} onRemove={removeToast} />
        </ToastContext.Provider>
    )
}

interface ToastContainerProps {
    toasts: Toast[]
    onRemove: (id: string) => void
}

function ToastContainer({ toasts, onRemove }: ToastContainerProps) {
    return (
        <div className="toast-container" role="region" aria-label="Notifications">
            {toasts.map(toast => (
                <ToastItem key={toast.id} toast={toast} onRemove={onRemove} />
            ))}
        </div>
    )
}

interface ToastItemProps {
    toast: Toast
    onRemove: (id: string) => void
}

function ToastItem({ toast, onRemove }: ToastItemProps) {
    const { id, type, message, description, duration = 5000 } = toast

    useEffect(() => {
        const timer = setTimeout(() => onRemove(id), duration)
        return () => clearTimeout(timer)
    }, [id, duration, onRemove])

    const icons = {
        success: CheckCircle,
        error: XCircle,
        warning: AlertTriangle,
        info: Info,
    }

    const Icon = icons[type]

    return (
        <div
            className={`toast toast-${type}`}
            role="alert"
            aria-live="polite"
        >
            <div className="toast-icon">
                <Icon size={18} />
            </div>
            <div className="toast-content">
                <div className="toast-message">{message}</div>
                {description && (
                    <div className="toast-description">{description}</div>
                )}
            </div>
            <button
                className="toast-close"
                onClick={() => onRemove(id)}
                aria-label="Dismiss notification"
            >
                <X size={16} />
            </button>

            <style>{`
        .toast-icon {
          flex-shrink: 0;
        }
        
        .toast-success .toast-icon { color: var(--status-success); }
        .toast-error .toast-icon { color: var(--status-danger); }
        .toast-warning .toast-icon { color: var(--status-warning); }
        .toast-info .toast-icon { color: var(--status-info); }
        
        .toast-content {
          flex: 1;
          min-width: 0;
        }
        
        .toast-message {
          font-weight: var(--font-medium);
          font-size: var(--text-sm);
        }
        
        .toast-description {
          font-size: var(--text-sm);
          color: var(--text-secondary);
          margin-top: var(--space-1);
        }
        
        .toast-close {
          flex-shrink: 0;
          padding: var(--space-1);
          background: none;
          border: none;
          color: var(--text-tertiary);
          cursor: pointer;
          border-radius: var(--radius-sm);
          transition: background-color var(--transition-fast);
        }
        
        .toast-close:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
        }
      `}</style>
        </div>
    )
}

// Standalone toast function for use outside of React
let globalAddToast: ((toast: Omit<Toast, 'id'>) => void) | null = null

export function setGlobalToast(fn: typeof globalAddToast) {
    globalAddToast = fn
}

export const toast = {
    success: (message: string, description?: string) =>
        globalAddToast?.({ type: 'success', message, description }),
    error: (message: string, description?: string) =>
        globalAddToast?.({ type: 'error', message, description, duration: 8000 }),
    warning: (message: string, description?: string) =>
        globalAddToast?.({ type: 'warning', message, description }),
    info: (message: string, description?: string) =>
        globalAddToast?.({ type: 'info', message, description }),
}
