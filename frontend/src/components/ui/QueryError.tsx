import { AlertCircle, RefreshCw, WifiOff } from 'lucide-react';
import { Button } from './Button';
import { APIError } from '@/lib/api';

interface QueryErrorProps {
    error: Error | null;
    onRetry?: () => void;
    compact?: boolean;
    message?: string;
}

/**
 * Inline error component for failed React Query requests
 * Displays appropriate error message and retry button
 */
export function QueryError({
    error,
    onRetry,
    compact = false,
    message,
}: QueryErrorProps) {
    const isNetworkError = error instanceof APIError && error.isNetworkError;
    const isServerError = error instanceof APIError && error.isServerError;

    const Icon = isNetworkError ? WifiOff : AlertCircle;

    const errorMessage = message
        || (isNetworkError
            ? 'Unable to connect to server'
            : isServerError
                ? 'Server error. Please try again later.'
                : error?.message || 'Something went wrong');

    if (compact) {
        return (
            <div className="query-error-compact" role="alert">
                <Icon size={16} aria-hidden="true" />
                <span>{errorMessage}</span>
                {onRetry && (
                    <button
                        className="retry-link"
                        onClick={onRetry}
                        type="button"
                    >
                        Retry
                    </button>
                )}

                <style>{`
                    .query-error-compact {
                        display: inline-flex;
                        align-items: center;
                        gap: var(--space-2);
                        font-size: var(--text-sm);
                        color: var(--status-danger);
                    }

                    .retry-link {
                        background: none;
                        border: none;
                        color: var(--accent-primary);
                        cursor: pointer;
                        font-size: var(--text-sm);
                        text-decoration: underline;
                        padding: 0;
                    }

                    .retry-link:hover {
                        color: var(--accent-primary-hover);
                    }
                `}</style>
            </div>
        );
    }

    return (
        <div className="query-error" role="alert">
            <div className="error-icon">
                <Icon size={24} aria-hidden="true" />
            </div>
            <div className="error-content">
                <h3 className="error-title">
                    {isNetworkError ? 'Connection Error' : 'Error Loading Data'}
                </h3>
                <p className="error-message">{errorMessage}</p>
                {onRetry && (
                    <Button
                        intent="secondary"
                        size="sm"
                        icon={<RefreshCw size={14} />}
                        onClick={onRetry}
                    >
                        Try Again
                    </Button>
                )}
            </div>

            <style>{`
                .query-error {
                    display: flex;
                    align-items: flex-start;
                    gap: var(--space-4);
                    padding: var(--space-5);
                    background: var(--status-danger-bg);
                    border: 1px solid var(--status-danger-border);
                    border-radius: var(--radius-lg);
                }

                .error-icon {
                    flex-shrink: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 40px;
                    height: 40px;
                    background: var(--status-danger);
                    color: white;
                    border-radius: var(--radius-md);
                }

                .error-content {
                    flex: 1;
                }

                .error-title {
                    font-size: var(--text-base);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-1) 0;
                }

                .error-message {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0 0 var(--space-3) 0;
                }
            `}</style>
        </div>
    );
}

/**
 * Empty state with optional error for when no data and no error
 */
export function QueryEmpty({
    icon,
    title,
    message,
    action,
}: {
    icon?: React.ReactNode;
    title: string;
    message?: string;
    action?: React.ReactNode;
}) {
    return (
        <div className="query-empty">
            {icon && <div className="empty-icon">{icon}</div>}
            <h3 className="empty-title">{title}</h3>
            {message && <p className="empty-message">{message}</p>}
            {action && <div className="empty-action">{action}</div>}

            <style>{`
                .query-empty {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: var(--space-12);
                    text-align: center;
                }

                .empty-icon {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 64px;
                    height: 64px;
                    margin-bottom: var(--space-4);
                    background: var(--bg-elevated);
                    color: var(--text-tertiary);
                    border-radius: var(--radius-full);
                }

                .empty-title {
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-1) 0;
                }

                .empty-message {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0 0 var(--space-4) 0;
                    max-width: 320px;
                }

                .empty-action {
                    margin-top: var(--space-2);
                }
            `}</style>
        </div>
    );
}
