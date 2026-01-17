import { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/Button';

interface Props {
    children: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error?: Error;
    errorInfo?: React.ErrorInfo;
}

/**
 * Error Boundary component for catching and displaying React errors
 * Wraps the entire app to prevent crashes from propagating
 */
export class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
        // Log to error reporting service in production
        console.error('Error caught by boundary:', error, errorInfo);
        this.setState({ errorInfo });
    }

    handleReset = (): void => {
        this.setState({ hasError: false, error: undefined, errorInfo: undefined });
    };

    handleReload = (): void => {
        window.location.reload();
    };

    render(): ReactNode {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback;
            }

            return (
                <div className="error-boundary">
                    <div className="error-content">
                        <div className="error-icon">
                            <AlertTriangle size={48} />
                        </div>
                        <h1 className="error-title">Something went wrong</h1>
                        <p className="error-message">
                            {this.state.error?.message || 'An unexpected error occurred'}
                        </p>

                        {import.meta.env.DEV && this.state.error && (
                            <details className="error-details">
                                <summary>Error Details</summary>
                                <pre>{this.state.error.stack}</pre>
                            </details>
                        )}

                        <div className="error-actions">
                            <Button
                                intent="secondary"
                                icon={<RefreshCw size={16} />}
                                onClick={this.handleReset}
                            >
                                Try Again
                            </Button>
                            <Button intent="primary" onClick={this.handleReload}>
                                Reload Page
                            </Button>
                        </div>
                    </div>

                    <style>{`
                        .error-boundary {
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            min-height: 100vh;
                            padding: var(--space-8);
                            background: var(--bg-app);
                        }

                        .error-content {
                            text-align: center;
                            max-width: 480px;
                        }

                        .error-icon {
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            width: 80px;
                            height: 80px;
                            margin-bottom: var(--space-6);
                            border-radius: 50%;
                            background: var(--status-danger-bg);
                            color: var(--status-danger);
                        }

                        .error-title {
                            font-size: var(--text-2xl);
                            font-weight: var(--font-bold);
                            color: var(--text-primary);
                            margin: 0 0 var(--space-2) 0;
                        }

                        .error-message {
                            font-size: var(--text-base);
                            color: var(--text-secondary);
                            margin: 0 0 var(--space-6) 0;
                        }

                        .error-details {
                            text-align: left;
                            margin-bottom: var(--space-6);
                            padding: var(--space-4);
                            background: var(--bg-surface);
                            border: 1px solid var(--border-subtle);
                            border-radius: var(--radius-md);
                        }

                        .error-details summary {
                            cursor: pointer;
                            font-size: var(--text-sm);
                            color: var(--text-secondary);
                            margin-bottom: var(--space-2);
                        }

                        .error-details pre {
                            font-family: var(--font-mono);
                            font-size: var(--text-xs);
                            color: var(--text-secondary);
                            white-space: pre-wrap;
                            word-break: break-all;
                            margin: 0;
                            max-height: 200px;
                            overflow: auto;
                        }

                        .error-actions {
                            display: flex;
                            gap: var(--space-3);
                            justify-content: center;
                        }
                    `}</style>
                </div>
            );
        }

        return this.props.children;
    }
}
