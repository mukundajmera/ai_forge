// =============================================================================
// Job Error Panel - Error Analysis with Root Cause and Quick Fixes
// =============================================================================

import { XCircle, AlertTriangle, Info, RefreshCw, FileText, Upload, Search } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import type { TrainingJob } from '@/types';

// =============================================================================
// Error Guide Configuration
// =============================================================================

interface ErrorGuide {
    icon: typeof XCircle;
    title: string;
    color: 'danger' | 'warning' | 'muted';
    description: string;
    causes: string[];
    solutions: Array<{
        label: string;
        config?: Partial<{ batchSize: number; rank: number; baseModel: string }>;
        action?: 'regenerate' | 'preview' | 'upload' | 'quality' | 'logs' | 'retry';
    }>;
}

const ERROR_GUIDES: Record<string, ErrorGuide> = {
    OOM: {
        icon: XCircle,
        title: 'Out of Memory',
        color: 'danger',
        description: 'Training ran out of system memory. This usually happens with models that are too large for available RAM.',
        causes: [
            'Batch size too large for available memory',
            'Model rank too high',
            'Base model too large (7B/13B on 16GB system)',
            'Too many concurrent processes running',
        ],
        solutions: [
            { label: 'Reduce batch size to 1', config: { batchSize: 1 } },
            { label: 'Reduce rank to 32', config: { rank: 32 } },
            { label: 'Use 3B model instead', config: { baseModel: 'Llama-3.2-3B' } },
        ],
    },
    ValidationError: {
        icon: AlertTriangle,
        title: 'Validation Error',
        color: 'warning',
        description: 'Training data did not pass validation checks.',
        causes: [
            'Dataset format incorrect (not Alpaca or ShareGPT)',
            'Missing required fields in examples',
            'Invalid or corrupted characters in text',
            'Dataset file corrupted or incomplete',
        ],
        solutions: [
            { label: 'Re-generate dataset', action: 'regenerate' },
            { label: 'Preview dataset', action: 'preview' },
        ],
    },
    DataError: {
        icon: AlertTriangle,
        title: 'Data Processing Error',
        color: 'warning',
        description: 'Failed to load or process training dataset.',
        causes: [
            'Dataset file not found at specified path',
            'Insufficient examples in dataset (minimum 10)',
            'Data quality score too low for training',
        ],
        solutions: [
            { label: 'Upload new dataset', action: 'upload' },
            { label: 'Check dataset quality', action: 'quality' },
        ],
    },
    CUDAError: {
        icon: XCircle,
        title: 'GPU Error',
        color: 'danger',
        description: 'CUDA/GPU error occurred during training.',
        causes: [
            'GPU driver crashed or reset',
            'CUDA out of memory',
            'Incompatible GPU driver version',
            'Multiple processes competing for GPU',
        ],
        solutions: [
            { label: 'Retry training', action: 'retry' },
            { label: 'Use smaller batch size', config: { batchSize: 1 } },
            { label: 'Use 3B model', config: { baseModel: 'Llama-3.2-3B' } },
        ],
    },
    Unknown: {
        icon: Info,
        title: 'Unknown Error',
        color: 'muted',
        description: 'An unexpected error occurred during training.',
        causes: [],
        solutions: [
            { label: 'View full logs', action: 'logs' },
            { label: 'Retry with same config', action: 'retry' },
        ],
    },
};

// =============================================================================
// Component
// =============================================================================

interface JobErrorPanelProps {
    job: TrainingJob;
    onRetry?: (configOverrides: Record<string, unknown>) => void;
    onAction?: (action: string) => void;
}

export function JobErrorPanel({ job, onRetry, onAction }: JobErrorPanelProps) {
    if (job.status !== 'failed' || !job.error) return null;

    // Determine error type
    const errorType = typeof job.error === 'string'
        ? 'Unknown'
        : (job.error.type || 'Unknown');

    const errorMessage = typeof job.error === 'string'
        ? job.error
        : job.error.message;

    const suggestions = typeof job.error === 'object' && job.error.suggestions
        ? job.error.suggestions
        : [];

    const guide = ERROR_GUIDES[errorType] || ERROR_GUIDES.Unknown;
    const Icon = guide.icon;

    const colorClasses = {
        danger: 'error-panel-danger',
        warning: 'error-panel-warning',
        muted: 'error-panel-muted',
    };

    const handleSolutionClick = (solution: typeof guide.solutions[0]) => {
        if (solution.config && onRetry) {
            onRetry(solution.config);
        } else if (solution.action && onAction) {
            onAction(solution.action);
        }
    };

    const getActionIcon = (action?: string) => {
        switch (action) {
            case 'regenerate':
            case 'retry':
                return <RefreshCw size={14} />;
            case 'logs':
            case 'preview':
                return <FileText size={14} />;
            case 'upload':
                return <Upload size={14} />;
            case 'quality':
                return <Search size={14} />;
            default:
                return <RefreshCw size={14} />;
        }
    };

    return (
        <div className={`error-panel ${colorClasses[guide.color]}`}>
            {/* Header */}
            <div className="error-header">
                <Icon className="error-icon" size={24} />
                <div className="error-header-content">
                    <h3 className="error-title">{guide.title}</h3>
                    <p className="error-message">{errorMessage}</p>
                </div>
            </div>

            {/* Description */}
            <p className="error-description">{guide.description}</p>

            {/* Common Causes */}
            {guide.causes.length > 0 && (
                <div className="error-section">
                    <h4 className="section-title">Common Causes</h4>
                    <ul className="causes-list">
                        {guide.causes.map((cause, i) => (
                            <li key={i}>{cause}</li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Suggested Solutions */}
            {guide.solutions.length > 0 && (
                <div className="error-section">
                    <h4 className="section-title">Quick Fixes</h4>
                    <div className="solutions-grid">
                        {guide.solutions.map((solution, i) => (
                            <Button
                                key={i}
                                intent="secondary"
                                size="sm"
                                icon={getActionIcon(solution.action)}
                                onClick={() => handleSolutionClick(solution)}
                            >
                                {solution.label}
                            </Button>
                        ))}
                    </div>
                </div>
            )}

            {/* Backend Suggestions */}
            {suggestions.length > 0 && (
                <div className="error-section">
                    <h4 className="section-title">Recommended Actions</h4>
                    <ul className="suggestions-list">
                        {suggestions.map((suggestion, i) => (
                            <li key={i}>{suggestion}</li>
                        ))}
                    </ul>
                </div>
            )}

            <style>{`
                .error-panel {
                    padding: var(--space-6);
                    border-radius: var(--radius-lg);
                    margin-bottom: var(--space-6);
                }

                .error-panel-danger {
                    background: rgba(239, 68, 68, 0.05);
                    border: 1px solid rgba(239, 68, 68, 0.2);
                }

                .error-panel-danger .error-icon,
                .error-panel-danger .error-title {
                    color: var(--status-danger);
                }

                .error-panel-warning {
                    background: rgba(245, 158, 11, 0.05);
                    border: 1px solid rgba(245, 158, 11, 0.2);
                }

                .error-panel-warning .error-icon,
                .error-panel-warning .error-title {
                    color: var(--status-warning);
                }

                .error-panel-muted {
                    background: var(--bg-elevated);
                    border: 1px solid var(--border-subtle);
                }

                .error-panel-muted .error-icon,
                .error-panel-muted .error-title {
                    color: var(--text-secondary);
                }

                .error-header {
                    display: flex;
                    gap: var(--space-3);
                    margin-bottom: var(--space-4);
                }

                .error-icon {
                    flex-shrink: 0;
                }

                .error-header-content {
                    flex: 1;
                }

                .error-title {
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    margin: 0 0 var(--space-1) 0;
                }

                .error-message {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0;
                    font-family: var(--font-mono);
                }

                .error-description {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0 0 var(--space-4) 0;
                    line-height: 1.5;
                }

                .error-section {
                    margin-top: var(--space-4);
                    padding-top: var(--space-4);
                    border-top: 1px solid var(--border-subtle);
                }

                .section-title {
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-3) 0;
                }

                .causes-list,
                .suggestions-list {
                    margin: 0;
                    padding-left: var(--space-5);
                    list-style-type: disc;
                }

                .causes-list li,
                .suggestions-list li {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin-bottom: var(--space-1);
                    line-height: 1.5;
                }

                .solutions-grid {
                    display: flex;
                    flex-wrap: wrap;
                    gap: var(--space-2);
                }
            `}</style>
        </div>
    );
}
