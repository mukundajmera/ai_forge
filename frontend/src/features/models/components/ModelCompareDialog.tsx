// =============================================================================
// Model Compare Dialog - Side-by-Side Metrics Comparison for Deployment
// =============================================================================

import { useState } from 'react';
import { Dialog } from '@/components/ui/Dialog';
import { Button } from '@/components/ui/Button';
import { useActiveModel, useValidation, useDeployModel, useActivateModel } from '@/lib/hooks';
import {
    AlertTriangle,
    TrendingUp,
    TrendingDown,
    Minus,
    Rocket,
    CheckCircle,
    Loader2
} from 'lucide-react';

// =============================================================================
// Types
// =============================================================================

interface MetricConfig {
    key: string;
    label: string;
    format: (v: number) => string;
    higherIsBetter: boolean;
    description: string;
}

const METRICS: MetricConfig[] = [
    {
        key: 'codeBleu',
        label: 'CodeBLEU',
        format: (v) => (v * 100).toFixed(1) + '%',
        higherIsBetter: true,
        description: 'Code generation quality score'
    },
    {
        key: 'humanEval',
        label: 'HumanEval Pass@1',
        format: (v) => (v * 100).toFixed(1) + '%',
        higherIsBetter: true,
        description: 'Functional correctness on coding tasks'
    },
    {
        key: 'perplexity',
        label: 'Perplexity',
        format: (v) => v.toFixed(2),
        higherIsBetter: false,
        description: 'Lower is better - model confidence'
    },
    {
        key: 'avgLatency',
        label: 'Avg Latency',
        format: (v) => v.toFixed(0) + ' ms',
        higherIsBetter: false,
        description: 'Response time per request'
    },
];

// =============================================================================
// Component
// =============================================================================

interface ModelCompareDialogProps {
    open: boolean;
    onClose: () => void;
    jobId: string;
}

export function ModelCompareDialog({ open, onClose, jobId }: ModelCompareDialogProps) {
    const [acknowledged, setAcknowledged] = useState(false);
    const { data: activeModel, isLoading: activeLoading } = useActiveModel();
    const { data: validation, isLoading: validationLoading } = useValidation(jobId);
    const deployModel = useDeployModel();
    const activateModel = useActivateModel();

    const isLoading = activeLoading || validationLoading;
    const isDeploying = deployModel.isPending || activateModel.isPending;

    // Calculate metric comparisons
    const metricComparisons = METRICS.map(metric => {
        const currentValue = Number(activeModel?.metrics?.[metric.key as keyof typeof activeModel.metrics]) || 0;
        const candidateValue = Number(validation?.[metric.key as keyof typeof validation]) || 0;
        const diff = candidateValue - currentValue;
        const percentChange = currentValue !== 0 ? ((diff / currentValue) * 100) : 0;
        const isImproved = metric.higherIsBetter ? diff > 0.001 : diff < -0.001;
        const isRegressed = metric.higherIsBetter ? diff < -0.001 : diff > 0.001;

        return {
            ...metric,
            currentValue,
            candidateValue,
            diff,
            percentChange,
            isImproved,
            isRegressed,
        };
    });

    const hasRegressions = metricComparisons.some(m => m.isRegressed);
    const hasImprovements = metricComparisons.some(m => m.isImproved);

    const handleDeploy = async () => {
        if (!validation?.modelId) return;

        try {
            await deployModel.mutateAsync(validation.modelId);
            await activateModel.mutateAsync(validation.modelId);
            onClose();
        } catch (error) {
            console.error('Deploy failed:', error);
        }
    };

    return (
        <Dialog
            isOpen={open}
            onClose={onClose}
            title="Compare & Deploy Model"
            size="lg"
        >
            {isLoading ? (
                <div className="loading-state">
                    <Loader2 className="spinner" size={32} />
                    <p>Loading validation results...</p>
                </div>
            ) : (
                <>
                    {/* Regression Warning */}
                    {hasRegressions && (
                        <div className="regression-warning">
                            <AlertTriangle size={20} />
                            <div>
                                <h4>Performance Regression Detected</h4>
                                <p>
                                    Some metrics are worse than the current active model.
                                    Review carefully before deploying.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Success Banner */}
                    {!hasRegressions && hasImprovements && (
                        <div className="success-banner">
                            <CheckCircle size={20} />
                            <div>
                                <h4>Model Ready for Deployment</h4>
                                <p>
                                    All metrics show improvement or are stable compared to the current model.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Comparison Table */}
                    <div className="comparison-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th className="text-center">Current Active</th>
                                    <th className="text-center">New Candidate</th>
                                    <th className="text-center">Change</th>
                                </tr>
                            </thead>
                            <tbody>
                                {metricComparisons.map((metric) => (
                                    <tr key={metric.key}>
                                        <td>
                                            <div className="metric-name">{metric.label}</div>
                                            <div className="metric-desc">{metric.description}</div>
                                        </td>
                                        <td className="text-center">
                                            <span className="metric-value muted">
                                                {activeModel ? metric.format(metric.currentValue) : 'N/A'}
                                            </span>
                                        </td>
                                        <td className="text-center">
                                            <span className={`metric-value ${metric.isImproved ? 'improved' :
                                                metric.isRegressed ? 'regressed' : ''
                                                }`}>
                                                {metric.format(metric.candidateValue)}
                                            </span>
                                        </td>
                                        <td className="text-center">
                                            <div className="change-indicator">
                                                {metric.isImproved && (
                                                    <>
                                                        <TrendingUp size={16} className="icon improved" />
                                                        <span className="change-value improved">
                                                            +{Math.abs(metric.percentChange).toFixed(1)}%
                                                        </span>
                                                    </>
                                                )}
                                                {metric.isRegressed && (
                                                    <>
                                                        <TrendingDown size={16} className="icon regressed" />
                                                        <span className="change-value regressed">
                                                            {metric.percentChange.toFixed(1)}%
                                                        </span>
                                                    </>
                                                )}
                                                {!metric.isImproved && !metric.isRegressed && (
                                                    <>
                                                        <Minus size={16} className="icon neutral" />
                                                        <span className="change-value neutral">~</span>
                                                    </>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Acknowledgment Checkbox */}
                    {hasRegressions && (
                        <label className="acknowledge-checkbox">
                            <input
                                type="checkbox"
                                checked={acknowledged}
                                onChange={(e) => setAcknowledged(e.target.checked)}
                            />
                            <span>
                                I understand this model has worse performance on some metrics.
                                I want to deploy it anyway.
                            </span>
                        </label>
                    )}

                    {/* Actions */}
                    <div className="dialog-actions">
                        <Button intent="ghost" onClick={onClose}>
                            Cancel
                        </Button>
                        <Button
                            intent="primary"
                            onClick={handleDeploy}
                            disabled={hasRegressions && !acknowledged}
                            loading={isDeploying}
                            icon={<Rocket size={16} />}
                        >
                            {hasRegressions ? 'Deploy Anyway' : 'Deploy & Activate'}
                        </Button>
                    </div>
                </>
            )}

            <style>{`
                .loading-state {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: var(--space-12);
                    color: var(--text-secondary);
                }

                .spinner {
                    animation: spin 1s linear infinite;
                    margin-bottom: var(--space-4);
                }

                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }

                .regression-warning {
                    display: flex;
                    gap: var(--space-3);
                    padding: var(--space-4);
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.2);
                    border-radius: var(--radius-lg);
                    margin-bottom: var(--space-6);
                }

                .regression-warning svg {
                    flex-shrink: 0;
                    color: var(--status-danger);
                }

                .regression-warning h4 {
                    font-weight: var(--font-semibold);
                    color: var(--status-danger);
                    margin: 0 0 var(--space-1) 0;
                }

                .regression-warning p {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0;
                }

                .success-banner {
                    display: flex;
                    gap: var(--space-3);
                    padding: var(--space-4);
                    background: rgba(34, 197, 94, 0.1);
                    border: 1px solid rgba(34, 197, 94, 0.2);
                    border-radius: var(--radius-lg);
                    margin-bottom: var(--space-6);
                }

                .success-banner svg {
                    flex-shrink: 0;
                    color: var(--status-success);
                }

                .success-banner h4 {
                    font-weight: var(--font-semibold);
                    color: var(--status-success);
                    margin: 0 0 var(--space-1) 0;
                }

                .success-banner p {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0;
                }

                .comparison-table {
                    overflow: hidden;
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    margin-bottom: var(--space-6);
                }

                .comparison-table table {
                    width: 100%;
                    border-collapse: collapse;
                }

                .comparison-table th {
                    padding: var(--space-3) var(--space-4);
                    background: var(--bg-elevated);
                    border-bottom: 1px solid var(--border-subtle);
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-secondary);
                    text-align: left;
                }

                .comparison-table td {
                    padding: var(--space-4);
                    border-bottom: 1px solid var(--border-subtle);
                    background: var(--bg-surface);
                }

                .comparison-table tr:last-child td {
                    border-bottom: none;
                }

                .text-center {
                    text-align: center;
                }

                .metric-name {
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .metric-desc {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    margin-top: var(--space-1);
                }

                .metric-value {
                    font-family: var(--font-mono);
                    font-weight: var(--font-semibold);
                }

                .metric-value.muted {
                    color: var(--text-secondary);
                }

                .metric-value.improved {
                    color: var(--status-success);
                }

                .metric-value.regressed {
                    color: var(--status-danger);
                }

                .change-indicator {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: var(--space-1);
                }

                .change-indicator .icon.improved {
                    color: var(--status-success);
                }

                .change-indicator .icon.regressed {
                    color: var(--status-danger);
                }

                .change-indicator .icon.neutral {
                    color: var(--text-tertiary);
                }

                .change-value {
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                }

                .change-value.improved {
                    color: var(--status-success);
                }

                .change-value.regressed {
                    color: var(--status-danger);
                }

                .change-value.neutral {
                    color: var(--text-tertiary);
                }

                .acknowledge-checkbox {
                    display: flex;
                    align-items: flex-start;
                    gap: var(--space-3);
                    padding: var(--space-4);
                    background: var(--bg-elevated);
                    border-radius: var(--radius-lg);
                    cursor: pointer;
                    margin-bottom: var(--space-6);
                }

                .acknowledge-checkbox input {
                    margin-top: var(--space-1);
                    accent-color: var(--accent-primary);
                }

                .acknowledge-checkbox span {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .dialog-actions {
                    display: flex;
                    justify-content: flex-end;
                    gap: var(--space-3);
                    padding-top: var(--space-4);
                    border-top: 1px solid var(--border-subtle);
                }
            `}</style>
        </Dialog>
    );
}
