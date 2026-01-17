// =============================================================================
// Metrics Grid - Real-time training progress metrics
// =============================================================================

import { TrendingDown, Clock, Zap, Target, TrendingUp } from 'lucide-react';

interface MetricsGridProps {
    currentLoss?: number;
    previousLoss?: number;
    currentStep: number;
    totalSteps: number;
    currentEpoch: number;
    totalEpochs: number;
    etaSeconds?: number;
}

interface MetricItemProps {
    label: string;
    value: string | number;
    unit?: string;
    icon: React.ReactNode;
    trend?: 'up' | 'down';
    trendValue?: string;
    status?: 'good' | 'warning' | 'critical';
}

function MetricItem({ label, value, unit, icon, trend, trendValue, status }: MetricItemProps) {
    const statusClass = status ? `metric-status-${status}` : '';

    return (
        <div className={`metric-item ${statusClass}`}>
            <div className="metric-header">
                <span className="metric-icon">{icon}</span>
                <span className="metric-label">{label}</span>
            </div>
            <div className="metric-value">
                {value}
                {unit && <span className="metric-unit">{unit}</span>}
            </div>
            {trend && trendValue && (
                <div className={`metric-trend ${trend === 'down' ? 'trend-good' : 'trend-bad'}`}>
                    {trend === 'down' ? <TrendingDown size={12} /> : <TrendingUp size={12} />}
                    <span>{trendValue}</span>
                </div>
            )}

            <style>{`
                .metric-item {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-4);
                }

                .metric-item.metric-status-good {
                    border-color: rgba(16, 185, 129, 0.3);
                }

                .metric-item.metric-status-warning {
                    border-color: rgba(251, 191, 36, 0.3);
                }

                .metric-item.metric-status-critical {
                    border-color: rgba(239, 68, 68, 0.3);
                }

                .metric-header {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    margin-bottom: var(--space-2);
                }

                .metric-icon {
                    color: var(--text-secondary);
                }

                .metric-label {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .metric-value {
                    font-size: var(--text-2xl);
                    font-weight: var(--font-bold);
                    color: var(--text-primary);
                    font-variant-numeric: tabular-nums;
                }

                .metric-unit {
                    font-size: var(--text-lg);
                    font-weight: var(--font-medium);
                    color: var(--text-secondary);
                    margin-left: var(--space-1);
                }

                .metric-trend {
                    display: flex;
                    align-items: center;
                    gap: var(--space-1);
                    font-size: var(--text-xs);
                    margin-top: var(--space-1);
                }

                .metric-trend.trend-good {
                    color: var(--status-success);
                }

                .metric-trend.trend-bad {
                    color: var(--status-danger);
                }
            `}</style>
        </div>
    );
}

export function MetricsGrid({
    currentLoss,
    previousLoss,
    currentStep,
    totalSteps,
    currentEpoch,
    totalEpochs,
    etaSeconds,
}: MetricsGridProps) {
    const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;

    // Calculate loss change
    const lossChange = previousLoss && currentLoss
        ? ((previousLoss - currentLoss) / previousLoss) * 100
        : undefined;

    // Determine loss status
    const lossStatus: 'good' | 'warning' | 'critical' | undefined =
        currentLoss !== undefined
            ? currentLoss < 1.0
                ? 'good'
                : currentLoss < 2.0
                    ? 'warning'
                    : 'critical'
            : undefined;

    // Format ETA
    const formatEta = (seconds: number): { value: string | number; unit: string } => {
        if (seconds < 60) return { value: seconds, unit: 'sec' };
        if (seconds < 3600) return { value: Math.ceil(seconds / 60), unit: 'min' };
        return { value: (seconds / 3600).toFixed(1), unit: 'hr' };
    };

    const eta = etaSeconds ? formatEta(etaSeconds) : null;

    return (
        <div className="metrics-grid">
            <MetricItem
                label="Current Loss"
                value={currentLoss !== undefined ? currentLoss.toFixed(4) : '—'}
                icon={<TrendingDown size={18} />}
                trend={lossChange !== undefined ? (lossChange > 0 ? 'down' : 'up') : undefined}
                trendValue={lossChange !== undefined ? `${Math.abs(lossChange).toFixed(1)}% from last` : undefined}
                status={lossStatus}
            />

            <MetricItem
                label="Progress"
                value={progress.toFixed(1)}
                unit="%"
                icon={<Target size={18} />}
            />

            <MetricItem
                label="Current Epoch"
                value={`${currentEpoch}/${totalEpochs}`}
                icon={<Zap size={18} />}
            />

            {eta && (
                <MetricItem
                    label="ETA"
                    value={eta.value}
                    unit={eta.unit}
                    icon={<Clock size={18} />}
                />
            )}

            <style>{`
                .metrics-grid {
                    display: grid;
                    gap: var(--space-4);
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                }

                @media (min-width: 768px) {
                    .metrics-grid {
                        grid-template-columns: repeat(4, 1fr);
                    }
                }
            `}</style>
        </div>
    );
}
