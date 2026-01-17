import { TrendingDown } from 'lucide-react'

interface LossCurveData {
    step: number
    trainLoss: number
    valLoss?: number
}

interface LossCurveChartProps {
    data?: LossCurveData[]
    height?: number
}

export function LossCurveChart({ data, height = 240 }: LossCurveChartProps) {
    // Placeholder chart - will be replaced with actual charting library in Phase 4
    const hasData = data && data.length > 0

    return (
        <div className="chart-container">
            {hasData ? (
                <div className="chart-placeholder">
                    <div className="chart-bars">
                        {data.slice(-20).map((point, i) => (
                            <div
                                key={i}
                                className="chart-bar"
                                style={{
                                    height: `${Math.max(10, (1 - point.trainLoss / 4) * 100)}%`,
                                    opacity: 0.3 + (i / 20) * 0.7,
                                }}
                            />
                        ))}
                    </div>
                    <div className="chart-trend">
                        <TrendingDown size={20} />
                        <span>Loss trending down</span>
                    </div>
                </div>
            ) : (
                <div className="chart-empty">
                    <TrendingDown size={32} />
                    <span>Loss curve will appear here during training</span>
                </div>
            )}

            <style>{`
                .chart-container {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-4);
                    height: ${height}px;
                }

                .chart-placeholder {
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                }

                .chart-bars {
                    display: flex;
                    align-items: flex-end;
                    gap: var(--space-1);
                    flex: 1;
                    padding-bottom: var(--space-4);
                    border-bottom: 1px solid var(--border-subtle);
                }

                .chart-bar {
                    flex: 1;
                    background: linear-gradient(to top, var(--accent-primary), var(--accent-primary-hover));
                    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
                    min-height: 10%;
                }

                .chart-trend {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    padding-top: var(--space-3);
                    color: var(--status-success);
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                }

                .chart-empty {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: var(--space-3);
                    height: 100%;
                    color: var(--text-tertiary);
                    font-size: var(--text-sm);
                }
            `}</style>
        </div>
    )
}
