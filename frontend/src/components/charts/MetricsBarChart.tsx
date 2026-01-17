import { BarChart } from 'lucide-react'

interface MetricsBarData {
    label: string
    value: number
    maxValue?: number
    color?: string
}

interface MetricsBarChartProps {
    data?: MetricsBarData[]
    title?: string
    height?: number
}

export function MetricsBarChart({ data, title, height = 200 }: MetricsBarChartProps) {
    const hasData = data && data.length > 0

    return (
        <div className="metrics-chart-container">
            {title && <h3 className="chart-title">{title}</h3>}

            {hasData ? (
                <div className="metrics-bars">
                    {data.map((item, i) => {
                        const percentage = (item.value / (item.maxValue || 1)) * 100
                        return (
                            <div key={i} className="metric-row">
                                <span className="metric-label">{item.label}</span>
                                <div className="metric-bar-container">
                                    <div
                                        className="metric-bar"
                                        style={{
                                            width: `${Math.min(100, percentage)}%`,
                                            background: item.color || 'var(--accent-primary)',
                                        }}
                                    />
                                </div>
                                <span className="metric-value">
                                    {typeof item.value === 'number' && item.value < 1
                                        ? (item.value * 100).toFixed(1) + '%'
                                        : item.value.toFixed(1)}
                                </span>
                            </div>
                        )
                    })}
                </div>
            ) : (
                <div className="chart-empty">
                    <BarChart size={32} />
                    <span>Metrics will appear here after evaluation</span>
                </div>
            )}

            <style>{`
                .metrics-chart-container {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-4);
                    min-height: ${height}px;
                }

                .chart-title {
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-4) 0;
                }

                .metrics-bars {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-3);
                }

                .metric-row {
                    display: flex;
                    align-items: center;
                    gap: var(--space-3);
                }

                .metric-label {
                    width: 80px;
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    flex-shrink: 0;
                }

                .metric-bar-container {
                    flex: 1;
                    height: 8px;
                    background: var(--bg-elevated);
                    border-radius: var(--radius-full);
                    overflow: hidden;
                }

                .metric-bar {
                    height: 100%;
                    border-radius: var(--radius-full);
                    transition: width var(--transition-slow);
                }

                .metric-value {
                    width: 60px;
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    text-align: right;
                    flex-shrink: 0;
                }

                .chart-empty {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: var(--space-3);
                    height: ${height - 32}px;
                    color: var(--text-tertiary);
                    font-size: var(--text-sm);
                }
            `}</style>
        </div>
    )
}
