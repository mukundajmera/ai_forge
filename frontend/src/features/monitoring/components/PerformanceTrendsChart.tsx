import { TrendingUp } from 'lucide-react'

export function PerformanceTrendsChart() {
    // Mock data for trend visualization
    const trendData = [
        { week: 'W1', codebleu: 0.72, humaneval: 0.58 },
        { week: 'W2', codebleu: 0.75, humaneval: 0.62 },
        { week: 'W3', codebleu: 0.78, humaneval: 0.65 },
        { week: 'W4', codebleu: 0.82, humaneval: 0.70 },
        { week: 'W5', codebleu: 0.84, humaneval: 0.72 },
    ]

    return (
        <div className="trends-container">
            <div className="trends-chart">
                {/* Placeholder chart */}
                <div className="chart-area">
                    <div className="chart-grid">
                        {[0.9, 0.75, 0.6, 0.45, 0.3].map((level, i) => (
                            <div key={i} className="grid-line">
                                <span className="grid-label">{(level * 100).toFixed(0)}%</span>
                            </div>
                        ))}
                    </div>
                    <div className="chart-bars">
                        {trendData.map((point, i) => (
                            <div key={i} className="bar-group">
                                <div
                                    className="bar codebleu"
                                    style={{ height: `${point.codebleu * 100}%` }}
                                    title={`CodeBLEU: ${(point.codebleu * 100).toFixed(1)}%`}
                                />
                                <div
                                    className="bar humaneval"
                                    style={{ height: `${point.humaneval * 100}%` }}
                                    title={`HumanEval: ${(point.humaneval * 100).toFixed(1)}%`}
                                />
                                <span className="bar-label">{point.week}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="trends-legend">
                <div className="legend-item">
                    <span className="legend-color codebleu" />
                    <span>CodeBLEU</span>
                    <span className="legend-value good">
                        <TrendingUp size={14} /> +16.7%
                    </span>
                </div>
                <div className="legend-item">
                    <span className="legend-color humaneval" />
                    <span>HumanEval</span>
                    <span className="legend-value good">
                        <TrendingUp size={14} /> +24.1%
                    </span>
                </div>
            </div>

            <style>{`
                .trends-container {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-5);
                }

                .trends-chart {
                    margin-bottom: var(--space-4);
                }

                .chart-area {
                    position: relative;
                    height: 200px;
                    display: flex;
                }

                .chart-grid {
                    position: absolute;
                    inset: 0;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                }

                .grid-line {
                    border-bottom: 1px dashed var(--border-subtle);
                    height: 0;
                }

                .grid-label {
                    position: absolute;
                    left: 0;
                    transform: translateY(-50%);
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    width: 40px;
                }

                .chart-bars {
                    display: flex;
                    align-items: flex-end;
                    justify-content: space-around;
                    flex: 1;
                    margin-left: 48px;
                    padding-bottom: var(--space-6);
                }

                .bar-group {
                    display: flex;
                    gap: 4px;
                    align-items: flex-end;
                    position: relative;
                    height: 100%;
                }

                .bar {
                    width: 20px;
                    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
                    transition: height var(--transition-slow);
                }

                .bar.codebleu {
                    background: var(--accent-primary);
                }

                .bar.humaneval {
                    background: var(--status-info);
                }

                .bar-label {
                    position: absolute;
                    bottom: -24px;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }

                .trends-legend {
                    display: flex;
                    gap: var(--space-6);
                    padding-top: var(--space-4);
                    border-top: 1px solid var(--border-subtle);
                }

                .legend-item {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .legend-color {
                    width: 12px;
                    height: 12px;
                    border-radius: var(--radius-sm);
                }

                .legend-color.codebleu {
                    background: var(--accent-primary);
                }

                .legend-color.humaneval {
                    background: var(--status-info);
                }

                .legend-value {
                    display: flex;
                    align-items: center;
                    gap: var(--space-1);
                    font-weight: var(--font-semibold);
                    margin-left: var(--space-2);
                }

                .legend-value.good {
                    color: var(--status-success);
                }
            `}</style>
        </div>
    )
}
