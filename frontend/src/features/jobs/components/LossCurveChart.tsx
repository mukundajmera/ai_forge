// =============================================================================
// Loss Curve Chart - Real-time training loss visualization
// =============================================================================

import { useMemo } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from 'recharts';
import { Loader2 } from 'lucide-react';
import { useJobMetrics } from '@/lib/hooks';

interface LossCurveChartProps {
    jobId: string;
    height?: number;
}

interface DataPoint {
    step: number;
    loss: number;
}

export function LossCurveChart({ jobId, height = 320 }: LossCurveChartProps) {
    const { data: metrics, isLoading } = useJobMetrics(jobId);

    const chartData = useMemo<DataPoint[]>(() => {
        if (!metrics?.steps || !metrics?.losses) return [];

        return metrics.steps.map((step, idx) => ({
            step,
            loss: metrics.losses[idx] ?? 0,
        }));
    }, [metrics]);

    // Calculate Y-axis domain with padding
    const yDomain = useMemo((): [number, number] => {
        if (!chartData.length) return [0, 1];

        const losses = chartData.map(d => d.loss);
        const minLoss = Math.min(...losses);
        const maxLoss = Math.max(...losses);
        const padding = (maxLoss - minLoss) * 0.1 || 0.1;

        return [Math.max(0, minLoss - padding), maxLoss + padding];
    }, [chartData]);

    if (isLoading) {
        return (
            <div className="chart-loading" style={{ minHeight: height }}>
                <Loader2 className="chart-loader" />
                <span>Loading training data...</span>

                <style>{`
                    .chart-loading {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        color: var(--text-secondary);
                        gap: var(--space-2);
                    }
                    .chart-loader {
                        width: 32px;
                        height: 32px;
                        animation: spin 1s linear infinite;
                    }
                    @keyframes spin {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                `}</style>
            </div>
        );
    }

    if (!chartData.length) {
        return (
            <div className="chart-empty" style={{ minHeight: height }}>
                <span>No training data yet</span>
                <p>Loss curve will appear once training begins</p>

                <style>{`
                    .chart-empty {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        color: var(--text-secondary);
                        gap: var(--space-1);
                    }
                    .chart-empty span {
                        font-weight: var(--font-medium);
                    }
                    .chart-empty p {
                        font-size: var(--text-sm);
                        color: var(--text-tertiary);
                    }
                `}</style>
            </div>
        );
    }

    return (
        <>
            <ResponsiveContainer width="100%" height={height}>
                <LineChart
                    data={chartData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                >
                    <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgba(255,255,255,0.06)"
                    />
                    <XAxis
                        dataKey="step"
                        stroke="#666"
                        tick={{ fill: '#888', fontSize: 12 }}
                        label={{
                            value: 'Training Step',
                            position: 'insideBottom',
                            offset: -15,
                            fill: '#888',
                            fontSize: 12,
                        }}
                    />
                    <YAxis
                        stroke="#666"
                        tick={{ fill: '#888', fontSize: 12 }}
                        domain={yDomain}
                        tickFormatter={(value) => value.toFixed(3)}
                        label={{
                            value: 'Loss',
                            angle: -90,
                            position: 'insideLeft',
                            fill: '#888',
                            fontSize: 12,
                        }}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: 'var(--bg-elevated)',
                            border: '1px solid var(--border-default)',
                            borderRadius: 'var(--radius-md)',
                            color: 'var(--text-primary)',
                            fontSize: 'var(--text-sm)',
                        }}
                        formatter={(value) => [(value as number).toFixed(4), 'Loss']}
                        labelFormatter={(step) => `Step ${(step as number).toLocaleString()}`}
                    />
                    <Legend
                        wrapperStyle={{ color: '#888', fontSize: 12, paddingTop: 10 }}
                        iconType="line"
                    />
                    <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="var(--accent-primary)"
                        strokeWidth={2}
                        dot={false}
                        name="Training Loss"
                        animationDuration={300}
                        isAnimationActive={chartData.length < 500}
                    />
                </LineChart>
            </ResponsiveContainer>
        </>
    );
}
