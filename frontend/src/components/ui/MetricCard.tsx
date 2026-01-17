import { ReactNode } from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
    label: string
    value: string | number
    unit?: string
    trend?: 'up' | 'down' | 'neutral'
    trendValue?: string
    status?: 'good' | 'warning' | 'critical'
    icon?: ReactNode
}

export function MetricCard({
    label,
    value,
    unit,
    trend,
    trendValue,
    status,
    icon,
}: MetricCardProps) {
    const statusBorderColors = {
        good: 'var(--status-success-border)',
        warning: 'var(--status-warning-border)',
        critical: 'var(--status-danger-border)',
    }

    return (
        <div
            className="card"
            style={{
                borderColor: status ? statusBorderColors[status] : undefined,
            }}
        >
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
            }}>
                <span style={{
                    fontSize: 'var(--text-sm)',
                    fontWeight: 'var(--font-medium)',
                    color: 'var(--text-secondary)',
                }}>
                    {label}
                </span>
                {icon && (
                    <span style={{ color: 'var(--text-secondary)' }}>
                        {icon}
                    </span>
                )}
            </div>

            <div style={{
                marginTop: 'var(--space-2)',
                display: 'flex',
                alignItems: 'baseline',
                gap: 'var(--space-1)',
            }}>
                <span style={{
                    fontSize: 'var(--text-2xl)',
                    fontWeight: 'var(--font-bold)',
                    color: 'var(--text-primary)',
                }}>
                    {value}
                </span>
                {unit && (
                    <span style={{
                        fontSize: 'var(--text-lg)',
                        fontWeight: 'var(--font-normal)',
                        color: 'var(--text-secondary)',
                    }}>
                        {unit}
                    </span>
                )}
            </div>

            {trend && trendValue && (
                <div style={{
                    marginTop: 'var(--space-2)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-1)',
                    fontSize: 'var(--text-sm)',
                }}>
                    {trend === 'up' && <TrendingUp size={16} style={{ color: 'var(--status-success)' }} />}
                    {trend === 'down' && <TrendingDown size={16} style={{ color: 'var(--status-danger)' }} />}
                    {trend === 'neutral' && <Minus size={16} style={{ color: 'var(--text-tertiary)' }} />}
                    <span style={{ color: 'var(--text-secondary)' }}>{trendValue}</span>
                </div>
            )}
        </div>
    )
}
