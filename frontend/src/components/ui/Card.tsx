import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

type MetricStatus = 'default' | 'good' | 'warning' | 'critical'
type TrendDirection = 'up' | 'down' | 'neutral'

interface CardProps {
    children: React.ReactNode
    className?: string
    elevated?: boolean
    padding?: 'none' | 'sm' | 'md' | 'lg'
    onClick?: () => void
}

export function Card({
    children,
    className = '',
    elevated = false,
    padding = 'md',
    onClick,
}: CardProps) {
    const paddingClass = {
        none: '',
        sm: 'p-3',
        md: 'p-5',
        lg: 'p-6',
    }[padding]

    const classes = [
        'card',
        elevated ? 'card-elevated' : '',
        paddingClass,
        onClick ? 'cursor-pointer hover:border-color-strong' : '',
        className,
    ].filter(Boolean).join(' ')

    return (
        <div className={classes} onClick={onClick} role={onClick ? 'button' : undefined}>
            {children}
        </div>
    )
}

// Stat Card for dashboard metrics
interface StatCardProps {
    label: string
    value: string | number
    icon?: React.ReactNode
    trend?: { direction: TrendDirection; value: string }
    status?: MetricStatus
    loading?: boolean
}

export function StatCard({
    label,
    value,
    icon,
    trend,
    status = 'default',
    loading = false,
}: StatCardProps) {
    const statusColors: Record<MetricStatus, string> = {
        default: 'var(--text-primary)',
        good: 'var(--status-success)',
        warning: 'var(--status-warning)',
        critical: 'var(--status-danger)',
    }

    const trendColors: Record<TrendDirection, string> = {
        up: 'var(--status-success)',
        down: 'var(--status-danger)',
        neutral: 'var(--text-tertiary)',
    }

    const TrendIcon = trend ? {
        up: TrendingUp,
        down: TrendingDown,
        neutral: Minus,
    }[trend.direction] : null

    if (loading) {
        return (
            <Card>
                <div className="skeleton" style={{ height: 16, width: '60%', marginBottom: 8 }} />
                <div className="skeleton" style={{ height: 32, width: '40%' }} />
            </Card>
        )
    }

    return (
        <Card className="stat-card">
            <div className="stat-header">
                <span className="stat-label">{label}</span>
                {icon && <span className="stat-icon">{icon}</span>}
            </div>
            <div className="stat-value" style={{ color: statusColors[status] }}>
                {value}
            </div>
            {trend && TrendIcon && (
                <div className="stat-trend" style={{ color: trendColors[trend.direction] }}>
                    <TrendIcon size={14} />
                    <span>{trend.value}</span>
                </div>
            )}

            <style>{`
        .stat-card {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }
        
        .stat-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .stat-label {
          font-size: var(--text-sm);
          color: var(--text-secondary);
          font-weight: var(--font-medium);
        }
        
        .stat-icon {
          color: var(--text-tertiary);
        }
        
        .stat-value {
          font-size: var(--text-2xl);
          font-weight: var(--font-bold);
          line-height: 1.2;
        }
        
        .stat-trend {
          display: flex;
          align-items: center;
          gap: var(--space-1);
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
        }
      `}</style>
        </Card>
    )
}

// Metric Card for detailed metrics display
interface MetricCardProps {
    label: string
    value: string | number
    unit?: string
    description?: string
    status?: MetricStatus
    compact?: boolean
}

export function MetricCard({
    label,
    value,
    unit,
    description,
    status = 'default',
    compact = false,
}: MetricCardProps) {
    const statusColors: Record<MetricStatus, string> = {
        default: 'var(--text-primary)',
        good: 'var(--status-success)',
        warning: 'var(--status-warning)',
        critical: 'var(--status-danger)',
    }

    if (compact) {
        return (
            <div className="metric-compact">
                <span className="metric-label">{label}</span>
                <span className="metric-value" style={{ color: statusColors[status] }}>
                    {value}{unit && <span className="metric-unit">{unit}</span>}
                </span>

                <style>{`
          .metric-compact {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: var(--space-2) 0;
          }
          
          .metric-compact .metric-label {
            font-size: var(--text-sm);
            color: var(--text-secondary);
          }
          
          .metric-compact .metric-value {
            font-size: var(--text-sm);
            font-weight: var(--font-semibold);
          }
          
          .metric-unit {
            font-size: var(--text-xs);
            color: var(--text-tertiary);
            margin-left: 2px;
          }
        `}</style>
            </div>
        )
    }

    return (
        <Card padding="sm" className="metric-card">
            <div className="metric-label">{label}</div>
            <div className="metric-value" style={{ color: statusColors[status] }}>
                {value}
                {unit && <span className="metric-unit">{unit}</span>}
            </div>
            {description && <div className="metric-desc">{description}</div>}

            <style>{`
        .metric-card {
          text-align: center;
        }
        
        .metric-card .metric-label {
          font-size: var(--text-xs);
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: var(--space-1);
        }
        
        .metric-card .metric-value {
          font-size: var(--text-xl);
          font-weight: var(--font-bold);
        }
        
        .metric-card .metric-unit {
          font-size: var(--text-sm);
          font-weight: var(--font-normal);
          color: var(--text-tertiary);
          margin-left: 2px;
        }
        
        .metric-card .metric-desc {
          font-size: var(--text-xs);
          color: var(--text-tertiary);
          margin-top: var(--space-1);
        }
      `}</style>
        </Card>
    )
}
