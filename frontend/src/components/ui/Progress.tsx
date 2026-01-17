import clsx from 'clsx'

interface ProgressProps {
    value: number; // 0-100
    size?: 'sm' | 'md';
    variant?: 'default' | 'success';
    showLabel?: boolean;
    className?: string;
}

export function Progress({
    value,
    size = 'md',
    variant = 'default',
    showLabel = false,
    className
}: ProgressProps) {
    const clampedValue = Math.max(0, Math.min(100, value));

    return (
        <div className={clsx('progress-wrapper', className)} style={{ width: '100%' }}>
            <div className={clsx(
                'progress',
                size === 'sm' && 'progress-sm',
                variant === 'success' && 'progress-success'
            )}>
                <div
                    className="progress-bar"
                    style={{ width: `${clampedValue}%` }}
                />
            </div>
            {showLabel && (
                <span style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--text-secondary)',
                    marginLeft: 'var(--space-2)'
                }}>
                    {Math.round(clampedValue)}%
                </span>
            )}
        </div>
    )
}

// Quality score visualization
interface QualityBarProps {
    score: number; // 0.0 - 1.0
    showValue?: boolean;
    segments?: number;
}

export function QualityBar({ score, showValue = true, segments = 5 }: QualityBarProps) {
    const level = score >= 0.7 ? 'high' : score >= 0.5 ? 'medium' : 'low';
    const filledSegments = Math.round(score * segments);

    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <div className="quality-bar">
                {Array.from({ length: segments }).map((_, i) => (
                    <div
                        key={i}
                        className={clsx(
                            'quality-bar-segment',
                            i < filledSegments && 'filled',
                            i < filledSegments && level
                        )}
                    />
                ))}
            </div>
            {showValue && (
                <span style={{
                    fontSize: 'var(--text-xs)',
                    color: `var(--quality-${level})`,
                    fontWeight: 500
                }}>
                    {score.toFixed(2)}
                </span>
            )}
        </div>
    )
}
