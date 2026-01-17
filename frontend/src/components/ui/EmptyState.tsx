import { ReactNode } from 'react'

interface EmptyStateProps {
    icon?: ReactNode;
    title: string;
    description?: string;
    action?: ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
    return (
        <div className="empty-state">
            {icon && <div className="empty-state-icon">{icon}</div>}
            <h3 className="empty-state-title">{title}</h3>
            {description && <p style={{ marginTop: 'var(--space-2)' }}>{description}</p>}
            {action && <div style={{ marginTop: 'var(--space-6)' }}>{action}</div>}
        </div>
    )
}

// Loading skeleton components
export function Skeleton({
    width = '100%',
    height = 16,
    className = ''
}: {
    width?: string | number;
    height?: number;
    className?: string;
}) {
    return (
        <div
            className={`skeleton ${className}`}
            style={{
                width: typeof width === 'number' ? `${width}px` : width,
                height: `${height}px`
            }}
        />
    )
}

export function SkeletonRows({ count = 3 }: { count?: number }) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
            {Array.from({ length: count }).map((_, i) => (
                <Skeleton key={i} width={i === count - 1 ? '60%' : '100%'} />
            ))}
        </div>
    )
}

export function TableSkeleton({ rows = 5, cols = 5 }: { rows?: number; cols?: number }) {
    return (
        <div className="table-container">
            <table className="table">
                <thead>
                    <tr>
                        {Array.from({ length: cols }).map((_, i) => (
                            <th key={i}><Skeleton width={80} height={12} /></th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {Array.from({ length: rows }).map((_, rowIdx) => (
                        <tr key={rowIdx}>
                            {Array.from({ length: cols }).map((_, colIdx) => (
                                <td key={colIdx}>
                                    <Skeleton
                                        width={colIdx === 0 ? 150 : 80}
                                        height={14}
                                    />
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}
