import { CSSProperties } from 'react'

interface SkeletonProps {
    width?: number | string
    height?: number | string
    borderRadius?: string
    style?: CSSProperties
    className?: string
}

export function Skeleton({
    width = '100%',
    height = 20,
    borderRadius,
    style,
    className = '',
}: SkeletonProps) {
    return (
        <div
            className={`skeleton ${className}`}
            style={{
                width: typeof width === 'number' ? `${width}px` : width,
                height: typeof height === 'number' ? `${height}px` : height,
                borderRadius: borderRadius || 'var(--radius-md)',
                ...style,
            }}
        />
    )
}

// Preset skeleton components for common use cases
export function SkeletonText({ lines = 3 }: { lines?: number }) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            {[...Array(lines)].map((_, i) => (
                <Skeleton
                    key={i}
                    height={16}
                    width={i === lines - 1 ? '60%' : '100%'}
                />
            ))}
        </div>
    )
}

export function SkeletonCard() {
    return (
        <div className="card">
            <Skeleton height={24} width="60%" style={{ marginBottom: 'var(--space-3)' }} />
            <Skeleton height={16} width="80%" style={{ marginBottom: 'var(--space-2)' }} />
            <Skeleton height={16} width="40%" style={{ marginBottom: 'var(--space-4)' }} />
            <Skeleton height={100} />
        </div>
    )
}
