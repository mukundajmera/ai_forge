import { CheckCircle, XCircle, Clock, Loader2, Ban, AlertCircle } from 'lucide-react'

type Status = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
type ModelStatus = 'active' | 'candidate' | 'deprecated' | 'exporting'
type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'muted' | 'primary'
type BadgeSize = 'sm' | 'md'

// Status configuration for job/task badges
const statusConfig: Record<Status, {
    variant: BadgeVariant
    icon: typeof CheckCircle
    label: string
    animate?: boolean
}> = {
    queued: { variant: 'muted', icon: Clock, label: 'Queued' },
    running: { variant: 'info', icon: Loader2, label: 'Running', animate: true },
    completed: { variant: 'success', icon: CheckCircle, label: 'Completed' },
    failed: { variant: 'danger', icon: XCircle, label: 'Failed' },
    cancelled: { variant: 'muted', icon: Ban, label: 'Cancelled' },
}

const modelStatusConfig: Record<ModelStatus, {
    variant: BadgeVariant
    label: string
}> = {
    active: { variant: 'success', label: 'Active' },
    candidate: { variant: 'info', label: 'Candidate' },
    deprecated: { variant: 'muted', label: 'Deprecated' },
    exporting: { variant: 'warning', label: 'Exporting' },
}

interface BadgeProps {
    variant?: BadgeVariant
    children: React.ReactNode
    size?: BadgeSize
    dot?: boolean
}

export function Badge({
    variant = 'muted',
    children,
    size = 'md',
    dot = false,
}: BadgeProps) {
    const sizeClass = size === 'sm' ? 'text-xs px-1.5 py-0.5' : 'text-xs px-2 py-1'

    return (
        <span className={`badge badge-${variant} ${sizeClass}`}>
            {dot && <span className="badge-dot" />}
            {children}
        </span>
    )
}

interface StatusBadgeProps {
    status: Status
    size?: BadgeSize
    showIcon?: boolean
    showLabel?: boolean
}

export function StatusBadge({
    status,
    size = 'md',
    showIcon = true,
    showLabel = true,
}: StatusBadgeProps) {
    const config = statusConfig[status]
    const Icon = config.icon
    const iconSize = size === 'sm' ? 12 : 14

    return (
        <span className={`badge badge-${config.variant}`}>
            {showIcon && (
                <Icon
                    size={iconSize}
                    className={config.animate ? 'animate-spin' : ''}
                />
            )}
            {showLabel && config.label}
        </span>
    )
}

interface ModelStatusBadgeProps {
    status: ModelStatus
    size?: BadgeSize
}

export function ModelStatusBadge({ status, size = 'md' }: ModelStatusBadgeProps) {
    const config = modelStatusConfig[status]

    return (
        <Badge variant={config.variant} size={size}>
            {status === 'active' && <CheckCircle size={12} />}
            {config.label}
        </Badge>
    )
}

// Quality badge for data quality scores
interface QualityBadgeProps {
    score: number // 0-1
    size?: BadgeSize
}

export function QualityBadge({ score, size = 'md' }: QualityBadgeProps) {
    let variant: BadgeVariant = 'danger'
    if (score >= 0.8) variant = 'success'
    else if (score >= 0.6) variant = 'warning'

    return (
        <Badge variant={variant} size={size}>
            {(score * 100).toFixed(0)}%
        </Badge>
    )
}

// Count badge for nav items
interface CountBadgeProps {
    count: number
    max?: number
}

export function CountBadge({ count, max = 99 }: CountBadgeProps) {
    if (count <= 0) return null

    return (
        <span className="badge badge-primary" style={{
            minWidth: 18,
            height: 18,
            padding: '0 4px',
            fontSize: 11,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
        }}>
            {count > max ? `${max}+` : count}
        </span>
    )
}

// CSS for badge dot
const badgeDotStyles = `
.badge-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}
`

// Inject styles
if (typeof document !== 'undefined') {
    const styleId = 'badge-styles'
    if (!document.getElementById(styleId)) {
        const style = document.createElement('style')
        style.id = styleId
        style.textContent = badgeDotStyles
        document.head.appendChild(style)
    }
}
