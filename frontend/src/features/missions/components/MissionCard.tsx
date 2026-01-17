import { Link } from 'react-router-dom'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import {
    AlertTriangle,
    CheckCircle,
    XCircle,
    Sparkles,
    ChevronRight,
    FileText
} from 'lucide-react'
import { formatRelativeTime } from '@/utils/formatters'
import type { Mission } from '@/types'

interface MissionCardProps {
    mission: Mission
}

const statusConfig = {
    pending_approval: {
        icon: AlertTriangle,
        badgeVariant: 'warning' as const,
        label: 'Pending Approval'
    },
    active: {
        icon: Sparkles,
        badgeVariant: 'info' as const,
        label: 'Active'
    },
    completed: {
        icon: CheckCircle,
        badgeVariant: 'success' as const,
        label: 'Completed'
    },
    failed: {
        icon: XCircle,
        badgeVariant: 'danger' as const,
        label: 'Failed'
    },
}

const typeLabels = {
    retrain_suggestion: 'Retrain Suggestion',
    deployment_approval: 'Deployment Approval',
    quality_alert: 'Quality Alert',
}

export function MissionCard({ mission }: MissionCardProps) {
    const status = statusConfig[mission.status]
    const StatusIcon = status.icon

    return (
        <div className={`mission-card ${mission.status}`}>
            <div className="card-header">
                <Badge variant={status.badgeVariant}>
                    <StatusIcon size={12} />
                    {status.label}
                </Badge>
                <span className="mission-type">{typeLabels[mission.type]}</span>
            </div>

            <h3 className="mission-title">{mission.title}</h3>
            <p className="mission-description">{mission.description}</p>

            <div className="card-meta">
                <div className="meta-item">
                    <span className="meta-label">Confidence</span>
                    <span className="meta-value">{(mission.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="meta-item">
                    <span className="meta-label">Artifacts</span>
                    <span className="meta-value">
                        <FileText size={14} />
                        {mission.artifacts.length}
                    </span>
                </div>
            </div>

            <div className="card-footer">
                <span className="created-at">
                    {formatRelativeTime(mission.createdAt)}
                </span>
                <Link to={`/missions/${mission.id}`}>
                    <Button intent="ghost" size="sm">
                        View Details
                        <ChevronRight size={14} />
                    </Button>
                </Link>
            </div>

            <style>{`
                .mission-card {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-5);
                    transition: all var(--transition-fast);
                }

                .mission-card:hover {
                    border-color: var(--border-default);
                    box-shadow: var(--shadow-md);
                }

                .mission-card.pending_approval {
                    border-left: 3px solid var(--status-warning);
                }

                .mission-card.active {
                    border-left: 3px solid var(--status-info);
                }

                .mission-card.completed {
                    border-left: 3px solid var(--status-success);
                }

                .mission-card.failed {
                    border-left: 3px solid var(--status-danger);
                }

                .card-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: var(--space-3);
                }

                .mission-type {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }

                .mission-title {
                    font-size: var(--text-base);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-2) 0;
                    line-height: var(--leading-snug);
                }

                .mission-description {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0 0 var(--space-4) 0;
                    line-height: var(--leading-relaxed);
                    display: -webkit-box;
                    -webkit-line-clamp: 2;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                }

                .card-meta {
                    display: flex;
                    gap: var(--space-6);
                    padding: var(--space-3) 0;
                    border-top: 1px solid var(--border-subtle);
                    border-bottom: 1px solid var(--border-subtle);
                    margin-bottom: var(--space-3);
                }

                .meta-item {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-1);
                }

                .meta-label {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }

                .meta-value {
                    display: flex;
                    align-items: center;
                    gap: var(--space-1);
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                }

                .card-footer {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }

                .created-at {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }
            `}</style>
        </div>
    )
}
