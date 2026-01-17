import { CheckCircle, Clock, Play, AlertTriangle } from 'lucide-react'
import { formatRelativeTime } from '@/utils/formatters'
import type { Mission } from '@/types'

interface MissionTimelineProps {
    mission: Mission
}

export function MissionTimeline({ mission }: MissionTimelineProps) {
    const events = [
        {
            id: 'created',
            icon: Clock,
            label: 'Mission Created',
            description: 'AI detected conditions for recommendation',
            time: mission.createdAt,
            status: 'completed' as const,
        },
        ...(mission.status === 'pending_approval' ? [{
            id: 'pending',
            icon: AlertTriangle,
            label: 'Awaiting Approval',
            description: 'Waiting for user review and confirmation',
            time: mission.createdAt,
            status: 'current' as const,
        }] : []),
        ...(mission.status === 'active' ? [{
            id: 'active',
            icon: Play,
            label: 'In Progress',
            description: 'Mission is being executed',
            time: mission.createdAt,
            status: 'current' as const,
        }] : []),
        ...(mission.completedAt ? [{
            id: 'completed',
            icon: CheckCircle,
            label: mission.status === 'completed' ? 'Completed' : 'Failed',
            description: mission.status === 'completed'
                ? 'Mission successfully executed'
                : 'Mission execution failed',
            time: mission.completedAt,
            status: 'completed' as const,
        }] : []),
    ]

    return (
        <div className="timeline">
            {events.map((event, index) => {
                const Icon = event.icon
                const isLast = index === events.length - 1

                return (
                    <div key={event.id} className={`timeline-item ${event.status}`}>
                        <div className="timeline-marker">
                            <div className="marker-icon">
                                <Icon size={14} />
                            </div>
                            {!isLast && <div className="marker-line" />}
                        </div>
                        <div className="timeline-content">
                            <div className="timeline-header">
                                <span className="timeline-label">{event.label}</span>
                                <span className="timeline-time">
                                    {formatRelativeTime(event.time)}
                                </span>
                            </div>
                            <p className="timeline-description">{event.description}</p>
                        </div>
                    </div>
                )
            })}

            <style>{`
                .timeline {
                    display: flex;
                    flex-direction: column;
                }

                .timeline-item {
                    display: flex;
                    gap: var(--space-3);
                }

                .timeline-marker {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }

                .marker-icon {
                    width: 28px;
                    height: 28px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    background: var(--bg-elevated);
                    color: var(--text-secondary);
                    flex-shrink: 0;
                }

                .timeline-item.completed .marker-icon {
                    background: var(--status-success-bg);
                    color: var(--status-success);
                }

                .timeline-item.current .marker-icon {
                    background: var(--status-info-bg);
                    color: var(--status-info);
                    animation: pulse 2s infinite;
                }

                .marker-line {
                    width: 2px;
                    flex: 1;
                    min-height: 24px;
                    background: var(--border-subtle);
                    margin: var(--space-1) 0;
                }

                .timeline-content {
                    flex: 1;
                    padding-bottom: var(--space-4);
                }

                .timeline-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: var(--space-1);
                }

                .timeline-label {
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                }

                .timeline-time {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }

                .timeline-description {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0;
                }
            `}</style>
        </div>
    )
}
