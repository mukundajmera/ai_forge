// =============================================================================
// Missions Page - List View with Filtering
// =============================================================================

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Badge } from '@/components/ui/Badge';
import { Skeleton } from '@/components/ui/Skeleton';
import {
    CheckCircle,
    XCircle,
    Clock,
    AlertCircle,
    TrendingUp,
    Compass,
    Zap,
    AlertTriangle,
    Database
} from 'lucide-react';
import { useMissions } from '@/lib/hooks';
import { formatDistanceToNow } from 'date-fns';
import type { Mission, MissionStatus, MissionType } from '@/lib/types';

// =============================================================================
// Configuration
// =============================================================================

const STATUS_CONFIG: Record<MissionStatus, { icon: typeof Clock; label: string; color: string }> = {
    pending_approval: { icon: Clock, label: 'Pending', color: 'warning' },
    active: { icon: TrendingUp, label: 'Active', color: 'info' },
    approved: { icon: CheckCircle, label: 'Approved', color: 'success' },
    rejected: { icon: XCircle, label: 'Rejected', color: 'danger' },
    completed: { icon: CheckCircle, label: 'Completed', color: 'success' },
    failed: { icon: AlertCircle, label: 'Failed', color: 'danger' },
};

const TYPE_CONFIG: Record<MissionType, { icon: typeof Zap; label: string }> = {
    retrain_suggestion: { icon: Zap, label: 'Retrain Suggestion' },
    deployment_approval: { icon: TrendingUp, label: 'Deployment Approval' },
    quality_alert: { icon: AlertTriangle, label: 'Quality Alert' },
    performance_drift: { icon: TrendingUp, label: 'Performance Drift' },
    data_quality_issue: { icon: Database, label: 'Data Quality' },
};

const PRIORITY_STYLES = {
    low: 'muted' as const,
    medium: 'warning' as const,
    high: 'danger' as const,
};

// =============================================================================
// Mission Card Component
// =============================================================================

interface MissionCardProps {
    mission: Mission;
    onClick: () => void;
}

function MissionCard({ mission, onClick }: MissionCardProps) {
    const statusConfig = STATUS_CONFIG[mission.status];
    const typeConfig = TYPE_CONFIG[mission.type];
    const StatusIcon = statusConfig.icon;
    const TypeIcon = typeConfig.icon;

    const confidenceClass =
        mission.confidence > 0.7 ? 'confidence-high' :
            mission.confidence > 0.5 ? 'confidence-medium' : 'confidence-low';

    return (
        <div className="mission-card" onClick={onClick}>
            {/* Header */}
            <div className="mission-header">
                <div className="mission-icon-wrapper">
                    <StatusIcon size={20} />
                </div>
                <div className="mission-header-content">
                    <h3 className="mission-title">{mission.title}</h3>
                    <p className="mission-description">{mission.description}</p>
                </div>
                <Badge variant={PRIORITY_STYLES[mission.priority]}>
                    {mission.priority}
                </Badge>
            </div>

            {/* Type & Confidence */}
            <div className="mission-meta">
                <div className="mission-type">
                    <TypeIcon size={14} />
                    <span>{typeConfig.label}</span>
                </div>
                <div className="meta-separator" />
                <div className="mission-confidence">
                    <span>Confidence:</span>
                    <div className="confidence-bar">
                        <div
                            className={`confidence-fill ${confidenceClass}`}
                            style={{ width: `${mission.confidence * 100}%` }}
                        />
                    </div>
                    <span className="confidence-value">
                        {(mission.confidence * 100).toFixed(0)}%
                    </span>
                </div>
                <div className="meta-separator" />
                <span className="mission-time">
                    {formatDistanceToNow(new Date(mission.createdAt), { addSuffix: true })}
                </span>
            </div>

            {/* Recommended Action */}
            {mission.recommendedAction && (
                <div className="mission-recommendation">
                    <span className="recommendation-label">Recommended: </span>
                    <span className="recommendation-action">
                        {mission.recommendedAction.type.replace(/_/g, ' ')}
                    </span>
                    {mission.recommendedAction.estimatedDuration && (
                        <span className="recommendation-duration">
                            (~{mission.recommendedAction.estimatedDuration})
                        </span>
                    )}
                </div>
            )}

            {/* Artifacts */}
            {mission.artifacts.length > 0 && (
                <div className="mission-artifacts">
                    {mission.artifacts.length} artifact{mission.artifacts.length > 1 ? 's' : ''} attached
                </div>
            )}

            <style>{`
                .mission-card {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-5);
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .mission-card:hover {
                    border-color: var(--border-default);
                    background: var(--bg-hover);
                }

                .mission-header {
                    display: flex;
                    align-items: flex-start;
                    gap: var(--space-3);
                    margin-bottom: var(--space-4);
                }

                .mission-icon-wrapper {
                    flex-shrink: 0;
                    padding: var(--space-2);
                    background: var(--bg-elevated);
                    border-radius: var(--radius-md);
                    color: var(--text-secondary);
                }

                .mission-header-content {
                    flex: 1;
                    min-width: 0;
                }

                .mission-title {
                    font-size: var(--text-base);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0;
                }

                .mission-description {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: var(--space-1) 0 0 0;
                    line-height: 1.4;
                }

                .mission-meta {
                    display: flex;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: var(--space-3);
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin-bottom: var(--space-3);
                }

                .meta-separator {
                    width: 1px;
                    height: 14px;
                    background: var(--border-subtle);
                }

                .mission-type {
                    display: flex;
                    align-items: center;
                    gap: var(--space-1);
                }

                .mission-confidence {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                }

                .confidence-bar {
                    width: 60px;
                    height: 6px;
                    background: var(--bg-elevated);
                    border-radius: var(--radius-full);
                    overflow: hidden;
                }

                .confidence-fill {
                    height: 100%;
                    border-radius: var(--radius-full);
                    transition: width 0.3s;
                }

                .confidence-high { background: var(--status-success); }
                .confidence-medium { background: var(--status-warning); }
                .confidence-low { background: var(--status-danger); }

                .confidence-value {
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .mission-time {
                    color: var(--text-tertiary);
                }

                .mission-recommendation {
                    padding: var(--space-2) var(--space-3);
                    background: rgba(99, 102, 241, 0.1);
                    border: 1px solid rgba(99, 102, 241, 0.2);
                    border-radius: var(--radius-md);
                    font-size: var(--text-sm);
                }

                .recommendation-label {
                    font-weight: var(--font-medium);
                    color: var(--accent-primary);
                }

                .recommendation-action {
                    color: var(--text-secondary);
                    text-transform: capitalize;
                }

                .recommendation-duration {
                    color: var(--text-tertiary);
                    margin-left: var(--space-1);
                }

                .mission-artifacts {
                    margin-top: var(--space-3);
                    font-size: var(--text-sm);
                    color: var(--text-tertiary);
                }
            `}</style>
        </div>
    );
}

// =============================================================================
// Main Component
// =============================================================================

export function MissionsPage() {
    const navigate = useNavigate();
    const [statusFilter, setStatusFilter] = useState<MissionStatus | undefined>(undefined);
    const { data, isLoading } = useMissions(statusFilter ? { status: statusFilter } : undefined);

    const missions = data?.missions || [];
    const pendingCount = data?.pending || 0;

    const tabs = [
        { id: 'all', label: 'All' },
        { id: 'pending_approval', label: `Pending (${pendingCount})` },
        { id: 'active', label: 'Active' },
        { id: 'completed', label: 'Completed' },
    ];

    return (
        <div className="missions-page">
            {/* Header */}
            <div className="missions-header">
                <div className="header-content">
                    <h1 className="page-title">Missions</h1>
                    <p className="page-subtitle">
                        AI agent suggestions and recommendations from Repo Guardian
                    </p>
                </div>
                {pendingCount > 0 && (
                    <Badge variant="warning">
                        {pendingCount} pending approval
                    </Badge>
                )}
            </div>
            {/* Filter Tabs */}
            <div className="mission-tabs">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        className={`tab-button ${(tab.id === 'all' && !statusFilter) ||
                            statusFilter === tab.id
                            ? 'active'
                            : ''
                            }`}
                        onClick={() => setStatusFilter(
                            tab.id === 'all' ? undefined : tab.id as MissionStatus
                        )}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content */}
            {isLoading ? (
                <div className="missions-grid">
                    {[...Array(4)].map((_, i) => (
                        <Skeleton key={i} className="mission-skeleton" />
                    ))}
                </div>
            ) : missions.length === 0 ? (
                <div className="missions-empty">
                    <Compass size={48} />
                    <h3>No missions yet</h3>
                    <p>
                        The Repo Guardian agent will suggest actions when it detects
                        opportunities for improvement in your codebase or models.
                    </p>
                </div>
            ) : (
                <div className="missions-grid">
                    {missions.map(mission => (
                        <MissionCard
                            key={mission.id}
                            mission={mission}
                            onClick={() => navigate(`/missions/${mission.id}`)}
                        />
                    ))}
                </div>
            )}

            <style>{`
                .missions-page {
                    padding: var(--space-6);
                    max-width: 1400px;
                    margin: 0 auto;
                }

                .missions-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    gap: var(--space-4);
                    margin-bottom: var(--space-6);
                    padding-bottom: var(--space-4);
                    border-bottom: 1px solid var(--border-subtle);
                }

                .header-content {
                    flex: 1;
                }

                .page-title {
                    font-size: var(--text-2xl);
                    font-weight: var(--font-bold);
                    color: var(--text-primary);
                    margin: 0;
                }

                .page-subtitle {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: var(--space-2) 0 0 0;
                }

                .mission-tabs {
                    display: flex;
                    gap: var(--space-2);
                    margin-bottom: var(--space-6);
                    border-bottom: 1px solid var(--border-subtle);
                    padding-bottom: var(--space-3);
                }

                .tab-button {
                    padding: var(--space-2) var(--space-4);
                    background: transparent;
                    border: none;
                    border-radius: var(--radius-md);
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-secondary);
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .tab-button:hover {
                    background: var(--bg-elevated);
                    color: var(--text-primary);
                }

                .tab-button.active {
                    background: var(--accent-primary);
                    color: white;
                }

                .missions-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                    gap: var(--space-4);
                }

                .mission-skeleton {
                    height: 180px;
                    border-radius: var(--radius-lg);
                }

                .missions-empty {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: var(--space-12);
                    text-align: center;
                    color: var(--text-secondary);
                }

                .missions-empty svg {
                    margin-bottom: var(--space-4);
                    opacity: 0.5;
                }

                .missions-empty h3 {
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-2) 0;
                }

                .missions-empty p {
                    max-width: 400px;
                    font-size: var(--text-sm);
                    line-height: 1.5;
                    margin: 0;
                }

                @media (max-width: 768px) {
                    .missions-grid {
                        grid-template-columns: 1fr;
                    }
                }
            `}</style>
        </div>
    );
}
