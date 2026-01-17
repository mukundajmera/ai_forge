// =============================================================================
// Mission Detail Page - Approval Flow with Agent Analysis
// =============================================================================

import { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Dialog } from '@/components/ui/Dialog';
import { Skeleton } from '@/components/ui/Skeleton';
import {
    CheckCircle,
    XCircle,
    ExternalLink,
    Download,
    ArrowLeft,
    Clock,
    Zap,
    Brain,
    Target
} from 'lucide-react';
import { useMission, useApproveMission, useRejectMission } from '@/lib/hooks';
import { formatDistanceToNow } from 'date-fns';
import type { Artifact, MissionPriority } from '@/lib/types';

// =============================================================================
// Configuration
// =============================================================================

const PRIORITY_STYLES: Record<MissionPriority, 'muted' | 'warning' | 'danger'> = {
    low: 'muted',
    medium: 'warning',
    high: 'danger',
};

// =============================================================================
// Artifact Item Component
// =============================================================================

function ArtifactItem({ artifact }: { artifact: Artifact }) {
    const handleDownload = () => {
        if (artifact.payload) {
            const blob = new Blob([JSON.stringify(artifact.payload, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${artifact.title.toLowerCase().replace(/\s+/g, '-')}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    };

    return (
        <div className="artifact-item">
            <div className="artifact-info">
                <div className="artifact-title">{artifact.title}</div>
                {artifact.description && (
                    <div className="artifact-description">{artifact.description}</div>
                )}
                <div className="artifact-meta">
                    <span className="artifact-type">{artifact.type}</span>
                    {artifact.format && (
                        <>
                            <span className="meta-dot">•</span>
                            <span>{artifact.format.toUpperCase()}</span>
                        </>
                    )}
                    {artifact.size && (
                        <>
                            <span className="meta-dot">•</span>
                            <span>{(artifact.size / 1024).toFixed(1)} KB</span>
                        </>
                    )}
                </div>
            </div>
            <div className="artifact-actions">
                {artifact.url && (
                    <Button
                        intent="ghost"
                        size="sm"
                        icon={<ExternalLink size={14} />}
                        onClick={() => window.open(artifact.url, '_blank')}
                    >
                        Open
                    </Button>
                )}
                {artifact.payload && (
                    <Button
                        intent="ghost"
                        size="sm"
                        icon={<Download size={14} />}
                        onClick={handleDownload}
                    >
                        Download
                    </Button>
                )}
            </div>
        </div>
    );
}

// =============================================================================
// Main Component
// =============================================================================

export function MissionDetailPage() {
    const { missionId } = useParams();
    const navigate = useNavigate();
    const { data: mission, isLoading, error } = useMission(missionId);
    const approveMission = useApproveMission();
    const rejectMission = useRejectMission();

    const [showRejectDialog, setShowRejectDialog] = useState(false);
    const [rejectionReason, setRejectionReason] = useState('');

    if (isLoading) {
        return (
            <div className="mission-page">
                <div className="mission-page-header">
                    <Skeleton className="skeleton-title" />
                </div>
                <div className="detail-loading">
                    <Skeleton className="skeleton-header" />
                    <div className="skeleton-grid">
                        <Skeleton className="skeleton-main" />
                        <Skeleton className="skeleton-sidebar" />
                    </div>
                </div>
            </div>
        );
    }

    if (error || !mission) {
        return (
            <div className="mission-page">
                <div className="mission-page-header">
                    <h1>Mission Not Found</h1>
                </div>
                <div className="detail-error">
                    <p>The mission you're looking for doesn't exist or has been deleted.</p>
                    <Button intent="primary" onClick={() => navigate('/missions')}>
                        Back to Missions
                    </Button>
                </div>
            </div>
        );
    }

    const isPending = mission.status === 'pending_approval';

    const handleApprove = async () => {
        await approveMission.mutateAsync({ id: mission.id });
        navigate('/missions');
    };

    const handleReject = async () => {
        if (!rejectionReason.trim()) return;
        await rejectMission.mutateAsync({ id: mission.id, reason: rejectionReason });
        setShowRejectDialog(false);
        navigate('/missions');
    };

    const confidenceClass =
        mission.confidence > 0.7 ? 'confidence-high' :
            mission.confidence > 0.5 ? 'confidence-medium' : 'confidence-low';

    return (
        <>
            <div className="mission-page">
                {/* Header */}
                <div className="mission-page-header">
                    <div className="header-content">
                        <Link to="/missions" className="back-link">
                            <ArrowLeft size={16} />
                            Back to Missions
                        </Link>
                        <h1 className="page-title">{mission.title}</h1>
                        <p className="page-subtitle">
                            Created {formatDistanceToNow(new Date(mission.createdAt), { addSuffix: true })}
                        </p>
                    </div>
                    {isPending && (
                        <div className="header-actions">
                            <Button
                                intent="secondary"
                                icon={<XCircle size={16} />}
                                onClick={() => setShowRejectDialog(true)}
                            >
                                Reject
                            </Button>
                            <Button
                                intent="primary"
                                icon={<CheckCircle size={16} />}
                                onClick={handleApprove}
                                loading={approveMission.isPending}
                            >
                                Approve
                            </Button>
                        </div>
                    )}
                </div>

                <div className="detail-grid">
                    {/* Main Content */}
                    <div className="detail-main">
                        {/* Description */}
                        <section className="detail-section">
                            <h2 className="section-title">Description</h2>
                            <p className="section-text">{mission.description}</p>
                        </section>

                        {/* Agent Reasoning */}
                        <section className="detail-section reasoning-section">
                            <h2 className="section-title">
                                <Brain size={18} />
                                Agent Analysis
                            </h2>

                            <div className="reasoning-grid">
                                <div className="reasoning-item">
                                    <h3 className="reasoning-label">
                                        <Clock size={14} />
                                        Trigger
                                    </h3>
                                    <p className="reasoning-text">{mission.reasoning.trigger}</p>
                                </div>

                                <div className="reasoning-item">
                                    <h3 className="reasoning-label">
                                        <Zap size={14} />
                                        Analysis
                                    </h3>
                                    <p className="reasoning-text">{mission.reasoning.analysis}</p>
                                </div>

                                <div className="reasoning-item">
                                    <h3 className="reasoning-label">
                                        <Target size={14} />
                                        Expected Outcome
                                    </h3>
                                    <p className="reasoning-text">{mission.reasoning.expectedOutcome}</p>
                                </div>
                            </div>
                        </section>

                        {/* Recommended Action */}
                        {mission.recommendedAction && (
                            <section className="detail-section recommendation-section">
                                <h2 className="section-title">Recommended Action</h2>
                                <div className="recommendation-content">
                                    <div className="recommendation-type">
                                        {mission.recommendedAction.type.replace(/_/g, ' ')}
                                    </div>
                                    {mission.recommendedAction.estimatedDuration && (
                                        <div className="recommendation-duration">
                                            Estimated duration: {mission.recommendedAction.estimatedDuration}
                                        </div>
                                    )}
                                    {mission.recommendedAction.params && (
                                        <pre className="recommendation-params">
                                            {JSON.stringify(mission.recommendedAction.params, null, 2)}
                                        </pre>
                                    )}
                                </div>
                            </section>
                        )}

                        {/* Artifacts */}
                        {mission.artifacts.length > 0 && (
                            <section className="detail-section">
                                <h2 className="section-title">
                                    Artifacts ({mission.artifacts.length})
                                </h2>
                                <div className="artifacts-list">
                                    {mission.artifacts.map(artifact => (
                                        <ArtifactItem key={artifact.id} artifact={artifact} />
                                    ))}
                                </div>
                            </section>
                        )}
                    </div>

                    {/* Sidebar */}
                    <div className="detail-sidebar">
                        {/* Status Card */}
                        <div className="sidebar-card">
                            <h3 className="card-title">Details</h3>
                            <div className="detail-rows">
                                <div className="detail-row">
                                    <span className="detail-label">Status</span>
                                    <Badge variant={
                                        mission.status === 'pending_approval' ? 'warning' :
                                            mission.status === 'approved' || mission.status === 'completed' ? 'success' :
                                                mission.status === 'rejected' || mission.status === 'failed' ? 'danger' :
                                                    'muted'
                                    }>
                                        {mission.status.replace(/_/g, ' ')}
                                    </Badge>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Type</span>
                                    <span className="detail-value">
                                        {mission.type.replace(/_/g, ' ')}
                                    </span>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Priority</span>
                                    <Badge variant={PRIORITY_STYLES[mission.priority]}>
                                        {mission.priority}
                                    </Badge>
                                </div>
                                <div className="detail-row">
                                    <span className="detail-label">Confidence</span>
                                    <div className="confidence-display">
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
                                </div>
                            </div>
                        </div>

                        {/* Related Entities */}
                        {(mission.relatedJobIds.length > 0 ||
                            mission.relatedModelIds.length > 0 ||
                            mission.relatedDatasetIds.length > 0) && (
                                <div className="sidebar-card">
                                    <h3 className="card-title">Related</h3>
                                    <div className="related-links">
                                        {mission.relatedJobIds.length > 0 && (
                                            <div className="related-group">
                                                <span className="related-label">Jobs:</span>
                                                {mission.relatedJobIds.map(id => (
                                                    <Link key={id} to={`/jobs/${id}`} className="related-link">
                                                        #{id.slice(0, 8)}
                                                    </Link>
                                                ))}
                                            </div>
                                        )}
                                        {mission.relatedModelIds.length > 0 && (
                                            <div className="related-group">
                                                <span className="related-label">Models:</span>
                                                {mission.relatedModelIds.map(id => (
                                                    <Link key={id} to={`/models/${id}`} className="related-link">
                                                        {id.slice(0, 12)}
                                                    </Link>
                                                ))}
                                            </div>
                                        )}
                                        {mission.relatedDatasetIds.length > 0 && (
                                            <div className="related-group">
                                                <span className="related-label">Datasets:</span>
                                                {mission.relatedDatasetIds.map(id => (
                                                    <Link key={id} to={`/datasets/${id}`} className="related-link">
                                                        {id.slice(0, 12)}
                                                    </Link>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                        {/* Approval Info */}
                        {mission.approval && (mission.approval.approvedAt || mission.approval.rejectedAt) && (
                            <div className="sidebar-card">
                                <h3 className="card-title">
                                    {mission.approval.approvedAt ? 'Approval' : 'Rejection'}
                                </h3>
                                <div className="approval-info">
                                    {mission.approval.approvedAt && (
                                        <p className="approval-time">
                                            Approved {formatDistanceToNow(new Date(mission.approval.approvedAt), { addSuffix: true })}
                                        </p>
                                    )}
                                    {mission.approval.rejectedAt && (
                                        <>
                                            <p className="approval-time">
                                                Rejected {formatDistanceToNow(new Date(mission.approval.rejectedAt), { addSuffix: true })}
                                            </p>
                                            {mission.approval.rejectionReason && (
                                                <p className="rejection-reason">
                                                    Reason: {mission.approval.rejectionReason}
                                                </p>
                                            )}
                                        </>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Reject Dialog */}
            <Dialog
                isOpen={showRejectDialog}
                onClose={() => setShowRejectDialog(false)}
                title="Reject Mission"
            >
                <div className="reject-dialog-content">
                    <p className="reject-description">
                        Please provide a reason for rejecting this mission.
                        This helps the agent learn and improve future suggestions.
                    </p>
                    <textarea
                        className="reject-textarea"
                        value={rejectionReason}
                        onChange={(e) => setRejectionReason(e.target.value)}
                        placeholder="e.g., Not enough new commits to justify retraining..."
                        rows={4}
                    />
                    <div className="reject-actions">
                        <Button intent="ghost" onClick={() => setShowRejectDialog(false)}>
                            Cancel
                        </Button>
                        <Button
                            intent="destructive"
                            onClick={handleReject}
                            disabled={!rejectionReason.trim()}
                            loading={rejectMission.isPending}
                        >
                            Reject Mission
                        </Button>
                    </div>
                </div>
            </Dialog>

            <style>{`
                .mission-page {
                    padding: var(--space-6);
                    max-width: 1200px;
                    margin: 0 auto;
                }

                .mission-page-header {
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
                    margin: var(--space-2) 0;
                }

                .page-subtitle {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0;
                }

                .skeleton-title {
                    height: 32px;
                    width: 250px;
                    border-radius: var(--radius-md);
                }

                .back-link {
                    display: inline-flex;
                    align-items: center;
                    gap: var(--space-2);
                    color: var(--text-secondary);
                    text-decoration: none;
                    font-size: var(--text-sm);
                    margin-bottom: var(--space-2);
                    transition: color 0.2s;
                }

                .back-link:hover {
                    color: var(--text-primary);
                }

                .header-actions {
                    display: flex;
                    gap: var(--space-2);
                }

                .detail-grid {
                    display: grid;
                    grid-template-columns: 1fr 300px;
                    gap: var(--space-6);
                }

                .detail-main {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-6);
                }

                .detail-sidebar {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-4);
                }

                .detail-section {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-6);
                }

                .section-title {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-4) 0;
                }

                .section-text {
                    color: var(--text-secondary);
                    line-height: 1.6;
                    margin: 0;
                }

                .reasoning-section {
                    background: var(--bg-elevated);
                }

                .reasoning-grid {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-4);
                }

                .reasoning-item {
                    padding: var(--space-4);
                    background: var(--bg-surface);
                    border-radius: var(--radius-md);
                }

                .reasoning-label {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-secondary);
                    margin: 0 0 var(--space-2) 0;
                }

                .reasoning-text {
                    font-size: var(--text-sm);
                    color: var(--text-primary);
                    line-height: 1.5;
                    margin: 0;
                }

                .recommendation-section {
                    background: rgba(99, 102, 241, 0.1);
                    border-color: rgba(99, 102, 241, 0.2);
                }

                .recommendation-section .section-title {
                    color: var(--accent-primary);
                }

                .recommendation-type {
                    font-size: var(--text-base);
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                    text-transform: capitalize;
                }

                .recommendation-duration {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin-top: var(--space-1);
                }

                .recommendation-params {
                    margin-top: var(--space-3);
                    padding: var(--space-3);
                    background: var(--bg-elevated);
                    border-radius: var(--radius-md);
                    font-size: var(--text-xs);
                    color: var(--text-secondary);
                    overflow-x: auto;
                }

                .artifacts-list {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-3);
                }

                .artifact-item {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: var(--space-3);
                    background: var(--bg-elevated);
                    border-radius: var(--radius-md);
                }

                .artifact-title {
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .artifact-description {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin-top: var(--space-1);
                }

                .artifact-meta {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    margin-top: var(--space-1);
                }

                .meta-dot {
                    color: var(--border-subtle);
                }

                .artifact-type {
                    text-transform: capitalize;
                }

                .artifact-actions {
                    display: flex;
                    gap: var(--space-2);
                }

                .sidebar-card {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-4);
                }

                .card-title {
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-3) 0;
                }

                .detail-rows {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-3);
                }

                .detail-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .detail-label {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .detail-value {
                    font-size: var(--text-sm);
                    color: var(--text-primary);
                    text-transform: capitalize;
                }

                .confidence-display {
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
                }

                .confidence-high { background: var(--status-success); }
                .confidence-medium { background: var(--status-warning); }
                .confidence-low { background: var(--status-danger); }

                .confidence-value {
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .related-links {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .related-group {
                    font-size: var(--text-sm);
                }

                .related-label {
                    color: var(--text-secondary);
                    margin-right: var(--space-2);
                }

                .related-link {
                    color: var(--accent-primary);
                    text-decoration: none;
                    margin-right: var(--space-2);
                }

                .related-link:hover {
                    text-decoration: underline;
                }

                .approval-info {
                    font-size: var(--text-sm);
                }

                .approval-time {
                    color: var(--text-secondary);
                    margin: 0;
                }

                .rejection-reason {
                    color: var(--text-primary);
                    margin: var(--space-2) 0 0 0;
                    padding: var(--space-2);
                    background: var(--bg-elevated);
                    border-radius: var(--radius-sm);
                }

                .detail-loading {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-6);
                }

                .skeleton-header {
                    height: 60px;
                    border-radius: var(--radius-lg);
                }

                .skeleton-grid {
                    display: grid;
                    grid-template-columns: 1fr 300px;
                    gap: var(--space-6);
                }

                .skeleton-main {
                    height: 400px;
                    border-radius: var(--radius-lg);
                }

                .skeleton-sidebar {
                    height: 300px;
                    border-radius: var(--radius-lg);
                }

                .detail-error {
                    text-align: center;
                    padding: var(--space-12);
                    color: var(--text-secondary);
                }

                .reject-dialog-content {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-4);
                }

                .reject-description {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0;
                }

                .reject-textarea {
                    width: 100%;
                    padding: var(--space-3);
                    border: 1px solid var(--border-default);
                    border-radius: var(--radius-md);
                    background: var(--bg-elevated);
                    color: var(--text-primary);
                    font-size: var(--text-sm);
                    resize: vertical;
                }

                .reject-textarea:focus {
                    outline: none;
                    border-color: var(--accent-primary);
                }

                .reject-actions {
                    display: flex;
                    justify-content: flex-end;
                    gap: var(--space-2);
                }

                @media (max-width: 900px) {
                    .detail-grid {
                        grid-template-columns: 1fr;
                    }

                    .detail-sidebar {
                        order: -1;
                    }
                }
            `}</style>
        </>
    );
}
