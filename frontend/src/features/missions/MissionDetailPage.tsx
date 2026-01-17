import { useParams, Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { mockMissions } from '@/utils/mock-data'
import { MissionTimeline } from './components/MissionTimeline'
import { ArtifactViewer } from './components/ArtifactViewer'
import {
    ArrowLeft,
    Play,
    XCircle,
    CheckCircle,
    AlertTriangle,
    Sparkles,
    Rocket
} from 'lucide-react'
import { formatRelativeTime } from '@/utils/formatters'

export function MissionDetailPage() {
    const { missionId } = useParams()
    const mission = mockMissions.find(m => m.id === missionId)

    if (!mission) {
        return (
            <div className="not-found">
                <h2>Mission not found</h2>
                <Link to="/missions">
                    <Button intent="secondary" icon={<ArrowLeft size={16} />}>
                        Back to Missions
                    </Button>
                </Link>
            </div>
        )
    }

    const statusConfig = {
        pending_approval: { icon: AlertTriangle, color: 'var(--status-warning)', label: 'Pending Approval' },
        active: { icon: Sparkles, color: 'var(--status-info)', label: 'Active' },
        completed: { icon: CheckCircle, color: 'var(--status-success)', label: 'Completed' },
        failed: { icon: XCircle, color: 'var(--status-danger)', label: 'Failed' },
    }

    const status = statusConfig[mission.status]
    const StatusIcon = status.icon

    return (
        <div className="mission-detail-page">
            <Link to="/missions" className="back-link">
                <ArrowLeft size={16} />
                Back to Missions
            </Link>

            <PageHeader
                title={mission.title}
                subtitle={`Created ${formatRelativeTime(mission.createdAt)}`}
                actions={
                    mission.status === 'pending_approval' && (
                        <div className="action-buttons">
                            <Button intent="secondary" icon={<XCircle size={16} />}>
                                Dismiss
                            </Button>
                            <Button icon={<Play size={16} />}>
                                Approve & Execute
                            </Button>
                        </div>
                    )
                }
            />

            {/* Status Banner */}
            <div className="status-banner" style={{ borderColor: status.color }}>
                <StatusIcon size={20} style={{ color: status.color }} />
                <span style={{ color: status.color, fontWeight: 'var(--font-semibold)' }}>
                    {status.label}
                </span>
                <span className="divider" />
                <span className="confidence">
                    Confidence: {(mission.confidence * 100).toFixed(0)}%
                </span>
            </div>

            {/* Description */}
            <section className="section">
                <h3 className="section-title">Description</h3>
                <p className="description">{mission.description}</p>
            </section>

            {/* Recommended Action */}
            {mission.recommendedAction && (
                <section className="section">
                    <h3 className="section-title">Recommended Action</h3>
                    <div className="action-card">
                        <Rocket size={20} />
                        <div className="action-content">
                            <span className="action-type">
                                {mission.recommendedAction.type.replace(/_/g, ' ')}
                            </span>
                            {mission.recommendedAction.params && (
                                <pre className="action-params">
                                    {JSON.stringify(mission.recommendedAction.params, null, 2)}
                                </pre>
                            )}
                        </div>
                    </div>
                </section>
            )}

            {/* Timeline */}
            <section className="section">
                <h3 className="section-title">Timeline</h3>
                <MissionTimeline mission={mission} />
            </section>

            {/* Artifacts */}
            {mission.artifacts.length > 0 && (
                <section className="section">
                    <h3 className="section-title">Artifacts</h3>
                    <div className="artifacts-grid">
                        {mission.artifacts.map(artifact => (
                            <ArtifactViewer key={artifact.id} artifact={artifact} />
                        ))}
                    </div>
                </section>
            )}

            {/* Related Items */}
            <section className="section">
                <h3 className="section-title">Related Items</h3>
                <div className="related-items">
                    {mission.relatedModelIds.length > 0 && (
                        <div className="related-group">
                            <span className="related-label">Models:</span>
                            {mission.relatedModelIds.map(id => (
                                <Badge key={id} variant="muted">{id}</Badge>
                            ))}
                        </div>
                    )}
                    {mission.relatedJobIds.length > 0 && (
                        <div className="related-group">
                            <span className="related-label">Jobs:</span>
                            {mission.relatedJobIds.map(id => (
                                <Badge key={id} variant="muted">{id}</Badge>
                            ))}
                        </div>
                    )}
                </div>
            </section>

            <style>{`
                .mission-detail-page {
                    padding: var(--space-8);
                    max-width: 900px;
                }

                .back-link {
                    display: inline-flex;
                    align-items: center;
                    gap: var(--space-2);
                    color: var(--text-secondary);
                    font-size: var(--text-sm);
                    text-decoration: none;
                    margin-bottom: var(--space-4);
                    transition: color var(--transition-fast);
                }

                .back-link:hover {
                    color: var(--text-primary);
                }

                .action-buttons {
                    display: flex;
                    gap: var(--space-2);
                }

                .status-banner {
                    display: flex;
                    align-items: center;
                    gap: var(--space-3);
                    padding: var(--space-4);
                    background: var(--bg-surface);
                    border: 1px solid;
                    border-radius: var(--radius-lg);
                    margin-bottom: var(--space-6);
                }

                .divider {
                    width: 1px;
                    height: 20px;
                    background: var(--border-subtle);
                }

                .confidence {
                    color: var(--text-secondary);
                    font-size: var(--text-sm);
                }

                .section {
                    margin-bottom: var(--space-6);
                }

                .section-title {
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    margin: 0 0 var(--space-3) 0;
                }

                .description {
                    color: var(--text-primary);
                    line-height: var(--leading-relaxed);
                    margin: 0;
                }

                .action-card {
                    display: flex;
                    gap: var(--space-3);
                    padding: var(--space-4);
                    background: var(--accent-primary-bg);
                    border: 1px solid var(--accent-primary-border);
                    border-radius: var(--radius-lg);
                    color: var(--accent-primary);
                }

                .action-content {
                    flex: 1;
                }

                .action-type {
                    font-weight: var(--font-semibold);
                    text-transform: capitalize;
                }

                .action-params {
                    margin: var(--space-2) 0 0;
                    padding: var(--space-3);
                    background: var(--bg-app);
                    border-radius: var(--radius-md);
                    font-family: var(--font-mono);
                    font-size: var(--text-xs);
                    color: var(--text-secondary);
                    overflow-x: auto;
                }

                .artifacts-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: var(--space-3);
                }

                .related-items {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .related-group {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                }

                .related-label {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .not-found {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: var(--space-4);
                    padding: var(--space-16);
                    color: var(--text-secondary);
                }
            `}</style>
        </div>
    )
}
