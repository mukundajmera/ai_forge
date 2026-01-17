import { BarChart, FileText, Terminal, Layout } from 'lucide-react'
import { formatRelativeTime } from '@/utils/formatters'
import type { Artifact } from '@/types'

interface ArtifactViewerProps {
    artifact: Artifact
}

const typeConfig = {
    chart: { icon: BarChart, label: 'Chart' },
    report: { icon: FileText, label: 'Report' },
    log: { icon: Terminal, label: 'Log' },
    dashboard: { icon: Layout, label: 'Dashboard' },
}

export function ArtifactViewer({ artifact }: ArtifactViewerProps) {
    const config = typeConfig[artifact.type]
    const Icon = config.icon

    return (
        <div className="artifact-card">
            <div className="artifact-icon">
                <Icon size={20} />
            </div>
            <div className="artifact-content">
                <span className="artifact-type">{config.label}</span>
                <span className="artifact-title">{artifact.title}</span>
                <span className="artifact-time">{formatRelativeTime(artifact.createdAt)}</span>
            </div>

            <style>{`
                .artifact-card {
                    display: flex;
                    gap: var(--space-3);
                    padding: var(--space-3);
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-md);
                    cursor: pointer;
                    transition: all var(--transition-fast);
                }

                .artifact-card:hover {
                    border-color: var(--border-default);
                    background: var(--bg-hover);
                }

                .artifact-icon {
                    width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: var(--bg-elevated);
                    border-radius: var(--radius-md);
                    color: var(--text-secondary);
                    flex-shrink: 0;
                }

                .artifact-content {
                    display: flex;
                    flex-direction: column;
                    min-width: 0;
                }

                .artifact-type {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }

                .artifact-title {
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }

                .artifact-time {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }
            `}</style>
        </div>
    )
}
