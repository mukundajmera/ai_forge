import { Download, Trash2, Zap, MoreVertical, Clock, CheckCircle, AlertCircle } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge, StatusBadge } from '@/components/ui/Badge'
import { QualityBar } from '@/components/ui/Progress'
import { formatRelativeTime, formatNumber } from '@/utils/formatters'
import type { TrainingDataset } from '@/types'

interface DatasetCardProps {
    dataset: TrainingDataset;
    onUseForTraining?: (id: string) => void;
    onDownload?: (id: string) => void;
    onDelete?: (id: string) => void;
}

export function DatasetCard({
    dataset,
    onUseForTraining,
    onDownload,
    onDelete,
}: DatasetCardProps) {
    const isReady = dataset.status === 'ready'
    const isGenerating = dataset.status === 'generating'
    const isError = dataset.status === 'error'

    return (
        <Card className="dataset-card" style={{ position: 'relative' }}>
            {/* Status indicator */}
            <div style={{
                position: 'absolute',
                top: 'var(--space-4)',
                right: 'var(--space-4)'
            }}>
                <StatusBadge status={dataset.status} />
            </div>

            {/* Header */}
            <div style={{ marginBottom: 'var(--space-4)', paddingRight: 80 }}>
                <h3 style={{
                    fontSize: 'var(--text-lg)',
                    fontWeight: 600,
                    marginBottom: 'var(--space-1)'
                }}>
                    {dataset.name}
                </h3>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    color: 'var(--text-muted)',
                    fontSize: 'var(--text-sm)'
                }}>
                    <Clock size={14} />
                    Created {formatRelativeTime(dataset.createdAt)}
                </div>
            </div>

            {/* Stats */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: 'var(--space-4)',
                marginBottom: 'var(--space-4)'
            }}>
                <StatItem label="Examples" value={formatNumber(dataset.exampleCount)} />
                <StatItem label="Sources" value={dataset.sourceIds.length} />
                <StatItem
                    label="Format"
                    value={
                        <Badge variant="neutral">
                            {dataset.format.toUpperCase()}
                        </Badge>
                    }
                />
                <StatItem
                    label="Version"
                    value={dataset.version || 1}
                />
            </div>

            {/* Quality Metrics */}
            {isReady && dataset.qualityMetrics && (
                <div style={{
                    padding: 'var(--space-3)',
                    background: 'var(--bg-elevated)',
                    borderRadius: 'var(--radius-md)',
                    marginBottom: 'var(--space-4)'
                }}>
                    <div style={{
                        fontSize: 'var(--text-xs)',
                        color: 'var(--text-muted)',
                        marginBottom: 'var(--space-2)',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                    }}>
                        Quality Metrics
                    </div>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(3, 1fr)',
                        gap: 'var(--space-2)'
                    }}>
                        <QualityMetric
                            label="Avg"
                            score={dataset.qualityMetrics.avgScore}
                        />
                        <QualityMetric
                            label="Div"
                            score={dataset.qualityMetrics.diversity}
                        />
                        <QualityMetric
                            label="RAFT"
                            score={dataset.qualityMetrics.raftDistractorQuality}
                        />
                    </div>
                </div>
            )}

            {/* Error message */}
            {isError && dataset.error && (
                <div style={{
                    padding: 'var(--space-3)',
                    background: 'rgba(239, 68, 68, 0.1)',
                    borderRadius: 'var(--radius-md)',
                    marginBottom: 'var(--space-4)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    fontSize: 'var(--text-sm)',
                    color: 'var(--error-500)'
                }}>
                    <AlertCircle size={16} />
                    {dataset.error}
                </div>
            )}

            {/* Actions */}
            <div style={{
                display: 'flex',
                gap: 'var(--space-2)',
                paddingTop: 'var(--space-4)',
                borderTop: '1px solid var(--border)'
            }}>
                {isReady && onUseForTraining && (
                    <Button
                        size="sm"
                        leftIcon={<Zap size={14} />}
                        onClick={() => onUseForTraining(dataset.id)}
                        style={{ flex: 1 }}
                    >
                        Use for Training
                    </Button>
                )}
                {isReady && onDownload && (
                    <Button
                        variant="secondary"
                        size="sm"
                        leftIcon={<Download size={14} />}
                        onClick={() => onDownload(dataset.id)}
                    >
                        JSON
                    </Button>
                )}
                {onDelete && (
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                            if (confirm('Delete this dataset?')) {
                                onDelete(dataset.id)
                            }
                        }}
                    >
                        <Trash2 size={14} />
                    </Button>
                )}
            </div>

            <style>{`
        .dataset-card {
          transition: transform var(--transition-fast), box-shadow var(--transition-fast);
        }
        
        .dataset-card:hover {
          transform: translateY(-2px);
          box-shadow: var(--shadow-md);
        }
      `}</style>
        </Card>
    )
}

function StatItem({ label, value }: { label: string; value: React.ReactNode }) {
    return (
        <div>
            <div style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--text-muted)',
                marginBottom: 'var(--space-1)'
            }}>
                {label}
            </div>
            <div style={{ fontWeight: 600 }}>{value}</div>
        </div>
    )
}

function QualityMetric({ label, score }: { label: string; score: number }) {
    return (
        <div style={{ textAlign: 'center' }}>
            <div style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--text-muted)',
                marginBottom: 'var(--space-1)'
            }}>
                {label}
            </div>
            <div style={{
                fontSize: 'var(--text-sm)',
                fontWeight: 600,
                color: score >= 0.7
                    ? 'var(--quality-high)'
                    : score >= 0.5
                        ? 'var(--quality-medium)'
                        : 'var(--quality-low)'
            }}>
                {score.toFixed(2)}
            </div>
        </div>
    )
}

// Dataset card grid component
interface DatasetGridProps {
    datasets: TrainingDataset[];
    onUseForTraining?: (id: string) => void;
    onDownload?: (id: string) => void;
    onDelete?: (id: string) => void;
}

export function DatasetGrid({
    datasets,
    onUseForTraining,
    onDownload,
    onDelete,
}: DatasetGridProps) {
    return (
        <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: 'var(--space-6)'
        }}>
            {datasets.map(dataset => (
                <DatasetCard
                    key={dataset.id}
                    dataset={dataset}
                    onUseForTraining={onUseForTraining}
                    onDownload={onDownload}
                    onDelete={onDelete}
                />
            ))}
        </div>
    )
}
