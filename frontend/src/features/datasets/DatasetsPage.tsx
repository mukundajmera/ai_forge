import { useState } from 'react'
import { Plus, Database, Sparkles, Download } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { EmptyState, TableSkeleton } from '@/components/ui/EmptyState'
import { DatasetGrid } from './DatasetCard'
import { useDatasets, useDeleteDataset } from './hooks'
import { api } from '@/api/client'

export function DatasetsPage() {
    const { data: datasets, isLoading, error, refetch } = useDatasets()
    const deleteMutation = useDeleteDataset()

    const handleUseForTraining = (datasetId: string) => {
        // Navigate to training wizard with dataset pre-selected
        // For now, just log
        console.log('Use for training:', datasetId)
        window.location.href = `/training?dataset=${datasetId}`
    }

    const handleDownload = async (datasetId: string) => {
        // Trigger download
        window.open(`/api/datasets/${datasetId}/download`, '_blank')
    }

    const handleDelete = (datasetId: string) => {
        deleteMutation.mutate(datasetId)
    }

    return (
        <div className="datasets-page">
            {/* Page Header */}
            <header className="page-header">
                <div>
                    <h1>Training Datasets</h1>
                    <p>Generated datasets ready for fine-tuning</p>
                </div>
                <div className="header-actions">
                    <Button
                        variant="secondary"
                        icon={<Sparkles size={16} />}
                        onClick={() => window.location.href = '/data-sources'}
                    >
                        Manage Sources
                    </Button>
                </div>
            </header>

            {/* Stats summary */}
            {datasets && datasets.length > 0 && (
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: 'var(--space-4)',
                    marginBottom: 'var(--space-8)'
                }}>
                    <StatCard
                        label="Total Datasets"
                        value={datasets.length}
                    />
                    <StatCard
                        label="Ready"
                        value={datasets.filter(d => d.status === 'ready').length}
                        color="var(--success-500)"
                    />
                    <StatCard
                        label="Generating"
                        value={datasets.filter(d => d.status === 'generating').length}
                        color="var(--info-500)"
                    />
                    <StatCard
                        label="Total Examples"
                        value={datasets.reduce((acc, d) => acc + d.exampleCount, 0).toLocaleString()}
                    />
                </div>
            )}

            {/* Content */}
            <div className="page-content">
                {isLoading ? (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
                        gap: 'var(--space-6)'
                    }}>
                        {[1, 2, 3].map(i => (
                            <div key={i} className="card" style={{ height: 280 }}>
                                <div className="skeleton" style={{ height: 24, width: '60%', marginBottom: 16 }} />
                                <div className="skeleton" style={{ height: 16, width: '40%', marginBottom: 24 }} />
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                                    <div className="skeleton" style={{ height: 40 }} />
                                    <div className="skeleton" style={{ height: 40 }} />
                                </div>
                            </div>
                        ))}
                    </div>
                ) : error ? (
                    <div className="card" style={{ padding: 'var(--space-8)', textAlign: 'center' }}>
                        <p style={{ color: 'var(--error-500)', marginBottom: 'var(--space-4)' }}>
                            Failed to load datasets
                        </p>
                        <Button variant="secondary" onClick={() => refetch()}>
                            Try Again
                        </Button>
                    </div>
                ) : !datasets || datasets.length === 0 ? (
                    <EmptyState
                        icon={<Database size={64} />}
                        title="No datasets yet"
                        description="Add data sources and generate training datasets to get started."
                        action={
                            <Button
                                icon={<Plus size={16} />}
                                onClick={() => window.location.href = '/data-sources'}
                            >
                                Add Data Source
                            </Button>
                        }
                    />
                ) : (
                    <DatasetGrid
                        datasets={datasets}
                        onUseForTraining={handleUseForTraining}
                        onDownload={handleDownload}
                        onDelete={handleDelete}
                    />
                )}
            </div>

            <style>{`
        .datasets-page {
          padding: var(--space-8);
          max-width: 1400px;
        }
        
        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: var(--space-8);
        }
        
        .page-header h1 {
          margin-bottom: var(--space-1);
        }
        
        .page-header p {
          color: var(--text-secondary);
          font-size: var(--text-sm);
        }
        
        .header-actions {
          display: flex;
          gap: var(--space-3);
        }
      `}</style>
        </div>
    )
}

function StatCard({
    label,
    value,
    color
}: {
    label: string;
    value: number | string;
    color?: string;
}) {
    return (
        <div className="card" style={{ padding: 'var(--space-4)' }}>
            <div style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--text-muted)',
                marginBottom: 'var(--space-1)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
            }}>
                {label}
            </div>
            <div style={{
                fontSize: 'var(--text-2xl)',
                fontWeight: 700,
                color: color || 'var(--text-primary)'
            }}>
                {value}
            </div>
        </div>
    )
}
