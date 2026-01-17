import { useState } from 'react'
import {
    Plus,
    RefreshCw,
    Trash2,
    GitBranch,
    Upload,
    FolderOpen,
    ChevronRight,
    Database,
    Sparkles
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/Badge'
import { EmptyState, TableSkeleton } from '@/components/ui/EmptyState'
import { AddDataSourceDialog } from './AddDataSourceDialog'
import { useDataSources, useSyncDataSource, useDeleteDataSource } from './hooks'
import { formatBytes, formatRelativeTime, truncatePath } from '@/utils/formatters'
import type { DataSource } from '@/types'

export function DataSourcesPage() {
    const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)
    const { data: dataSources, isLoading, error, refetch } = useDataSources()
    const syncMutation = useSyncDataSource()
    const deleteMutation = useDeleteDataSource()

    const getTypeIcon = (type: DataSource['type']) => {
        switch (type) {
            case 'git':
                return <GitBranch size={16} />
            case 'upload':
                return <Upload size={16} />
            case 'local':
                return <FolderOpen size={16} />
        }
    }

    const handleSync = (id: string) => {
        syncMutation.mutate(id)
    }

    const handleDelete = (id: string) => {
        if (confirm('Are you sure you want to delete this data source?')) {
            deleteMutation.mutate(id)
        }
    }

    return (
        <div className="data-sources-page">
            {/* Page Header */}
            <header className="page-header">
                <div>
                    <h1>Data Sources</h1>
                    <p>Manage your training data sources for fine-tuning</p>
                </div>
                <div className="header-actions">
                    <Button
                        variant="secondary"
                        leftIcon={<Sparkles size={16} />}
                        onClick={() => window.location.href = '/datasets'}
                    >
                        View Datasets
                    </Button>
                    <Button
                        leftIcon={<Plus size={16} />}
                        onClick={() => setIsAddDialogOpen(true)}
                    >
                        Add Data Source
                    </Button>
                </div>
            </header>

            {/* Content */}
            <div className="page-content">
                {isLoading ? (
                    <TableSkeleton rows={5} cols={7} />
                ) : error ? (
                    <div className="card" style={{ padding: 'var(--space-8)', textAlign: 'center' }}>
                        <p style={{ color: 'var(--error-500)', marginBottom: 'var(--space-4)' }}>
                            Failed to load data sources
                        </p>
                        <Button variant="secondary" onClick={() => refetch()}>
                            Try Again
                        </Button>
                    </div>
                ) : !dataSources || dataSources.length === 0 ? (
                    <EmptyState
                        icon={<Database size={64} />}
                        title="No data sources yet"
                        description="Add your first data source to start generating training data for fine-tuning."
                        action={
                            <Button leftIcon={<Plus size={16} />} onClick={() => setIsAddDialogOpen(true)}>
                                Add Data Source
                            </Button>
                        }
                    />
                ) : (
                    <div className="table-container">
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Type</th>
                                    <th>Path / URL</th>
                                    <th>Status</th>
                                    <th>Files</th>
                                    <th>Size</th>
                                    <th>Last Synced</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                {dataSources.map((source) => (
                                    <tr key={source.id}>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                                <span style={{ fontWeight: 500 }}>{source.name}</span>
                                            </div>
                                        </td>
                                        <td>
                                            <div style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: 'var(--space-2)',
                                                color: 'var(--text-secondary)'
                                            }}>
                                                {getTypeIcon(source.type)}
                                                <span style={{ textTransform: 'capitalize' }}>{source.type}</span>
                                            </div>
                                        </td>
                                        <td>
                                            <span
                                                className="tooltip"
                                                data-tooltip={source.url || source.path}
                                                style={{
                                                    color: 'var(--text-secondary)',
                                                    fontFamily: 'var(--font-mono)',
                                                    fontSize: 'var(--text-xs)'
                                                }}
                                            >
                                                {truncatePath(source.url || source.path || '-', 40)}
                                            </span>
                                        </td>
                                        <td>
                                            <StatusBadge status={source.status} />
                                            {source.error && (
                                                <span
                                                    className="tooltip"
                                                    data-tooltip={source.error}
                                                    style={{
                                                        marginLeft: 'var(--space-2)',
                                                        cursor: 'help'
                                                    }}
                                                >
                                                    ⚠️
                                                </span>
                                            )}
                                        </td>
                                        <td>{source.fileCount.toLocaleString()}</td>
                                        <td>{formatBytes(source.totalSize)}</td>
                                        <td style={{ color: 'var(--text-secondary)' }}>
                                            {formatRelativeTime(source.lastSynced)}
                                        </td>
                                        <td>
                                            <div className="table-actions">
                                                <button
                                                    className="btn btn-ghost btn-icon tooltip"
                                                    data-tooltip="Refresh"
                                                    onClick={() => handleSync(source.id)}
                                                    disabled={source.status === 'syncing' || syncMutation.isPending}
                                                >
                                                    <RefreshCw
                                                        size={14}
                                                        className={source.status === 'syncing' ? 'spinning' : ''}
                                                    />
                                                </button>
                                                <button
                                                    className="btn btn-ghost btn-icon tooltip"
                                                    data-tooltip="Delete"
                                                    onClick={() => handleDelete(source.id)}
                                                    disabled={deleteMutation.isPending}
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                                <button
                                                    className="btn btn-ghost btn-icon tooltip"
                                                    data-tooltip="View Details"
                                                >
                                                    <ChevronRight size={14} />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* Add Data Source Dialog */}
            <AddDataSourceDialog
                isOpen={isAddDialogOpen}
                onClose={() => setIsAddDialogOpen(false)}
                onSuccess={() => {
                    refetch()
                    setIsAddDialogOpen(false)
                }}
            />

            <style>{`
        .data-sources-page {
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
        
        .page-content {
          /* Content styles */
        }
        
        .spinning {
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
          .page-header {
            flex-direction: column;
            gap: var(--space-4);
          }
          
          .header-actions {
            width: 100%;
          }
          
          .header-actions .btn {
            flex: 1;
          }
        }
      `}</style>
        </div>
    )
}
