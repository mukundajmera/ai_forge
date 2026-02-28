import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
    FlaskConical,
    Search,
    GitCompareArrows,
    Trash2,
    Clock,
    CheckCircle,
    XCircle,
    Loader2,
    Tag,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { EmptyState, TableSkeleton } from '@/components/ui/EmptyState'
import { formatRelativeTime } from '@/utils/formatters'
import { useExperiments, useDeleteExperiment, useCompareExperiments } from './hooks/useExperiments'
import type { ExperimentStatus } from '@/types'

const statusFilters = [
    { value: 'all', label: 'All' },
    { value: 'running', label: 'Running' },
    { value: 'completed', label: 'Completed' },
    { value: 'failed', label: 'Failed' },
]

const statusIcons: Record<ExperimentStatus, React.ReactNode> = {
    pending: <Clock size={14} />,
    running: <Loader2 size={14} className="animate-spin" />,
    completed: <CheckCircle size={14} />,
    failed: <XCircle size={14} />,
    cancelled: <Clock size={14} />,
}

const statusClasses: Record<ExperimentStatus, string> = {
    pending: 'badge-muted',
    running: 'badge-info',
    completed: 'badge-success',
    failed: 'badge-danger',
    cancelled: 'badge-muted',
}

export function ExperimentsPage() {
    const [statusFilter, setStatusFilter] = useState('all')
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedIds, setSelectedIds] = useState<string[]>([])

    const { data: experiments = [], isLoading, error } = useExperiments(
        statusFilter !== 'all' ? { status: statusFilter } : undefined
    )
    const deleteMutation = useDeleteExperiment()

    const { data: comparison } = useCompareExperiments(selectedIds)

    const filteredExperiments = experiments.filter(exp => {
        if (searchQuery && !exp.name.toLowerCase().includes(searchQuery.toLowerCase())) return false
        return true
    })

    const toggleSelection = (id: string) => {
        setSelectedIds(prev =>
            prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
        )
    }

    const formatMetric = (value: number | undefined | null): string => {
        if (value === null || value === undefined) return '—'
        return value.toFixed(4)
    }

    return (
        <div className="experiments-page">
            <header className="page-header">
                <div>
                    <h1>Experiments</h1>
                    <p>Track, compare, and analyze training runs</p>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
                    {selectedIds.length >= 2 && (
                        <Button
                            intent="secondary"
                            icon={<GitCompareArrows size={16} />}
                        >
                            Comparing {selectedIds.length}
                        </Button>
                    )}
                    <Link to="/jobs/new">
                        <Button icon={<FlaskConical size={16} />}>
                            New Experiment
                        </Button>
                    </Link>
                </div>
            </header>

            {/* Filters */}
            <div style={{
                display: 'flex',
                gap: 'var(--space-3)',
                marginBottom: 'var(--space-6)',
                flexWrap: 'wrap',
            }}>
                <div style={{ position: 'relative', flex: '1 1 250px', maxWidth: '360px' }}>
                    <Search size={16} style={{
                        position: 'absolute',
                        left: 'var(--space-3)',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        color: 'var(--text-muted)',
                    }} />
                    <Input
                        placeholder="Search experiments..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        style={{ paddingLeft: 'var(--space-9)' }}
                    />
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                    {statusFilters.map(f => (
                        <button
                            key={f.value}
                            onClick={() => setStatusFilter(f.value)}
                            className={`filter-chip ${statusFilter === f.value ? 'active' : ''}`}
                        >
                            {f.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Comparison Panel */}
            {comparison && selectedIds.length >= 2 && (
                <div style={{
                    background: 'var(--surface-2)',
                    borderRadius: 'var(--radius-lg)',
                    padding: 'var(--space-6)',
                    marginBottom: 'var(--space-6)',
                    border: '1px solid var(--border)',
                }}>
                    <h3 style={{ margin: '0 0 var(--space-4) 0', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                        <GitCompareArrows size={18} />
                        Experiment Comparison
                    </h3>
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr>
                                    <th style={{ textAlign: 'left', padding: 'var(--space-2) var(--space-3)', borderBottom: '1px solid var(--border)' }}>Metric</th>
                                    {comparison.experiments.map(exp => (
                                        <th key={exp.id} style={{ textAlign: 'right', padding: 'var(--space-2) var(--space-3)', borderBottom: '1px solid var(--border)' }}>
                                            {exp.name}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(comparison.metricSummary).map(([metric, values]) => {
                                    const hasValues = Object.values(values).some(v => v !== null)
                                    if (!hasValues) return null
                                    return (
                                        <tr key={metric}>
                                            <td style={{ padding: 'var(--space-2) var(--space-3)', fontWeight: 500 }}>
                                                {metric.replace(/_/g, ' ')}
                                            </td>
                                            {comparison.experiments.map(exp => (
                                                <td key={exp.id} style={{ textAlign: 'right', padding: 'var(--space-2) var(--space-3)', fontFamily: 'var(--font-mono)' }}>
                                                    {formatMetric(values[exp.id])}
                                                </td>
                                            ))}
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>
                    <Button
                        intent="ghost"
                        size="sm"
                        onClick={() => setSelectedIds([])}
                        style={{ marginTop: 'var(--space-3)' }}
                    >
                        Clear Selection
                    </Button>
                </div>
            )}

            {/* Experiments List */}
            {isLoading ? (
                <TableSkeleton rows={5} cols={6} />
            ) : error ? (
                <EmptyState
                    icon={<XCircle size={48} />}
                    title="Failed to load experiments"
                    description={error instanceof Error ? error.message : 'Unknown error'}
                />
            ) : filteredExperiments.length === 0 ? (
                <EmptyState
                    icon={<FlaskConical size={48} />}
                    title="No experiments yet"
                    description="Start a training job to create your first experiment, or create one from a recipe."
                    action={
                        <Link to="/recipes">
                            <Button>Browse Recipes</Button>
                        </Link>
                    }
                />
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                    {filteredExperiments.map(exp => (
                        <div
                            key={exp.id}
                            style={{
                                background: selectedIds.includes(exp.id) ? 'var(--surface-3)' : 'var(--surface-2)',
                                borderRadius: 'var(--radius-lg)',
                                padding: 'var(--space-4) var(--space-5)',
                                border: selectedIds.includes(exp.id) ? '2px solid var(--accent)' : '1px solid var(--border)',
                                display: 'flex',
                                alignItems: 'center',
                                gap: 'var(--space-4)',
                                cursor: 'pointer',
                                transition: 'all 0.15s ease',
                            }}
                            onClick={() => toggleSelection(exp.id)}
                        >
                            <input
                                type="checkbox"
                                checked={selectedIds.includes(exp.id)}
                                onChange={() => toggleSelection(exp.id)}
                                aria-label={`Select experiment ${exp.name}`}
                                style={{ width: 16, height: 16, cursor: 'pointer' }}
                            />
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-1)' }}>
                                    <span style={{ fontWeight: 600, fontSize: 'var(--text-base)' }}>{exp.name}</span>
                                    <span className={statusClasses[exp.status]} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                                        {statusIcons[exp.status]}
                                        {exp.status}
                                    </span>
                                </div>
                                <div style={{ display: 'flex', gap: 'var(--space-4)', fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>
                                    <span>{exp.baseModel}</span>
                                    {exp.metrics?.loss !== undefined && exp.metrics.loss !== null && (
                                        <span>Loss: {exp.metrics.loss.toFixed(4)}</span>
                                    )}
                                    {exp.metrics?.perplexity !== undefined && exp.metrics.perplexity !== null && (
                                        <span>PPL: {exp.metrics.perplexity.toFixed(2)}</span>
                                    )}
                                    <span>{formatRelativeTime(exp.createdAt)}</span>
                                </div>
                                {exp.tags.length > 0 && (
                                    <div style={{ display: 'flex', gap: 'var(--space-1)', marginTop: 'var(--space-2)' }}>
                                        {exp.tags.map(tag => (
                                            <span key={tag} className="badge-muted" style={{ display: 'inline-flex', alignItems: 'center', gap: 2, fontSize: 'var(--text-xs)' }}>
                                                <Tag size={10} />{tag}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                            <Button
                                intent="ghost"
                                size="sm"
                                icon={<Trash2 size={14} />}
                                onClick={(e) => {
                                    e.stopPropagation()
                                    deleteMutation.mutate(exp.id)
                                }}
                            />
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
