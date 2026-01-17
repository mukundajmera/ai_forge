import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
    Plus,
    Filter,
    Search,
    Play,
    CheckCircle,
    XCircle,
    Clock,
    ChevronRight,
    RefreshCw,
    MoreVertical,
    Trash2
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Badge, StatusBadge } from '@/components/ui/Badge'
import { Progress } from '@/components/ui/Progress'
import { TableSkeleton, EmptyState } from '@/components/ui/EmptyState'
import { formatRelativeTime, formatDuration } from '@/utils/formatters'
import type { TrainingJob } from './hooks/useJobs'

// Mock data for initial development
const mockJobs: TrainingJob[] = [
    {
        id: 'job-1',
        projectName: 'my-project',
        baseModel: 'Llama-3.2-3B',
        status: 'running',
        progress: 78,
        currentEpoch: 2,
        epochs: 3,
        currentStep: 156,
        totalSteps: 234,
        loss: 1.23,
        learningRate: 2e-4,
        rank: 64,
        batchSize: 2,
        datasetId: 'ds-1',
        datasetName: 'my-codebase',
        startedAt: new Date(Date.now() - 45 * 60000).toISOString(),
        eta: '12 min',
        checkpoints: ['step_100', 'step_150'],
    },
    {
        id: 'job-2',
        projectName: 'api-helper',
        baseModel: 'Llama-3.2-3B',
        status: 'completed',
        progress: 100,
        currentEpoch: 3,
        epochs: 3,
        loss: 0.89,
        learningRate: 2e-4,
        rank: 64,
        batchSize: 2,
        datasetId: 'ds-2',
        datasetName: 'api-docs',
        startedAt: new Date(Date.now() - 24 * 60 * 60000).toISOString(),
        completedAt: new Date(Date.now() - 22 * 60 * 60000).toISOString(),
        duration: '1:45:12',
        checkpoints: ['final'],
    },
    {
        id: 'job-3',
        projectName: 'test-run',
        baseModel: 'Llama-3.2-7B',
        status: 'failed',
        progress: 18,
        currentEpoch: 1,
        epochs: 3,
        currentStep: 42,
        totalSteps: 150,
        learningRate: 2e-4,
        rank: 128,
        batchSize: 4,
        datasetId: 'ds-1',
        datasetName: 'my-codebase',
        startedAt: new Date(Date.now() - 2 * 24 * 60 * 60000).toISOString(),
        error: 'OutOfMemoryError: Training ran out of memory at step 42/150',
        checkpoints: [],
    },
]

const statusFilters = [
    { value: 'all', label: 'All' },
    { value: 'running', label: 'Running' },
    { value: 'completed', label: 'Completed' },
    { value: 'failed', label: 'Failed' },
]

export function JobsListPage() {
    const [statusFilter, setStatusFilter] = useState('all')
    const [searchQuery, setSearchQuery] = useState('')

    // In production: const { data: jobs, isLoading } = useJobs({ status: statusFilter })
    const jobs = mockJobs
    const isLoading = false

    const filteredJobs = jobs.filter(job => {
        if (statusFilter !== 'all' && job.status !== statusFilter) return false
        if (searchQuery && !job.projectName.toLowerCase().includes(searchQuery.toLowerCase())) return false
        return true
    })

    const getStatusIcon = (status: TrainingJob['status']) => {
        switch (status) {
            case 'running':
            case 'queued':
                return <Play size={14} className="status-icon running" />
            case 'completed':
                return <CheckCircle size={14} className="status-icon success" />
            case 'failed':
                return <XCircle size={14} className="status-icon error" />
            case 'cancelled':
                return <Clock size={14} className="status-icon muted" />
        }
    }

    const getStatusClass = (status: TrainingJob['status']): string => {
        switch (status) {
            case 'running': return 'badge-info'
            case 'queued': return 'badge-muted'
            case 'completed': return 'badge-success'
            case 'failed': return 'badge-danger'
            case 'cancelled': return 'badge-muted'
        }
    }

    return (
        <div className="jobs-page">
            {/* Page Header */}
            <header className="page-header">
                <div>
                    <h1>Training Jobs</h1>
                    <p>All fine-tuning runs past and present</p>
                </div>
                <Link to="/jobs/new">
                    <Button icon={<Plus size={16} />}>
                        New Fine-Tune
                    </Button>
                </Link>
            </header>

            {/* Filters Bar */}
            <div className="filters-bar">
                <div className="filter-tabs">
                    {statusFilters.map(filter => (
                        <button
                            key={filter.value}
                            className={`filter-tab ${statusFilter === filter.value ? 'active' : ''}`}
                            onClick={() => setStatusFilter(filter.value)}
                        >
                            {filter.label}
                            {filter.value === 'running' && jobs.filter(j => j.status === 'running').length > 0 && (
                                <span className="filter-count">{jobs.filter(j => j.status === 'running').length}</span>
                            )}
                        </button>
                    ))}
                </div>
                <div className="filter-search">
                    <Input
                        placeholder="Search projects..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        icon={<Search size={16} />}
                    />
                </div>
            </div>

            {/* Jobs Table */}
            {isLoading ? (
                <TableSkeleton rows={5} cols={8} />
            ) : filteredJobs.length === 0 ? (
                <EmptyState
                    icon={<Play size={64} />}
                    title={searchQuery ? 'No jobs match your search' : 'No training jobs yet'}
                    description={searchQuery ? 'Try a different search term' : 'Start fine-tuning your first model with your custom code dataset.'}
                    action={!searchQuery && (
                        <Link to="/jobs/new">
                            <Button icon={<Plus size={16} />}>New Fine-Tune</Button>
                        </Link>
                    )}
                />
            ) : (
                <div className="table-container">
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Status</th>
                                <th>Project</th>
                                <th>Model</th>
                                <th>Started</th>
                                <th>Duration</th>
                                <th>Progress</th>
                                <th>Loss</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredJobs.map(job => (
                                <tr key={job.id} className={job.status === 'failed' ? 'row-error' : ''}>
                                    <td>
                                        <span className={`badge ${getStatusClass(job.status)}`}>
                                            {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                                        </span>
                                    </td>
                                    <td>
                                        <Link to={`/jobs/${job.id}`} className="job-link">
                                            <span className="job-project">{job.projectName}</span>
                                            <span className="job-dataset">{job.datasetName}</span>
                                        </Link>
                                    </td>
                                    <td style={{ color: 'var(--text-secondary)' }}>
                                        {job.baseModel}
                                    </td>
                                    <td style={{ color: 'var(--text-secondary)' }}>
                                        {formatRelativeTime(job.startedAt)}
                                    </td>
                                    <td style={{ color: 'var(--text-secondary)' }}>
                                        {job.duration || (job.status === 'running' ? job.eta : '--')}
                                    </td>
                                    <td style={{ width: 120 }}>
                                        {job.status === 'running' ? (
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                                <Progress value={job.progress} size="sm" />
                                                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                                                    {job.progress}%
                                                </span>
                                            </div>
                                        ) : job.status === 'completed' ? (
                                            <span style={{ color: 'var(--success-500)' }}>✓ Complete</span>
                                        ) : job.status === 'failed' ? (
                                            <span style={{ color: 'var(--error-500)' }}>✗ Failed</span>
                                        ) : (
                                            <span style={{ color: 'var(--text-muted)' }}>--</span>
                                        )}
                                    </td>
                                    <td>
                                        {job.loss ? (
                                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)' }}>
                                                {job.loss.toFixed(2)}
                                            </span>
                                        ) : (
                                            <span style={{ color: 'var(--text-muted)' }}>--</span>
                                        )}
                                    </td>
                                    <td>
                                        <div className="table-actions">
                                            <Link to={`/jobs/${job.id}`}>
                                                <button className="btn btn-ghost btn-icon">
                                                    <ChevronRight size={14} />
                                                </button>
                                            </Link>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            <style>{`
        .jobs-page {
          padding: var(--space-8);
          max-width: 1400px;
        }

        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: var(--space-6);
        }

        .page-header h1 {
          margin-bottom: var(--space-1);
        }

        .page-header p {
          color: var(--text-secondary);
          font-size: var(--text-sm);
        }

        .filters-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-6);
          gap: var(--space-4);
        }

        .filter-tabs {
          display: flex;
          gap: var(--space-1);
          background: var(--bg-surface);
          padding: var(--space-1);
          border-radius: var(--radius-md);
        }

        .filter-tab {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-2) var(--space-4);
          border: none;
          background: none;
          color: var(--text-secondary);
          font-size: var(--text-sm);
          font-weight: 500;
          border-radius: var(--radius-sm);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .filter-tab:hover {
          color: var(--text-primary);
        }

        .filter-tab.active {
          background: var(--bg-base);
          color: var(--text-primary);
        }

        .filter-count {
          background: var(--primary-500);
          color: white;
          padding: 0 6px;
          border-radius: var(--radius-full);
          font-size: var(--text-xs);
        }

        .filter-search {
          width: 280px;
        }

        .job-link {
          display: flex;
          flex-direction: column;
          text-decoration: none;
          color: inherit;
        }

        .job-project {
          font-weight: 600;
        }

        .job-dataset {
          font-size: var(--text-xs);
          color: var(--text-muted);
        }

        .row-error {
          background: rgba(239, 68, 68, 0.05);
        }

        .status-icon.running { color: var(--info-500); }
        .status-icon.success { color: var(--success-500); }
        .status-icon.error { color: var(--error-500); }
        .status-icon.muted { color: var(--text-muted); }
      `}</style>
        </div>
    )
}
