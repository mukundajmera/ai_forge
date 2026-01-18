import { useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  XCircle,
  CheckCircle,
  AlertTriangle,
  Terminal,
  Settings,
  BarChart2,
  Rocket
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'
import { Tabs, TabPanel } from '@/components/ui/Tabs'
import { Skeleton } from '@/components/ui/Skeleton'
import { QueryError } from '@/components/ui/QueryError'
import { formatRelativeTime } from '@/utils/formatters'
import { useJob, useJobMetrics, useJobLogs, useCancelJob, useStartFineTune } from '@/lib/hooks'

export function JobDetailPage() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('metrics')

  // Use real hooks
  const { data: job, isLoading, error, refetch } = useJob(jobId)
  const { data: metrics } = useJobMetrics(jobId)
  const { data: logsData } = useJobLogs(jobId)
  const cancelMutation = useCancelJob()
  const rerunMutation = useStartFineTune()

  const logs = logsData?.logs || []

  if (isLoading) {
    return (
      <div className="job-detail-page">
        <Skeleton className="h-8 w-48 mb-6" />
        <Skeleton className="h-32 w-full mb-6" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (error || !job) {
    return (
      <div className="job-detail-page">
        <div className="breadcrumb">
          <Link to="/jobs">
            <ArrowLeft size={16} />
            Jobs
          </Link>
        </div>
        <QueryError
          error={error || new Error('Job not found')}
          onRetry={refetch}
        />
      </div>
    )
  }

  const isActive = job.status === 'running' || job.status === 'queued'
  const isCompleted = job.status === 'completed'
  const isFailed = job.status === 'failed'

  const handleCancel = async () => {
    if (confirm('Cancel this training job?')) {
      try {
        await cancelMutation.mutateAsync(jobId!)
        navigate('/jobs')
      } catch (err) {
        console.error('Failed to cancel job:', err)
      }
    }
  }

  const handleRerun = async (overrides?: Record<string, unknown>) => {
    try {
      const config = {
        projectName: job.projectName,
        datasetId: job.datasetId || 'ds-1',
        baseModel: overrides?.baseModel as string || job.baseModel,
        epochs: job.epochs || 3,
        learningRate: job.learningRate || 0.0001,
        rank: overrides?.rank as number || job.rank || 64,
        batchSize: overrides?.batchSize as number || job.batchSize || 4,
        method: job.method || 'pissa',
      }
      const result = await rerunMutation.mutateAsync(config)
      navigate(`/jobs/${result.jobId}`)
    } catch (err) {
      console.error('Failed to rerun job:', err)
    }
  }

  return (
    <div className="job-detail-page">
      {/* Breadcrumb */}
      <div className="breadcrumb">
        <Link to="/jobs">
          <ArrowLeft size={16} />
          Jobs
        </Link>
        <span>/</span>
        <span>{job.projectName}</span>
      </div>

      {/* Status Banner */}
      {isActive && (
        <div className="status-banner training">
          <div className="banner-content">
            <div className="banner-icon">
              <span className="spinner" />
            </div>
            <div className="banner-text">
              <h2>Training in Progress</h2>
              <p>
                Epoch {job.currentEpoch || 0}/{job.epochs || 3} •
                Step {job.currentStep || 0}/{job.totalSteps || '?'} •
                ETA: {job.eta || 'Calculating...'}
              </p>
            </div>
          </div>
          <div className="banner-progress">
            <Progress value={job.progress || 0} />
            <span>{job.progress || 0}%</span>
          </div>
          <Button
            intent="secondary"
            onClick={handleCancel}
            loading={cancelMutation.isPending}
          >
            Cancel
          </Button>
        </div>
      )}

      {isCompleted && (
        <div className="status-banner completed">
          <div className="banner-content">
            <CheckCircle size={24} />
            <div className="banner-text">
              <h2>Training Complete</h2>
              <p>Finished {job.completedAt ? formatRelativeTime(job.completedAt) : 'recently'}</p>
            </div>
          </div>
          <Link to={`/models/deploy?job=${jobId}`}>
            <Button icon={<Rocket size={16} />}>
              Evaluate & Deploy
            </Button>
          </Link>
        </div>
      )}

      {isFailed && (
        <div className="status-banner failed">
          <div className="banner-content">
            <XCircle size={24} />
            <div className="banner-text">
              <h2>Training Failed</h2>
              <p>{typeof job.error === 'string' ? job.error : job.error?.message || 'An error occurred during training'}</p>
            </div>
          </div>
        </div>
      )}

      {/* Error Suggestions (for failed jobs) */}
      {isFailed && (
        <Card className="error-suggestions">
          <h3>
            <AlertTriangle size={18} />
            Suggested Fixes
          </h3>
          <div className="suggestions-list">
            <div className="suggestion">
              <div className="suggestion-text">
                <strong>Reduce batch size</strong>
                <span>Current: {job.batchSize || 4}, Try: {Math.max(1, (job.batchSize || 4) / 2)}</span>
              </div>
              <Button
                size="sm"
                intent="secondary"
                onClick={() => handleRerun({ batchSize: Math.max(1, (job.batchSize || 4) / 2) })}
                loading={rerunMutation.isPending}
              >
                Apply & Retry
              </Button>
            </div>
            <div className="suggestion">
              <div className="suggestion-text">
                <strong>Reduce rank</strong>
                <span>Current: {job.rank || 64}, Try: {Math.max(16, (job.rank || 64) / 2)}</span>
              </div>
              <Button
                size="sm"
                intent="secondary"
                onClick={() => handleRerun({ rank: Math.max(16, (job.rank || 64) / 2) })}
                loading={rerunMutation.isPending}
              >
                Apply & Retry
              </Button>
            </div>
            <div className="suggestion">
              <div className="suggestion-text">
                <strong>Use smaller model</strong>
                <span>Current: {job.baseModel}, Try: Llama-3.2-3B</span>
              </div>
              <Button
                size="sm"
                intent="secondary"
                onClick={() => handleRerun({ baseModel: 'Llama-3.2-3B' })}
                loading={rerunMutation.isPending}
              >
                Apply & Retry
              </Button>
            </div>
          </div>
          <div className="suggestions-footer">
            <Button intent="ghost" onClick={() => handleRerun()}>
              Re-run with Same Settings
            </Button>
          </div>
        </Card>
      )}

      {/* Main Content Grid */}
      <div className="content-grid">
        {/* Left: Metrics/Loss Chart */}
        <div className="main-content">
          <Tabs
            tabs={[
              { id: 'metrics', label: 'Metrics', icon: <BarChart2 size={14} /> },
              { id: 'logs', label: 'Logs', icon: <Terminal size={14} /> },
              { id: 'config', label: 'Config', icon: <Settings size={14} /> },
            ]}
            activeTab={activeTab}
            onChange={setActiveTab}
          />

          <TabPanel value="metrics" activeValue={activeTab}>
            <Card className="metrics-card">
              <h3>Loss Curve</h3>
              <div className="loss-chart">
                {metrics?.losses && metrics.losses.length > 0 ? (
                  <div className="chart-placeholder">
                    <svg viewBox="0 0 400 200" className="mini-chart">
                      <polyline
                        fill="none"
                        stroke="var(--accent-primary)"
                        strokeWidth="2"
                        points={metrics.losses.map((loss: number, i: number) =>
                          `${(i / metrics.losses.length) * 400},${200 - (loss / Math.max(...metrics.losses)) * 180}`
                        ).join(' ')}
                      />
                    </svg>
                    <div className="current-loss">
                      <span className="loss-label">Current Loss</span>
                      <span className="loss-value">{job.loss?.toFixed(2) || '--'}</span>
                    </div>
                  </div>
                ) : (
                  <div className="chart-placeholder">
                    <svg viewBox="0 0 400 200" className="mini-chart">
                      <path
                        d="M 0 150 Q 50 100 100 80 T 200 50 T 300 40 T 400 35"
                        fill="none"
                        stroke="var(--accent-primary)"
                        strokeWidth="2"
                      />
                      <circle cx="400" cy="35" r="4" fill="var(--accent-primary)" />
                    </svg>
                    <div className="current-loss">
                      <span className="loss-label">Current Loss</span>
                      <span className="loss-value">{job.loss?.toFixed(2) || '--'}</span>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </TabPanel>

          <TabPanel value="logs" activeValue={activeTab}>
            <Card className="logs-card">
              <div className="logs-list">
                {logs.length > 0 ? (
                  logs.map((log: string, i: number) => (
                    <div key={i} className="log-entry">
                      <span className="log-message">{log}</span>
                    </div>
                  ))
                ) : (
                  <div className="empty-logs">
                    <Terminal size={24} />
                    <p>No logs available yet</p>
                  </div>
                )}
              </div>
            </Card>
          </TabPanel>

          <TabPanel value="config" activeValue={activeTab}>
            <Card>
              <div className="config-grid">
                <ConfigItem label="Base Model" value={job.baseModel} />
                <ConfigItem label="Dataset" value={job.datasetName || 'N/A'} />
                <ConfigItem label="Epochs" value={`${job.epochs || 3}`} />
                <ConfigItem label="Learning Rate" value={(job.learningRate || 0.0001).toExponential(1)} />
                <ConfigItem label="Rank (LoRA)" value={`${job.rank || 64}`} />
                <ConfigItem label="Batch Size" value={`${job.batchSize || 4}`} />
              </div>
            </Card>
          </TabPanel>
        </div>

        {/* Right: Configuration Summary */}
        <div className="sidebar">
          <Card>
            <h3>Job Info</h3>
            <div className="config-summary">
              <ConfigItem label="Status" value={job.status} />
              <ConfigItem label="Base Model" value={job.baseModel} />
              <ConfigItem label="Method" value={job.method || 'pissa'} />
              <ConfigItem label="Started" value={job.startedAt ? formatRelativeTime(job.startedAt) : 'N/A'} />
              {job.completedAt && (
                <ConfigItem label="Completed" value={formatRelativeTime(job.completedAt)} />
              )}
            </div>
          </Card>

          <Card className="sidebar-config-card">
            <h3>Configuration</h3>
            <div className="config-summary">
              <ConfigItem label="Dataset" value={job.datasetName || 'N/A'} />
              <ConfigItem label="Epochs" value={`${job.currentEpoch || 0}/${job.epochs || 3}`} />
              <ConfigItem label="Learning Rate" value={(job.learningRate || 0.0001).toExponential(1)} />
              <ConfigItem label="Rank" value={`${job.rank || 64}`} />
              <ConfigItem label="Batch Size" value={`${job.batchSize || 4}`} />
            </div>
          </Card>
        </div>
      </div>

      <style>{`
                .job-detail-page {
                    padding: var(--space-6);
                    max-width: 1200px;
                }

                .breadcrumb {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    margin-bottom: var(--space-6);
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .breadcrumb a {
                    display: flex;
                    align-items: center;
                    gap: var(--space-1);
                    color: var(--text-secondary);
                    text-decoration: none;
                }

                .breadcrumb a:hover {
                    color: var(--accent-primary);
                }

                .status-banner {
                    display: flex;
                    align-items: center;
                    gap: var(--space-6);
                    padding: var(--space-4) var(--space-6);
                    border-radius: var(--radius-lg);
                    margin-bottom: var(--space-6);
                }

                .status-banner.training {
                    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(99, 102, 241, 0.05));
                    border: 1px solid var(--accent-primary);
                }

                .status-banner.completed {
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
                    border: 1px solid var(--status-success);
                }

                .status-banner.failed {
                    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
                    border: 1px solid var(--status-danger);
                }

                .banner-content {
                    display: flex;
                    align-items: center;
                    gap: var(--space-3);
                }

                .status-banner.completed .banner-content svg {
                    color: var(--status-success);
                }

                .status-banner.failed .banner-content svg {
                    color: var(--status-danger);
                }

                .banner-text h2 {
                    font-size: var(--text-base);
                    font-weight: 600;
                    margin-bottom: var(--space-1);
                }

                .banner-text p {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .banner-progress {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    gap: var(--space-3);
                }

                .banner-progress span {
                    font-weight: 600;
                    font-size: var(--text-sm);
                }

                .spinner {
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    border: 2px solid var(--accent-primary);
                    border-top-color: transparent;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }

                @keyframes spin {
                    to { transform: rotate(360deg); }
                }

                .error-suggestions {
                    margin-bottom: var(--space-6);
                    border-color: var(--status-danger);
                }

                .error-suggestions h3 {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    color: var(--status-danger);
                    margin-bottom: var(--space-4);
                }

                .suggestions-list {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-3);
                }

                .suggestion {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: var(--space-3);
                    background: var(--bg-elevated);
                    border-radius: var(--radius-md);
                }

                .suggestion-text {
                    display: flex;
                    flex-direction: column;
                }

                .suggestion-text span {
                    font-size: var(--text-sm);
                    color: var(--text-tertiary);
                }

                .suggestions-footer {
                    margin-top: var(--space-4);
                    padding-top: var(--space-4);
                    border-top: 1px solid var(--border-subtle);
                }

                .content-grid {
                    display: grid;
                    grid-template-columns: 1fr 300px;
                    gap: var(--space-6);
                }

                .main-content .tabs {
                    margin-bottom: var(--space-4);
                }

                .metrics-card h3,
                .logs-card h3 {
                    margin-bottom: var(--space-4);
                }

                .loss-chart {
                    height: 250px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .chart-placeholder {
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }

                .mini-chart {
                    width: 100%;
                    height: 150px;
                }

                .current-loss {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    margin-top: var(--space-4);
                }

                .loss-label {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }

                .loss-value {
                    font-size: var(--text-2xl);
                    font-weight: 700;
                    color: var(--accent-primary);
                }

                .logs-list {
                    font-family: var(--font-mono);
                    font-size: var(--text-sm);
                    max-height: 300px;
                    overflow-y: auto;
                }

                .log-entry {
                    padding: var(--space-1) 0;
                }

                .log-time {
                    color: var(--text-tertiary);
                    margin-right: var(--space-2);
                }

                .log-entry.error .log-message {
                    color: var(--status-danger);
                }

                .log-entry.warning .log-message {
                    color: var(--status-warning);
                }

                .empty-logs {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: var(--space-3);
                    padding: var(--space-8);
                    color: var(--text-tertiary);
                }

                .config-grid, .config-summary {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-3);
                }

                .sidebar h3 {
                    font-size: var(--text-sm);
                    font-weight: 600;
                    margin-bottom: var(--space-4);
                    color: var(--text-secondary);
                }

                .sidebar-config-card {
                    margin-top: var(--space-4);
                }

                .checkpoints-list {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .checkpoint-item {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    font-size: var(--text-sm);
                    color: var(--status-success);
                }

                @media (max-width: 900px) {
                    .content-grid {
                        grid-template-columns: 1fr;
                    }
                }
            `}</style>
    </div>
  )
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      fontSize: 'var(--text-sm)'
    }}>
      <span style={{ color: 'var(--text-tertiary)' }}>{label}</span>
      <span style={{ fontWeight: 500 }}>{value}</span>
    </div>
  )
}
