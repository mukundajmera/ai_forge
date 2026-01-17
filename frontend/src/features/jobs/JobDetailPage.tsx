import { useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import {
    ArrowLeft,
    RefreshCw,
    XCircle,
    Play,
    CheckCircle,
    AlertTriangle,
    Terminal,
    Settings,
    BarChart2,
    Rocket
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Progress } from '@/components/ui/Progress'
import { Tabs, TabPanel } from '@/components/ui/Tabs'
import { formatRelativeTime } from '@/utils/formatters'
import type { TrainingJob, JobLog } from './hooks/useJobs'

// Mock data
const mockJob: TrainingJob = {
    id: 'job-1',
    project: 'my-project',
    baseModel: 'Llama-3.2-3B',
    status: 'training',
    progress: 78,
    currentEpoch: 2,
    totalEpochs: 3,
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
}

const mockFailedJob: TrainingJob = {
    ...mockJob,
    id: 'job-3',
    project: 'test-run',
    status: 'failed',
    progress: 18,
    currentStep: 42,
    totalSteps: 150,
    rank: 128,
    batchSize: 4,
    baseModel: 'Llama-3.2-7B',
    error: 'OutOfMemoryError: Training ran out of memory at step 42/150',
    checkpoints: [],
}

const mockLogs: JobLog[] = [
    { timestamp: '15:42:12', level: 'info', message: 'Epoch 2 started' },
    { timestamp: '15:42:10', level: 'info', message: 'Checkpoint saved: step_150' },
    { timestamp: '15:41:45', level: 'info', message: 'Memory: 24.3GB / 32GB' },
    { timestamp: '15:41:20', level: 'info', message: 'Step 150/234, Loss: 1.25' },
    { timestamp: '15:40:55', level: 'info', message: 'Step 145/234, Loss: 1.28' },
]

export function JobDetailPage() {
    const { jobId } = useParams()
    const navigate = useNavigate()
    const [activeTab, setActiveTab] = useState('metrics')

    // In production: const { data: job } = useJob(jobId)
    const job = jobId === 'job-3' ? mockFailedJob : mockJob
    const logs = mockLogs

    const isActive = job.status === 'training' || job.status === 'queued'
    const isCompleted = job.status === 'completed'
    const isFailed = job.status === 'failed'

    const handleCancel = () => {
        if (confirm('Cancel this training job?')) {
            // cancelMutation.mutate(jobId)
            navigate('/jobs')
        }
    }

    const handleRerun = (overrides?: Partial<typeof job>) => {
        // rerunMutation.mutate({ jobId, overrides })
        console.log('Rerun with', overrides)
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
                <span>{job.project}</span>
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
                                Epoch {job.currentEpoch}/{job.totalEpochs} •
                                Step {job.currentStep}/{job.totalSteps} •
                                ETA: {job.eta}
                            </p>
                        </div>
                    </div>
                    <div className="banner-progress">
                        <Progress value={job.progress} />
                        <span>{job.progress}%</span>
                    </div>
                    <Button variant="secondary" onClick={handleCancel}>
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
                            <p>Finished {formatRelativeTime(job.completedAt!)} • Duration: {job.duration}</p>
                        </div>
                    </div>
                    <Link to={`/models/deploy?job=${jobId}`}>
                        <Button leftIcon={<Rocket size={16} />}>
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
                            <p>{job.error}</p>
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
                                <span>Current: {job.batchSize}, Try: {Math.max(1, job.batchSize / 2)}</span>
                            </div>
                            <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => handleRerun({ batchSize: Math.max(1, job.batchSize / 2) })}
                            >
                                Apply & Retry
                            </Button>
                        </div>
                        <div className="suggestion">
                            <div className="suggestion-text">
                                <strong>Reduce rank</strong>
                                <span>Current: {job.rank}, Try: {Math.max(16, job.rank / 2)}</span>
                            </div>
                            <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => handleRerun({ rank: Math.max(16, job.rank / 2) })}
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
                                variant="secondary"
                                onClick={() => handleRerun({ baseModel: 'Llama-3.2-3B' })}
                            >
                                Apply & Retry
                            </Button>
                        </div>
                    </div>
                    <div className="suggestions-footer">
                        <Button variant="ghost" onClick={() => handleRerun()}>
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
                                {/* Placeholder for chart */}
                                <div className="chart-placeholder">
                                    <svg viewBox="0 0 400 200" className="mini-chart">
                                        <path
                                            d="M 0 150 Q 50 100 100 80 T 200 50 T 300 40 T 400 35"
                                            fill="none"
                                            stroke="var(--primary-500)"
                                            strokeWidth="2"
                                        />
                                        <circle cx="400" cy="35" r="4" fill="var(--primary-400)" />
                                    </svg>
                                    <div className="current-loss">
                                        <span className="loss-label">Current Loss</span>
                                        <span className="loss-value">{job.loss?.toFixed(2) || '--'}</span>
                                    </div>
                                </div>
                            </div>
                        </Card>
                    </TabPanel>

                    <TabPanel value="logs" activeValue={activeTab}>
                        <Card className="logs-card">
                            <div className="logs-list">
                                {logs.map((log, i) => (
                                    <div key={i} className={`log-entry ${log.level}`}>
                                        <span className="log-time">[{log.timestamp}]</span>
                                        <span className="log-message">{log.message}</span>
                                    </div>
                                ))}
                            </div>
                        </Card>
                    </TabPanel>

                    <TabPanel value="config" activeValue={activeTab}>
                        <Card>
                            <div className="config-grid">
                                <ConfigItem label="Base Model" value={job.baseModel} />
                                <ConfigItem label="Dataset" value={job.datasetName} />
                                <ConfigItem label="Epochs" value={`${job.totalEpochs}`} />
                                <ConfigItem label="Learning Rate" value={job.learningRate.toExponential(1)} />
                                <ConfigItem label="Rank (LoRA)" value={`${job.rank}`} />
                                <ConfigItem label="Batch Size" value={`${job.batchSize}`} />
                            </div>
                        </Card>
                    </TabPanel>
                </div>

                {/* Right: Configuration Summary */}
                <div className="sidebar">
                    <Card>
                        <h3>Configuration</h3>
                        <div className="config-summary">
                            <ConfigItem label="Base Model" value={job.baseModel} />
                            <ConfigItem label="Dataset" value={job.datasetName} />
                            <ConfigItem label="Epochs" value={`${job.currentEpoch || 0}/${job.totalEpochs}`} />
                            <ConfigItem label="Learning Rate" value={job.learningRate.toExponential(1)} />
                            <ConfigItem label="Rank" value={`${job.rank}`} />
                            <ConfigItem label="Batch Size" value={`${job.batchSize}`} />
                        </div>
                    </Card>

                    {job.checkpoints.length > 0 && (
                        <Card style={{ marginTop: 'var(--space-4)' }}>
                            <h3>Checkpoints</h3>
                            <div className="checkpoints-list">
                                {job.checkpoints.map((cp, i) => (
                                    <div key={i} className="checkpoint-item">
                                        <CheckCircle size={14} />
                                        {cp}
                                    </div>
                                ))}
                            </div>
                        </Card>
                    )}
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
          color: var(--primary-400);
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
          border: 1px solid var(--primary-500);
        }

        .status-banner.completed {
          background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
          border: 1px solid var(--success-500);
        }

        .status-banner.failed {
          background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
          border: 1px solid var(--error-500);
        }

        .banner-content {
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }

        .status-banner.completed .banner-content svg {
          color: var(--success-500);
        }

        .status-banner.failed .banner-content svg {
          color: var(--error-500);
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

        .error-suggestions {
          margin-bottom: var(--space-6);
          border-color: var(--error-500);
        }

        .error-suggestions h3 {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          color: var(--error-500);
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
          color: var(--text-muted);
        }

        .suggestions-footer {
          margin-top: var(--space-4);
          padding-top: var(--space-4);
          border-top: 1px solid var(--border);
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
          color: var(--text-muted);
        }

        .loss-value {
          font-size: var(--text-2xl);
          font-weight: 700;
          color: var(--primary-400);
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
          color: var(--text-muted);
          margin-right: var(--space-2);
        }

        .log-entry.error .log-message {
          color: var(--error-500);
        }

        .log-entry.warning .log-message {
          color: var(--warning-500);
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
          color: var(--success-500);
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
            <span style={{ color: 'var(--text-muted)' }}>{label}</span>
            <span style={{ fontWeight: 500 }}>{value}</span>
        </div>
    )
}
