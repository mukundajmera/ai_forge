import {
    Activity,
    Cpu,
    Database,
    Zap,
    Plus,
    FolderOpen,
    Boxes,
    Clock,
    CheckCircle,
    XCircle,
    AlertTriangle,
    Rocket
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'
import { Badge } from '@/components/ui/Badge'
import { formatRelativeTime } from '@/utils/formatters'
import type { SystemHealth, ActiveJob, RecentActivity } from './hooks/useDashboard'

// Mock data for initial implementation
const mockHealth: SystemHealth = {
    status: 'healthy',
    ollama: { connected: true, version: '0.1.25', models: 3 },
    system: { memory_used: 24.3, memory_total: 32, gpu: 'Apple M2 Max' },
    activeModel: { name: 'myproject', version: 'v2' },
}

const mockActiveJobs: ActiveJob[] = [
    {
        id: 'job-1',
        project: 'my-project',
        status: 'training',
        progress: 78,
        currentEpoch: 2,
        totalEpochs: 3,
        loss: 1.23,
        eta: '12 min',
        startedAt: new Date(Date.now() - 10 * 60000).toISOString(),
    },
    {
        id: 'job-2',
        project: 'api-helper',
        status: 'training',
        progress: 15,
        currentEpoch: 1,
        totalEpochs: 3,
        loss: 2.45,
        eta: '45 min',
        startedAt: new Date(Date.now() - 5 * 60000).toISOString(),
    },
]

const mockActivity: RecentActivity[] = [
    { id: '1', type: 'job_started', message: 'Job "my-project" started', timestamp: new Date(Date.now() - 2 * 60000).toISOString() },
    { id: '2', type: 'model_deployed', message: 'Model v2 deployed to Ollama', timestamp: new Date(Date.now() - 60 * 60000).toISOString() },
    { id: '3', type: 'dataset_created', message: 'Dataset "codebase" parsed successfully', timestamp: new Date(Date.now() - 3 * 60 * 60000).toISOString() },
    { id: '4', type: 'job_completed', message: 'Job "api-helper-v1" completed', timestamp: new Date(Date.now() - 5 * 60 * 60000).toISOString() },
]

export function DashboardPage() {
    // In production, use: const { data, isLoading } = useDashboard()
    const health = mockHealth
    const activeJobs = mockActiveJobs
    const activity = mockActivity

    return (
        <div className="dashboard-page">
            {/* System Status Bar */}
            <div className="status-bar">
                <div className="status-indicator">
                    <span className={`status-dot ${health.status}`} />
                    <span className="status-text">
                        {health.status === 'healthy' ? 'System Healthy' :
                            health.status === 'degraded' ? 'System Degraded' : 'System Error'}
                    </span>
                </div>
                <div className="status-details">
                    <span className="status-item">
                        <Cpu size={14} />
                        Ollama: {health.ollama.connected ? 'Connected' : 'Disconnected'}
                    </span>
                    <span className="status-item">
                        <Activity size={14} />
                        {health.system.gpu || 'CPU'}
                    </span>
                    <span className="status-item">
                        <Database size={14} />
                        Memory: {health.system.memory_used.toFixed(1)}GB / {health.system.memory_total}GB
                    </span>
                </div>
                {health.activeModel && (
                    <div className="active-model-pill">
                        <Zap size={14} />
                        Active: {health.activeModel.name}:{health.activeModel.version}
                    </div>
                )}
            </div>

            {/* Main Content Grid */}
            <div className="dashboard-grid">
                {/* Active Jobs Widget */}
                <Card className="active-jobs-card">
                    <div className="card-header">
                        <h2>
                            <Activity size={18} />
                            Active Jobs
                            {activeJobs.length > 0 && (
                                <Badge variant="info">{activeJobs.length}</Badge>
                            )}
                        </h2>
                        <Link to="/jobs">View All</Link>
                    </div>

                    {activeJobs.length === 0 ? (
                        <div className="empty-jobs">
                            <p>No active training jobs</p>
                            <Link to="/jobs/new">
                                <Button size="sm" leftIcon={<Plus size={14} />}>
                                    New Fine-Tune
                                </Button>
                            </Link>
                        </div>
                    ) : (
                        <div className="jobs-list">
                            {activeJobs.map(job => (
                                <Link to={`/jobs/${job.id}`} key={job.id} className="job-item">
                                    <div className="job-info">
                                        <span className="job-name">{job.project}</span>
                                        <span className="job-epoch">
                                            Epoch {job.currentEpoch}/{job.totalEpochs}
                                        </span>
                                    </div>
                                    <div className="job-progress">
                                        <Progress value={job.progress} size="sm" />
                                        <span className="job-eta">ETA: {job.eta}</span>
                                    </div>
                                    {job.loss && (
                                        <span className="job-loss">Loss: {job.loss.toFixed(2)}</span>
                                    )}
                                </Link>
                            ))}
                        </div>
                    )}
                </Card>

                {/* Quick Actions */}
                <div className="quick-actions">
                    <h2>Quick Actions</h2>
                    <div className="actions-grid">
                        <Link to="/jobs/new" className="action-card primary">
                            <Plus size={24} />
                            <span>New Fine-Tune</span>
                        </Link>
                        <Link to="/datasets" className="action-card">
                            <FolderOpen size={24} />
                            <span>View Datasets</span>
                        </Link>
                        <Link to="/models" className="action-card">
                            <Boxes size={24} />
                            <span>Models</span>
                        </Link>
                        <Link to="/jobs" className="action-card">
                            <Activity size={24} />
                            <span>All Jobs</span>
                        </Link>
                    </div>
                </div>

                {/* Recent Activity */}
                <Card className="activity-card">
                    <div className="card-header">
                        <h2>
                            <Clock size={18} />
                            Recent Activity
                        </h2>
                    </div>
                    <div className="activity-list">
                        {activity.map(item => (
                            <div key={item.id} className="activity-item">
                                <span className="activity-icon">
                                    {item.type === 'job_started' && <Activity size={14} />}
                                    {item.type === 'job_completed' && <CheckCircle size={14} className="success" />}
                                    {item.type === 'job_failed' && <XCircle size={14} className="error" />}
                                    {item.type === 'model_deployed' && <Rocket size={14} className="info" />}
                                    {item.type === 'dataset_created' && <Database size={14} />}
                                </span>
                                <span className="activity-message">{item.message}</span>
                                <span className="activity-time">{formatRelativeTime(item.timestamp)}</span>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>

            {/* First-time user welcome (show if no jobs exist) */}
            {activeJobs.length === 0 && activity.length === 0 && (
                <Card className="welcome-card">
                    <div className="welcome-content">
                        <h2>👋 Welcome to AI Forge!</h2>
                        <p>Let's set up your first fine-tune:</p>
                        <div className="welcome-steps">
                            <div className="welcome-step">
                                <span className="step-number">1</span>
                                <span>Add your code</span>
                                <Link to="/datasets">
                                    <Button variant="secondary" size="sm">Add Dataset</Button>
                                </Link>
                            </div>
                            <div className="welcome-step">
                                <span className="step-number">2</span>
                                <span>Generate training data</span>
                            </div>
                            <div className="welcome-step">
                                <span className="step-number">3</span>
                                <span>Start fine-tuning</span>
                            </div>
                        </div>
                        <p className="welcome-alt">Or: <Link to="/datasets/sample">Try with sample project</Link></p>
                    </div>
                </Card>
            )}

            <style>{`
        .dashboard-page {
          padding: var(--space-6);
          max-width: 1400px;
        }

        .status-bar {
          display: flex;
          align-items: center;
          gap: var(--space-6);
          padding: var(--space-4);
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          margin-bottom: var(--space-6);
        }

        .status-indicator {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .status-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }

        .status-dot.healthy { background: var(--success-500); }
        .status-dot.degraded { background: var(--warning-500); }
        .status-dot.error { background: var(--error-500); }

        .status-text {
          font-weight: 600;
        }

        .status-details {
          display: flex;
          gap: var(--space-4);
          flex: 1;
        }

        .status-item {
          display: flex;
          align-items: center;
          gap: var(--space-1);
          font-size: var(--text-sm);
          color: var(--text-secondary);
        }

        .active-model-pill {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-2) var(--space-3);
          background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(99, 102, 241, 0.1));
          border: 1px solid var(--primary-500);
          border-radius: var(--radius-full);
          font-size: var(--text-sm);
          font-weight: 500;
          color: var(--primary-400);
        }

        .dashboard-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-6);
        }

        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-4);
        }

        .card-header h2 {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          font-size: var(--text-base);
          font-weight: 600;
        }

        .card-header a {
          font-size: var(--text-sm);
          color: var(--primary-400);
        }

        .active-jobs-card {
          grid-row: span 2;
        }

        .empty-jobs {
          text-align: center;
          padding: var(--space-8);
          color: var(--text-muted);
        }

        .empty-jobs p {
          margin-bottom: var(--space-4);
        }

        .jobs-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .job-item {
          display: block;
          padding: var(--space-4);
          background: var(--bg-elevated);
          border-radius: var(--radius-md);
          border: 1px solid var(--border);
          text-decoration: none;
          color: inherit;
          transition: border-color var(--transition-fast);
        }

        .job-item:hover {
          border-color: var(--primary-500);
        }

        .job-info {
          display: flex;
          justify-content: space-between;
          margin-bottom: var(--space-2);
        }

        .job-name {
          font-weight: 600;
        }

        .job-epoch {
          font-size: var(--text-sm);
          color: var(--text-secondary);
        }

        .job-progress {
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }

        .job-eta {
          font-size: var(--text-xs);
          color: var(--text-muted);
          white-space: nowrap;
        }

        .job-loss {
          display: block;
          margin-top: var(--space-2);
          font-size: var(--text-sm);
          color: var(--text-secondary);
        }

        .quick-actions h2 {
          font-size: var(--text-base);
          font-weight: 600;
          margin-bottom: var(--space-4);
        }

        .actions-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: var(--space-3);
        }

        .action-card {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: var(--space-2);
          padding: var(--space-6);
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          text-decoration: none;
          color: var(--text-secondary);
          transition: all var(--transition-fast);
        }

        .action-card:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
          border-color: var(--primary-500);
          transform: translateY(-2px);
        }

        .action-card.primary {
          background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
          border-color: var(--primary-500);
          color: white;
        }

        .action-card.primary:hover {
          background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
        }

        .activity-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .activity-item {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3);
          border-radius: var(--radius-md);
        }

        .activity-item:hover {
          background: var(--bg-elevated);
        }

        .activity-icon {
          color: var(--text-muted);
        }

        .activity-icon .success { color: var(--success-500); }
        .activity-icon .error { color: var(--error-500); }
        .activity-icon .info { color: var(--info-500); }

        .activity-message {
          flex: 1;
          font-size: var(--text-sm);
        }

        .activity-time {
          font-size: var(--text-xs);
          color: var(--text-muted);
        }

        .welcome-card {
          margin-top: var(--space-6);
          padding: var(--space-8);
          background: linear-gradient(135deg, var(--bg-surface), var(--bg-elevated));
          border: 1px dashed var(--border);
        }

        .welcome-content {
          text-align: center;
        }

        .welcome-content h2 {
          font-size: var(--text-xl);
          margin-bottom: var(--space-2);
        }

        .welcome-steps {
          display: flex;
          justify-content: center;
          gap: var(--space-8);
          margin: var(--space-6) 0;
        }

        .welcome-step {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: var(--space-2);
        }

        .step-number {
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--primary-500);
          color: white;
          border-radius: 50%;
          font-weight: 600;
        }

        .welcome-alt {
          color: var(--text-muted);
          font-size: var(--text-sm);
        }

        @media (max-width: 900px) {
          .dashboard-grid {
            grid-template-columns: 1fr;
          }
          
          .status-bar {
            flex-wrap: wrap;
          }
          
          .status-details {
            order: 3;
            width: 100%;
            justify-content: center;
            margin-top: var(--space-3);
          }
        }
      `}</style>
        </div>
    )
}
