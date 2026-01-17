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
  Rocket
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'
import { Badge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'
import { QueryError } from '@/components/ui/QueryError'
import { formatRelativeTime } from '@/utils/formatters'
import { useJobs, useActiveModel, useSystemStatus } from '@/lib/hooks'
import type { TrainingJob } from '@/types'

export function DashboardPage() {
  const {
    data: jobs,
    isLoading: jobsLoading,
    error: jobsError,
    refetch: refetchJobs
  } = useJobs()

  const {
    data: activeModel,
    isLoading: modelLoading
  } = useActiveModel()

  const {
    data: systemStatus,
    isLoading: statusLoading
  } = useSystemStatus()

  const isLoading = jobsLoading || modelLoading || statusLoading

  // Filter active/running jobs
  const activeJobs = jobs?.filter(
    (j: TrainingJob) => j.status === 'running' || j.status === 'queued'
  ) || []

  // Mock activity until we have activity API
  const activity = [
    { id: '1', type: 'job_started', message: 'Training job started', timestamp: new Date(Date.now() - 2 * 60000).toISOString() },
    { id: '2', type: 'model_deployed', message: 'Model deployed to Ollama', timestamp: new Date(Date.now() - 60 * 60000).toISOString() },
  ]

  return (
    <div className="dashboard-page">
      {/* System Status Bar */}
      <div className="status-bar">
        {statusLoading ? (
          <Skeleton className="h-6 w-32" />
        ) : (
          <>
            <div className="status-indicator">
              <span className={`status-dot ${systemStatus?.healthy ? 'healthy' : 'error'}`} />
              <span className="status-text">
                {systemStatus?.healthy ? 'System Healthy' : 'System Error'}
              </span>
            </div>
            <div className="status-details">
              <span className="status-item">
                <Cpu size={14} />
                Ollama: {systemStatus?.ollama?.status === 'running' ? 'Connected' : 'Disconnected'}
              </span>
              <span className="status-item">
                <Activity size={14} />
                {systemStatus?.gpu?.name || 'CPU Mode'}
              </span>
              <span className="status-item">
                <Database size={14} />
                Memory: {systemStatus?.memory
                  ? `${(systemStatus.memory.used / 1024 / 1024 / 1024).toFixed(1)}GB / ${(systemStatus.memory.total / 1024 / 1024 / 1024).toFixed(0)}GB`
                  : 'N/A'}
              </span>
            </div>
          </>
        )}
        {activeModel && (
          <div className="active-model-pill">
            <Zap size={14} />
            Active: {activeModel.name}:{activeModel.version}
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

          {jobsError ? (
            <QueryError
              error={jobsError}
              onRetry={refetchJobs}
              compact
            />
          ) : jobsLoading ? (
            <div className="jobs-list">
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : activeJobs.length === 0 ? (
            <div className="empty-jobs">
              <p>No active training jobs</p>
              <Link to="/jobs/new">
                <Button size="sm" icon={<Plus size={14} />}>
                  New Fine-Tune
                </Button>
              </Link>
            </div>
          ) : (
            <div className="jobs-list">
              {activeJobs.map((job: TrainingJob) => (
                <Link to={`/jobs/${job.id}`} key={job.id} className="job-item">
                  <div className="job-info">
                    <span className="job-name">{job.projectName}</span>
                    <span className="job-epoch">
                      Epoch {job.currentEpoch || 0}/{job.epochs}
                    </span>
                  </div>
                  <div className="job-progress">
                    <Progress value={job.progress || 0} size="sm" />
                    <span className="job-eta">
                      {job.eta ? `ETA: ${job.eta}` : 'Starting...'}
                    </span>
                  </div>
                  {job.loss !== undefined && (
                    <span className="job-loss">Loss: {job.loss.toFixed(4)}</span>
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

      {/* First-time user welcome */}
      {!isLoading && jobs?.length === 0 && (
        <Card className="welcome-card">
          <div className="welcome-content">
            <h2>👋 Welcome to AI Forge!</h2>
            <p>Let&apos;s set up your first fine-tune:</p>
            <div className="welcome-steps">
              <div className="welcome-step">
                <span className="step-number">1</span>
                <span>Add your code</span>
                <Link to="/datasets">
                  <Button intent="secondary" size="sm">Add Dataset</Button>
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
          border: 1px solid var(--border-subtle);
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

        .status-dot.healthy { background: var(--status-success); }
        .status-dot.degraded { background: var(--status-warning); }
        .status-dot.error { background: var(--status-danger); }

        .status-text {
          font-weight: var(--font-semibold);
          color: var(--text-primary);
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
          background: var(--accent-primary-bg);
          border: 1px solid var(--accent-primary);
          border-radius: var(--radius-full);
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
          color: var(--accent-primary);
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
          font-weight: var(--font-semibold);
          color: var(--text-primary);
          margin: 0;
        }

        .card-header a {
          font-size: var(--text-sm);
          color: var(--accent-primary);
        }

        .active-jobs-card {
          grid-row: span 2;
        }

        .empty-jobs {
          text-align: center;
          padding: var(--space-8);
          color: var(--text-secondary);
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
          border: 1px solid var(--border-subtle);
          text-decoration: none;
          color: inherit;
          transition: border-color var(--transition-fast);
        }

        .job-item:hover {
          border-color: var(--accent-primary);
        }

        .job-info {
          display: flex;
          justify-content: space-between;
          margin-bottom: var(--space-2);
        }

        .job-name {
          font-weight: var(--font-semibold);
          color: var(--text-primary);
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
          color: var(--text-tertiary);
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
          font-weight: var(--font-semibold);
          margin-bottom: var(--space-4);
          color: var(--text-primary);
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
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-lg);
          text-decoration: none;
          color: var(--text-secondary);
          transition: all var(--transition-fast);
        }

        .action-card:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
          border-color: var(--accent-primary);
          transform: translateY(-2px);
        }

        .action-card.primary {
          background: var(--accent-primary);
          border-color: var(--accent-primary);
          color: white;
        }

        .action-card.primary:hover {
          background: var(--accent-primary-hover);
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
          color: var(--text-tertiary);
        }

        .activity-icon .success { color: var(--status-success); }
        .activity-icon .error { color: var(--status-danger); }
        .activity-icon .info { color: var(--status-info); }

        .activity-message {
          flex: 1;
          font-size: var(--text-sm);
          color: var(--text-secondary);
        }

        .activity-time {
          font-size: var(--text-xs);
          color: var(--text-tertiary);
        }

        .welcome-card {
          margin-top: var(--space-6);
          padding: var(--space-8);
          background: var(--bg-surface);
          border: 1px dashed var(--border-subtle);
        }

        .welcome-content {
          text-align: center;
        }

        .welcome-content h2 {
          font-size: var(--text-xl);
          margin-bottom: var(--space-2);
          color: var(--text-primary);
        }

        .welcome-content > p {
          color: var(--text-secondary);
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
          color: var(--text-secondary);
        }

        .step-number {
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--accent-primary);
          color: white;
          border-radius: 50%;
          font-weight: var(--font-semibold);
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
