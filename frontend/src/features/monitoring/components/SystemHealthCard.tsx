// =============================================================================
// System Health Card - Real-time CPU, RAM, GPU, Disk monitoring
// =============================================================================

import { Cpu, Zap, AlertTriangle } from 'lucide-react';
import { useSystemStatus } from '@/lib/hooks';

function formatBytes(bytes: number): string {
    const gb = bytes / (1024 ** 3);
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    const mb = bytes / (1024 ** 2);
    return `${mb.toFixed(0)} MB`;
}

function getUsageColor(percent: number): string {
    if (percent < 60) return 'usage-good';
    if (percent < 80) return 'usage-warning';
    return 'usage-critical';
}

interface ResourceItemProps {
    icon: React.ReactNode;
    label: string;
    usage: number;
    detail: string;
}

function ResourceItem({ icon, label, usage, detail }: ResourceItemProps) {
    const colorClass = getUsageColor(usage);

    return (
        <div className="resource-item">
            <div className="resource-header">
                <div className="resource-info">
                    {icon}
                    <span className="resource-label">{label}</span>
                </div>
                <span className={`resource-usage ${colorClass}`}>
                    {usage.toFixed(1)}%
                </span>
            </div>

            <div className="resource-bar">
                <div
                    className={`resource-fill ${colorClass}`}
                    style={{ width: `${Math.min(usage, 100)}%` }}
                />
            </div>

            <div className="resource-detail">{detail}</div>
        </div>
    );
}

// Custom memory icon since MemoryStick isn't directly available
function MemoryIcon({ size }: { size: number }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <line x1="7" y1="8" x2="7" y2="16" />
            <line x1="11" y1="8" x2="11" y2="16" />
            <line x1="15" y1="8" x2="15" y2="16" />
        </svg>
    );
}

export function SystemHealthCard() {
    const { data: status, isLoading } = useSystemStatus();

    if (isLoading) {
        return (
            <div className="system-health-card">
                <h3 className="card-title">System Resources</h3>
                <div className="loading-state">Loading system status...</div>

                <style>{`
                    .loading-state {
                        padding: var(--space-8);
                        text-align: center;
                        color: var(--text-secondary);
                    }
                `}</style>
            </div>
        );
    }

    if (!status) {
        return (
            <div className="system-health-card">
                <h3 className="card-title">System Resources</h3>
                <div className="error-state">Unable to fetch system status</div>

                <style>{`
                    .error-state {
                        padding: var(--space-8);
                        text-align: center;
                        color: var(--status-danger);
                    }
                `}</style>
            </div>
        );
    }

    const resources = [
        {
            icon: <Cpu size={16} />,
            label: 'CPU',
            usage: status.cpu.utilization,
            detail: `${status.cpu.cores} cores`,
        },
        {
            icon: <MemoryIcon size={16} />,
            label: 'Memory',
            usage: (status.memory.used / status.memory.total) * 100,
            detail: `${formatBytes(status.memory.used)} / ${formatBytes(status.memory.total)}`,
        },
        ...(status.gpu
            ? [
                {
                    icon: <Zap size={16} />,
                    label: 'GPU',
                    usage: status.gpu.utilization,
                    detail: status.gpu.name,
                },
            ]
            : []),
    ];

    const memoryUsagePercent = (status.memory.used / status.memory.total) * 100;
    const showMemoryWarning = memoryUsagePercent > 85;
    const showGpuWarning = status.gpu && status.gpu.utilization > 90;

    return (
        <div className="system-health-card">
            <h3 className="card-title">System Resources</h3>

            <div className="resources-grid">
                {resources.map((resource) => (
                    <ResourceItem key={resource.label} {...resource} />
                ))}
            </div>

            {/* Active jobs indicator */}
            <div className="jobs-indicator">
                <span className="jobs-label">Running Jobs</span>
                <span className="jobs-count">
                    {status.runningJobs}
                </span>
            </div>

            {/* Warnings */}
            {(showMemoryWarning || showGpuWarning) && (
                <div className="resource-warning">
                    <AlertTriangle size={16} />
                    <p>
                        {showMemoryWarning &&
                            'Memory usage is high. Consider waiting for current jobs to finish.'}
                        {showGpuWarning && 'GPU is heavily utilized.'}
                    </p>
                </div>
            )}

            <style>{`
                .system-health-card {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-5);
                }

                .card-title {
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-4) 0;
                }

                .resources-grid {
                    display: grid;
                    gap: var(--space-4);
                    grid-template-columns: repeat(2, 1fr);
                }

                .resource-item {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .resource-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .resource-info {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    color: var(--text-secondary);
                }

                .resource-label {
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .resource-usage {
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                }

                .resource-usage.usage-good {
                    color: var(--status-success);
                }

                .resource-usage.usage-warning {
                    color: var(--status-warning);
                }

                .resource-usage.usage-critical {
                    color: var(--status-danger);
                }

                .resource-bar {
                    height: 6px;
                    background: var(--bg-elevated);
                    border-radius: var(--radius-full);
                    overflow: hidden;
                }

                .resource-fill {
                    height: 100%;
                    border-radius: var(--radius-full);
                    transition: width 0.3s ease;
                }

                .resource-fill.usage-good {
                    background: var(--status-success);
                }

                .resource-fill.usage-warning {
                    background: var(--status-warning);
                }

                .resource-fill.usage-critical {
                    background: var(--status-danger);
                }

                .resource-detail {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }

                .jobs-indicator {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-top: var(--space-4);
                    padding: var(--space-3);
                    background: var(--bg-elevated);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-md);
                }

                .jobs-label {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .jobs-count {
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                }

                .resource-warning {
                    display: flex;
                    align-items: flex-start;
                    gap: var(--space-2);
                    margin-top: var(--space-4);
                    padding: var(--space-3);
                    background: rgba(251, 191, 36, 0.1);
                    border: 1px solid rgba(251, 191, 36, 0.3);
                    border-radius: var(--radius-md);
                    color: var(--status-warning);
                }

                .resource-warning p {
                    margin: 0;
                    font-size: var(--text-sm);
                }

                @media (max-width: 480px) {
                    .resources-grid {
                        grid-template-columns: 1fr;
                    }
                }
            `}</style>
        </div>
    );
}
