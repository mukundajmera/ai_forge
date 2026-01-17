import { mockSystemHealth } from '@/utils/mock-data'
import { Cpu, HardDrive, Thermometer, Server, CheckCircle } from 'lucide-react'
import { formatBytes } from '@/utils/formatters'

export function SystemHealthPanel() {
    const { gpu, cpu, memory, ollama } = mockSystemHealth

    return (
        <div className="health-grid">
            {/* GPU Card */}
            <div className="health-card">
                <div className="card-header">
                    <HardDrive size={20} />
                    <span>GPU</span>
                </div>
                <div className="card-content">
                    <div className="metric-large">{gpu.name}</div>
                    <div className="metric-row">
                        <span>Memory</span>
                        <span>{formatBytes(gpu.memoryUsed)} / {formatBytes(gpu.memoryTotal)}</span>
                    </div>
                    <div className="progress-bar">
                        <div
                            className="progress-fill"
                            style={{
                                width: `${(gpu.memoryUsed / gpu.memoryTotal) * 100}%`,
                                background: gpu.memoryUsed / gpu.memoryTotal > 0.8
                                    ? 'var(--status-warning)'
                                    : 'var(--accent-primary)'
                            }}
                        />
                    </div>
                    <div className="metric-row">
                        <span>Utilization</span>
                        <span>{gpu.utilization}%</span>
                    </div>
                    <div className="metric-row">
                        <Thermometer size={14} />
                        <span style={{ marginLeft: 4 }}>{gpu.temperature}°C</span>
                    </div>
                </div>
            </div>

            {/* CPU Card */}
            <div className="health-card">
                <div className="card-header">
                    <Cpu size={20} />
                    <span>CPU</span>
                </div>
                <div className="card-content">
                    <div className="metric-large">{cpu.cores} Cores</div>
                    <div className="metric-row">
                        <span>Utilization</span>
                        <span>{cpu.utilization}%</span>
                    </div>
                    <div className="progress-bar">
                        <div
                            className="progress-fill"
                            style={{ width: `${cpu.utilization}%` }}
                        />
                    </div>
                </div>
            </div>

            {/* Memory Card */}
            <div className="health-card">
                <div className="card-header">
                    <HardDrive size={20} />
                    <span>Memory</span>
                </div>
                <div className="card-content">
                    <div className="metric-large">
                        {formatBytes(memory.used)} / {formatBytes(memory.total)}
                    </div>
                    <div className="progress-bar">
                        <div
                            className="progress-fill"
                            style={{
                                width: `${(memory.used / memory.total) * 100}%`,
                                background: memory.used / memory.total > 0.8
                                    ? 'var(--status-warning)'
                                    : 'var(--accent-primary)'
                            }}
                        />
                    </div>
                    <div className="metric-row">
                        <span>Available</span>
                        <span>{formatBytes(memory.total - memory.used)}</span>
                    </div>
                </div>
            </div>

            {/* Ollama Status Card */}
            <div className="health-card">
                <div className="card-header">
                    <Server size={20} />
                    <span>Ollama</span>
                </div>
                <div className="card-content">
                    <div className="status-indicator">
                        <span
                            className="status-dot"
                            style={{
                                background: ollama.status === 'running'
                                    ? 'var(--status-success)'
                                    : 'var(--status-danger)'
                            }}
                        />
                        <span style={{
                            color: ollama.status === 'running'
                                ? 'var(--status-success)'
                                : 'var(--status-danger)',
                            textTransform: 'capitalize',
                            fontWeight: 'var(--font-semibold)',
                        }}>
                            {ollama.status}
                        </span>
                    </div>
                    {ollama.modelsLoaded.length > 0 && (
                        <div className="models-loaded">
                            <span className="models-label">Loaded Models:</span>
                            {ollama.modelsLoaded.map((model, i) => (
                                <span key={i} className="model-tag">
                                    <CheckCircle size={12} />
                                    {model}
                                </span>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <style>{`
                .health-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: var(--space-4);
                }

                .health-card {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-4);
                }

                .card-header {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-secondary);
                    margin-bottom: var(--space-4);
                }

                .card-content {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-3);
                }

                .metric-large {
                    font-size: var(--text-lg);
                    font-weight: var(--font-bold);
                    color: var(--text-primary);
                }

                .metric-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .progress-bar {
                    height: 6px;
                    background: var(--bg-elevated);
                    border-radius: var(--radius-full);
                    overflow: hidden;
                }

                .progress-fill {
                    height: 100%;
                    background: var(--accent-primary);
                    border-radius: var(--radius-full);
                    transition: width var(--transition-slow);
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
                    animation: pulse 2s infinite;
                }

                .models-loaded {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .models-label {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    text-transform: uppercase;
                }

                .model-tag {
                    display: inline-flex;
                    align-items: center;
                    gap: var(--space-1);
                    padding: var(--space-1) var(--space-2);
                    background: var(--status-success-bg);
                    color: var(--status-success);
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    border-radius: var(--radius-md);
                    width: fit-content;
                }
            `}</style>
        </div>
    )
}
