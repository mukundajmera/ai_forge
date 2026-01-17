import { useState } from 'react'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Badge } from '@/components/ui/Badge'
import { Save, RefreshCw, CheckCircle, XCircle, Server, Database } from 'lucide-react'

export function APIConfig() {
    const [ollamaHost, setOllamaHost] = useState('http://localhost:11434')
    const [apiEndpoint, setApiEndpoint] = useState('http://localhost:8000')

    // Mock connection status
    const [ollamaStatus, setOllamaStatus] = useState<'connected' | 'disconnected' | 'checking'>('connected')
    const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected' | 'checking'>('connected')

    const testConnection = (type: 'ollama' | 'api') => {
        if (type === 'ollama') {
            setOllamaStatus('checking')
            setTimeout(() => setOllamaStatus('connected'), 1000)
        } else {
            setApiStatus('checking')
            setTimeout(() => setApiStatus('connected'), 1000)
        }
    }

    return (
        <div className="settings-section">
            <div className="section-header">
                <h2>API & Connections</h2>
                <p>Configure backend services and integrations</p>
            </div>

            <div className="settings-grid">
                {/* Ollama Connection */}
                <div className="connection-card">
                    <div className="connection-header">
                        <Server size={20} />
                        <span className="connection-title">Ollama</span>
                        <ConnectionBadge status={ollamaStatus} />
                    </div>
                    <div className="connection-body">
                        <div className="setting-row">
                            <label className="setting-label">Host URL</label>
                            <div className="input-row">
                                <Input
                                    value={ollamaHost}
                                    onChange={(e) => setOllamaHost(e.target.value)}
                                    placeholder="http://localhost:11434"
                                />
                                <Button
                                    intent="secondary"
                                    size="sm"
                                    onClick={() => testConnection('ollama')}
                                    disabled={ollamaStatus === 'checking'}
                                >
                                    <RefreshCw size={14} className={ollamaStatus === 'checking' ? 'spin' : ''} />
                                    Test
                                </Button>
                            </div>
                        </div>
                        {ollamaStatus === 'connected' && (
                            <div className="connection-info">
                                <span>Version: 0.1.42</span>
                                <span>•</span>
                                <span>Models: 3 loaded</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* API Endpoint */}
                <div className="connection-card">
                    <div className="connection-header">
                        <Database size={20} />
                        <span className="connection-title">AI Forge API</span>
                        <ConnectionBadge status={apiStatus} />
                    </div>
                    <div className="connection-body">
                        <div className="setting-row">
                            <label className="setting-label">Endpoint URL</label>
                            <div className="input-row">
                                <Input
                                    value={apiEndpoint}
                                    onChange={(e) => setApiEndpoint(e.target.value)}
                                    placeholder="http://localhost:8000"
                                />
                                <Button
                                    intent="secondary"
                                    size="sm"
                                    onClick={() => testConnection('api')}
                                    disabled={apiStatus === 'checking'}
                                >
                                    <RefreshCw size={14} className={apiStatus === 'checking' ? 'spin' : ''} />
                                    Test
                                </Button>
                            </div>
                        </div>
                        {apiStatus === 'connected' && (
                            <div className="connection-info">
                                <span>Version: 0.1.0</span>
                                <span>•</span>
                                <span>Healthy</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="section-footer">
                <Button icon={<Save size={16} />}>
                    Save Configuration
                </Button>
            </div>

            <style>{`
                .settings-section {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    padding: var(--space-6);
                }

                .section-header {
                    margin-bottom: var(--space-6);
                    padding-bottom: var(--space-4);
                    border-bottom: 1px solid var(--border-subtle);
                }

                .section-header h2 {
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    margin: 0 0 var(--space-1) 0;
                }

                .section-header p {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin: 0;
                }

                .settings-grid {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-4);
                }

                .connection-card {
                    background: var(--bg-elevated);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-md);
                    padding: var(--space-4);
                }

                .connection-header {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    margin-bottom: var(--space-4);
                    color: var(--text-secondary);
                }

                .connection-title {
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    flex: 1;
                }

                .connection-body {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-3);
                }

                .setting-row {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .setting-label {
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .input-row {
                    display: flex;
                    gap: var(--space-2);
                }

                .input-row input {
                    flex: 1;
                }

                .connection-info {
                    display: flex;
                    gap: var(--space-2);
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }

                .section-footer {
                    display: flex;
                    justify-content: flex-end;
                    gap: var(--space-2);
                    margin-top: var(--space-6);
                    padding-top: var(--space-4);
                    border-top: 1px solid var(--border-subtle);
                }

                .spin {
                    animation: spin 1s linear infinite;
                }
            `}</style>
        </div>
    )
}

function ConnectionBadge({ status }: { status: 'connected' | 'disconnected' | 'checking' }) {
    if (status === 'checking') {
        return <Badge variant="info">Checking...</Badge>
    }
    if (status === 'connected') {
        return (
            <Badge variant="success">
                <CheckCircle size={12} />
                Connected
            </Badge>
        )
    }
    return (
        <Badge variant="danger">
            <XCircle size={12} />
            Disconnected
        </Badge>
    )
}
