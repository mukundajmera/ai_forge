// =============================================================================
// Logs Panel - Real-time training log viewer with auto-scroll and search
// =============================================================================

import { useRef, useEffect, useState, useMemo } from 'react';
import { Download, Pause, Play, Trash2, Search } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useLogStream, LogEntry } from '@/lib/hooks/useLogStream';

interface LogsPanelProps {
    jobId: string;
    height?: number;
}

const LOG_LEVEL_COLORS: Record<LogEntry['level'], string> = {
    info: 'log-level-info',
    warning: 'log-level-warning',
    error: 'log-level-error',
    debug: 'log-level-debug',
};

export function LogsPanel({ jobId, height = 384 }: LogsPanelProps) {
    const { logs, isConnected, clearLogs, downloadLogs } = useLogStream(jobId);
    const [autoScroll, setAutoScroll] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const logsEndRef = useRef<HTMLDivElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom when new logs arrive
    useEffect(() => {
        if (autoScroll && logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs, autoScroll]);

    // Detect manual scroll to pause auto-scroll
    const handleScroll = () => {
        if (!containerRef.current) return;

        const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
        const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;

        if (!isAtBottom && autoScroll) {
            setAutoScroll(false);
        }
    };

    // Filter logs by search query
    const filteredLogs = useMemo(() => {
        if (!searchQuery.trim()) return logs;

        const query = searchQuery.toLowerCase();
        return logs.filter(
            (log) =>
                log.message.toLowerCase().includes(query) ||
                log.level.toLowerCase().includes(query)
        );
    }, [logs, searchQuery]);

    // Resume auto-scroll
    const resumeAutoScroll = () => {
        setAutoScroll(true);
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    return (
        <div className="logs-panel" style={{ height }}>
            {/* Toolbar */}
            <div className="logs-toolbar">
                <div className="logs-toolbar-left">
                    <div className="logs-status">
                        <div
                            className={`status-dot ${isConnected ? 'status-connected' : 'status-disconnected'}`}
                        />
                        <span>{isConnected ? 'Live' : 'Disconnected'}</span>
                    </div>
                    <span className="logs-count">{filteredLogs.length} lines</span>
                </div>

                <div className="logs-toolbar-right">
                    {/* Search */}
                    <div className="logs-search">
                        <Search className="search-icon" size={14} />
                        <input
                            type="text"
                            placeholder="Search logs..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="search-input"
                        />
                    </div>

                    {/* Auto-scroll toggle */}
                    <Button
                        intent="ghost"
                        size="sm"
                        icon={autoScroll ? <Pause size={14} /> : <Play size={14} />}
                        onClick={() => autoScroll ? setAutoScroll(false) : resumeAutoScroll()}
                    >
                        {autoScroll ? 'Pause' : 'Resume'}
                    </Button>

                    {/* Clear logs */}
                    <Button
                        intent="ghost"
                        size="sm"
                        icon={<Trash2 size={14} />}
                        onClick={clearLogs}
                    >
                        Clear
                    </Button>

                    {/* Download logs */}
                    <Button
                        intent="ghost"
                        size="sm"
                        icon={<Download size={14} />}
                        onClick={downloadLogs}
                        disabled={logs.length === 0}
                    >
                        Download
                    </Button>
                </div>
            </div>

            {/* Logs container */}
            <div
                ref={containerRef}
                onScroll={handleScroll}
                className="logs-container"
            >
                {filteredLogs.length === 0 ? (
                    <div className="logs-empty">
                        {searchQuery ? 'No matching logs' : 'No logs yet'}
                    </div>
                ) : (
                    <div className="logs-list">
                        {filteredLogs.map((log, idx) => (
                            <div key={idx} className="log-entry">
                                <span className="log-timestamp">
                                    [{new Date(log.timestamp).toLocaleTimeString()}]
                                </span>
                                <span className={`log-level ${LOG_LEVEL_COLORS[log.level]}`}>
                                    [{log.level.toUpperCase()}]
                                </span>
                                <span className="log-message">{log.message}</span>
                            </div>
                        ))}
                        <div ref={logsEndRef} />
                    </div>
                )}
            </div>

            <style>{`
                .logs-panel {
                    display: flex;
                    flex-direction: column;
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    overflow: hidden;
                }

                .logs-toolbar {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: var(--space-2) var(--space-4);
                    border-bottom: 1px solid var(--border-subtle);
                    background: var(--bg-elevated);
                    gap: var(--space-2);
                    flex-wrap: wrap;
                }

                .logs-toolbar-left {
                    display: flex;
                    align-items: center;
                    gap: var(--space-3);
                }

                .logs-toolbar-right {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                }

                .logs-status {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .status-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                }

                .status-dot.status-connected {
                    background: var(--status-success);
                    animation: pulse 2s infinite;
                }

                .status-dot.status-disconnected {
                    background: var(--status-danger);
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }

                .logs-count {
                    font-size: var(--text-sm);
                    color: var(--text-tertiary);
                }

                .logs-search {
                    position: relative;
                }

                .search-icon {
                    position: absolute;
                    left: var(--space-3);
                    top: 50%;
                    transform: translateY(-50%);
                    color: var(--text-tertiary);
                }

                .search-input {
                    width: 180px;
                    padding: var(--space-2) var(--space-3);
                    padding-left: var(--space-8);
                    background: var(--bg-surface);
                    border: 1px solid var(--border-default);
                    border-radius: var(--radius-md);
                    font-size: var(--text-sm);
                    color: var(--text-primary);
                }

                .search-input:focus {
                    outline: none;
                    border-color: var(--accent-primary);
                }

                .logs-container {
                    flex: 1;
                    overflow-y: auto;
                    padding: var(--space-3);
                    font-family: var(--font-mono);
                    font-size: var(--text-xs);
                    line-height: 1.5;
                }

                .logs-empty {
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: var(--text-secondary);
                }

                .logs-list {
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                }

                .log-entry {
                    display: flex;
                    gap: var(--space-2);
                    white-space: pre-wrap;
                    word-break: break-word;
                }

                .log-timestamp {
                    color: var(--text-tertiary);
                    flex-shrink: 0;
                }

                .log-level {
                    font-weight: var(--font-semibold);
                    flex-shrink: 0;
                }

                .log-level-info {
                    color: var(--text-secondary);
                }

                .log-level-warning {
                    color: var(--status-warning);
                }

                .log-level-error {
                    color: var(--status-danger);
                }

                .log-level-debug {
                    color: var(--accent-primary);
                }

                .log-message {
                    color: var(--text-primary);
                    flex: 1;
                }
            `}</style>
        </div>
    );
}
