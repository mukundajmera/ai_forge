import { Bell, HelpCircle, Search } from 'lucide-react'
import { mockModels } from '@/utils/mock-data'

export function TopBar() {
    const activeModel = mockModels.find(m => m.isActive)

    return (
        <header className="top-bar">
            <div className="top-bar-left">
                {/* Active model indicator */}
                <div className="active-model-chip">
                    <div className="pulse-dot" />
                    <span className="model-label">Active:</span>
                    <span className="model-name">
                        {activeModel ? `${activeModel.name}:${activeModel.version}` : 'No model'}
                    </span>
                </div>

                {/* Quick stats */}
                <div className="quick-stats">
                    <div className="stat">
                        <div className="stat-dot running" />
                        <span>2 running</span>
                    </div>
                    <div className="stat-divider" />
                    <div className="stat">
                        <div className="stat-dot ready" />
                        <span>5 datasets</span>
                    </div>
                </div>
            </div>

            <div className="top-bar-right">
                <button className="icon-button">
                    <Search size={20} />
                </button>
                <button className="icon-button notification">
                    <Bell size={20} />
                    <span className="notification-dot" />
                </button>
                <button className="icon-button">
                    <HelpCircle size={20} />
                </button>
            </div>

            <style>{`
                .top-bar {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    height: 56px;
                    padding: 0 var(--space-6);
                    background: var(--bg-surface);
                    border-bottom: 1px solid var(--border-subtle);
                }

                .top-bar-left {
                    display: flex;
                    align-items: center;
                    gap: var(--space-6);
                }

                .active-model-chip {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    padding: var(--space-2) var(--space-3);
                    background: var(--bg-elevated);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-full);
                }

                .pulse-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: var(--status-success);
                    animation: pulse 2s infinite;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }

                .model-label {
                    font-size: var(--text-xs);
                    color: var(--text-secondary);
                }

                .model-name {
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                }

                .quick-stats {
                    display: flex;
                    align-items: center;
                    gap: var(--space-4);
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                }

                .stat {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                }

                .stat-dot {
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                }

                .stat-dot.running {
                    background: var(--status-info);
                }

                .stat-dot.ready {
                    background: var(--status-success);
                }

                .stat-divider {
                    width: 1px;
                    height: 16px;
                    background: var(--border-subtle);
                }

                .top-bar-right {
                    display: flex;
                    align-items: center;
                    gap: var(--space-1);
                }

                .icon-button {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 40px;
                    height: 40px;
                    border: none;
                    background: none;
                    color: var(--text-secondary);
                    border-radius: var(--radius-md);
                    cursor: pointer;
                    transition: all var(--transition-fast);
                    position: relative;
                }

                .icon-button:hover {
                    background: var(--bg-hover);
                    color: var(--text-primary);
                }

                .notification {
                    position: relative;
                }

                .notification-dot {
                    position: absolute;
                    top: 8px;
                    right: 8px;
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: var(--status-danger);
                }

                @media (max-width: 768px) {
                    .quick-stats {
                        display: none;
                    }
                }
            `}</style>
        </header>
    )
}
