import { useState } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Settings, Cpu, Link } from 'lucide-react'
import { GeneralSettings } from './components/GeneralSettings'
import { ModelDefaults } from './components/ModelDefaults'
import { APIConfig } from './components/APIConfig'

type SettingsTab = 'general' | 'models' | 'api'

const tabs: { id: SettingsTab; label: string; icon: typeof Settings }[] = [
    { id: 'general', label: 'General', icon: Settings },
    { id: 'models', label: 'Model Defaults', icon: Cpu },
    { id: 'api', label: 'API & Connections', icon: Link },
]

export function SettingsPage() {
    const [activeTab, setActiveTab] = useState<SettingsTab>('general')

    return (
        <div className="settings-page">
            <PageHeader
                title="Settings"
                subtitle="Manage your AI Forge configuration"
            />

            <div className="settings-layout">
                {/* Tabs */}
                <nav className="settings-nav">
                    {tabs.map(tab => {
                        const Icon = tab.icon
                        return (
                            <button
                                key={tab.id}
                                className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                                onClick={() => setActiveTab(tab.id)}
                            >
                                <Icon size={18} />
                                <span>{tab.label}</span>
                            </button>
                        )
                    })}
                </nav>

                {/* Content */}
                <div className="settings-content">
                    {activeTab === 'general' && <GeneralSettings />}
                    {activeTab === 'models' && <ModelDefaults />}
                    {activeTab === 'api' && <APIConfig />}
                </div>
            </div>

            <style>{`
                .settings-page {
                    padding: var(--space-8);
                    max-width: 1200px;
                }

                .settings-layout {
                    display: flex;
                    gap: var(--space-6);
                }

                .settings-nav {
                    width: 220px;
                    flex-shrink: 0;
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-1);
                }

                .nav-tab {
                    display: flex;
                    align-items: center;
                    gap: var(--space-3);
                    padding: var(--space-3) var(--space-4);
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-secondary);
                    background: none;
                    border: none;
                    border-radius: var(--radius-md);
                    cursor: pointer;
                    transition: all var(--transition-fast);
                    text-align: left;
                    width: 100%;
                }

                .nav-tab:hover {
                    background: var(--bg-hover);
                    color: var(--text-primary);
                }

                .nav-tab.active {
                    background: var(--accent-primary-bg);
                    color: var(--accent-primary);
                }

                .settings-content {
                    flex: 1;
                    min-width: 0;
                }

                @media (max-width: 768px) {
                    .settings-layout {
                        flex-direction: column;
                    }

                    .settings-nav {
                        width: 100%;
                        flex-direction: row;
                        overflow-x: auto;
                    }

                    .nav-tab {
                        white-space: nowrap;
                    }
                }
            `}</style>
        </div>
    )
}
