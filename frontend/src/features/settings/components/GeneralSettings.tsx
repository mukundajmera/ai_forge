import { useState } from 'react'
import { Button } from '@/components/ui/Button'
import { Select } from '@/components/ui/Select'
import { Moon, Bell, Clock, Save } from 'lucide-react'

export function GeneralSettings() {
    const [theme, setTheme] = useState('dark')
    const [notifications, setNotifications] = useState('all')
    const [timezone, setTimezone] = useState('auto')

    return (
        <div className="settings-section">
            <div className="section-header">
                <h2>General Settings</h2>
                <p>Manage appearance and notification preferences</p>
            </div>

            <div className="settings-grid">
                {/* Theme */}
                <div className="setting-item">
                    <div className="setting-info">
                        <Moon size={18} />
                        <div>
                            <label className="setting-label">Theme</label>
                            <p className="setting-description">Choose your preferred color scheme</p>
                        </div>
                    </div>
                    <Select
                        value={theme}
                        onChange={(e) => setTheme(e.target.value)}
                        options={[
                            { value: 'dark', label: 'Dark' },
                            { value: 'light', label: 'Light' },
                            { value: 'system', label: 'System' },
                        ]}
                        size="sm"
                    />
                </div>

                {/* Notifications */}
                <div className="setting-item">
                    <div className="setting-info">
                        <Bell size={18} />
                        <div>
                            <label className="setting-label">Notifications</label>
                            <p className="setting-description">Control when you receive notifications</p>
                        </div>
                    </div>
                    <Select
                        value={notifications}
                        onChange={(e) => setNotifications(e.target.value)}
                        options={[
                            { value: 'all', label: 'All notifications' },
                            { value: 'important', label: 'Important only' },
                            { value: 'none', label: 'None' },
                        ]}
                        size="sm"
                    />
                </div>

                {/* Timezone */}
                <div className="setting-item">
                    <div className="setting-info">
                        <Clock size={18} />
                        <div>
                            <label className="setting-label">Timezone</label>
                            <p className="setting-description">Set your preferred timezone for timestamps</p>
                        </div>
                    </div>
                    <Select
                        value={timezone}
                        onChange={(e) => setTimezone(e.target.value)}
                        options={[
                            { value: 'auto', label: 'Auto-detect' },
                            { value: 'utc', label: 'UTC' },
                            { value: 'local', label: 'Local time' },
                        ]}
                        size="sm"
                    />
                </div>
            </div>

            <div className="section-footer">
                <Button icon={<Save size={16} />}>
                    Save Changes
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

                .setting-item {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: var(--space-4);
                    background: var(--bg-elevated);
                    border-radius: var(--radius-md);
                }

                .setting-info {
                    display: flex;
                    align-items: flex-start;
                    gap: var(--space-3);
                    color: var(--text-secondary);
                }

                .setting-label {
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                    display: block;
                }

                .setting-description {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    margin: var(--space-1) 0 0;
                }

                .section-footer {
                    display: flex;
                    justify-content: flex-end;
                    gap: var(--space-2);
                    margin-top: var(--space-6);
                    padding-top: var(--space-4);
                    border-top: 1px solid var(--border-subtle);
                }
            `}</style>
        </div>
    )
}
