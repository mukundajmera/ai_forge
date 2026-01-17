import { useState } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Button } from '@/components/ui/Button'
import { mockMissions } from '@/utils/mock-data'
import { MissionCard } from './components/MissionCard'
import { Filter, RefreshCw } from 'lucide-react'
import type { MissionStatus } from '@/types'

const statusFilters: { label: string; value: MissionStatus | 'all' }[] = [
    { label: 'All', value: 'all' },
    { label: 'Pending', value: 'pending_approval' },
    { label: 'Active', value: 'active' },
    { label: 'Completed', value: 'completed' },
    { label: 'Failed', value: 'failed' },
]

export function MissionsPage() {
    const [activeFilter, setActiveFilter] = useState<MissionStatus | 'all'>('all')

    const filteredMissions = activeFilter === 'all'
        ? mockMissions
        : mockMissions.filter(m => m.status === activeFilter)

    const pendingCount = mockMissions.filter(m => m.status === 'pending_approval').length

    return (
        <div className="missions-page">
            <PageHeader
                title="Missions"
                subtitle="AI-powered recommendations and automated workflows"
                actions={
                    <Button intent="secondary" icon={<RefreshCw size={16} />}>
                        Refresh
                    </Button>
                }
            />

            {/* Pending Alert Banner */}
            {pendingCount > 0 && (
                <div className="pending-banner">
                    <span className="pending-badge">{pendingCount}</span>
                    <span>mission{pendingCount > 1 ? 's' : ''} pending your approval</span>
                </div>
            )}

            {/* Filters */}
            <div className="filters">
                <Filter size={16} />
                {statusFilters.map(filter => (
                    <button
                        key={filter.value}
                        className={`filter-btn ${activeFilter === filter.value ? 'active' : ''}`}
                        onClick={() => setActiveFilter(filter.value)}
                    >
                        {filter.label}
                    </button>
                ))}
            </div>

            {/* Missions Grid */}
            {filteredMissions.length === 0 ? (
                <div className="empty-state">
                    <p>No missions match this filter</p>
                </div>
            ) : (
                <div className="missions-grid">
                    {filteredMissions.map(mission => (
                        <MissionCard key={mission.id} mission={mission} />
                    ))}
                </div>
            )}

            <style>{`
                .missions-page {
                    padding: var(--space-8);
                    max-width: 1400px;
                }

                .pending-banner {
                    display: flex;
                    align-items: center;
                    gap: var(--space-3);
                    padding: var(--space-3) var(--space-4);
                    background: var(--status-warning-bg);
                    border: 1px solid var(--status-warning-border);
                    border-radius: var(--radius-lg);
                    margin-bottom: var(--space-6);
                    color: var(--status-warning);
                    font-weight: var(--font-medium);
                }

                .pending-badge {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 24px;
                    height: 24px;
                    background: var(--status-warning);
                    color: var(--text-inverse);
                    font-size: var(--text-sm);
                    font-weight: var(--font-bold);
                    border-radius: var(--radius-full);
                }

                .filters {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    margin-bottom: var(--space-6);
                    color: var(--text-secondary);
                }

                .filter-btn {
                    padding: var(--space-2) var(--space-3);
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-secondary);
                    background: none;
                    border: 1px solid var(--border-default);
                    border-radius: var(--radius-full);
                    cursor: pointer;
                    transition: all var(--transition-fast);
                }

                .filter-btn:hover {
                    background: var(--bg-hover);
                    color: var(--text-primary);
                }

                .filter-btn.active {
                    background: var(--accent-primary);
                    border-color: var(--accent-primary);
                    color: white;
                }

                .missions-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
                    gap: var(--space-4);
                }

                .empty-state {
                    text-align: center;
                    padding: var(--space-12);
                    color: var(--text-secondary);
                }
            `}</style>
        </div>
    )
}
