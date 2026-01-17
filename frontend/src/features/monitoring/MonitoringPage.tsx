import { PageHeader } from '@/components/layout/PageHeader'
import { Button } from '@/components/ui/Button'
import { RefreshCw } from 'lucide-react'
import { ValidationResultsTable } from './components/ValidationResultsTable'
import { PerformanceTrendsChart } from './components/PerformanceTrendsChart'
import { SystemHealthPanel } from './components/SystemHealthPanel'

export function MonitoringPage() {
    return (
        <div className="monitoring-page">
            <PageHeader
                title="Monitoring"
                subtitle="System health, validation results, and performance trends"
                actions={
                    <Button intent="secondary" icon={<RefreshCw size={16} />}>
                        Refresh
                    </Button>
                }
            />

            {/* System Health */}
            <section className="section">
                <h2 className="section-title">System Health</h2>
                <SystemHealthPanel />
            </section>

            {/* Performance Trends */}
            <section className="section">
                <h2 className="section-title">Performance Trends</h2>
                <PerformanceTrendsChart />
            </section>

            {/* Validation Results */}
            <section className="section">
                <h2 className="section-title">Recent Validation Results</h2>
                <ValidationResultsTable />
            </section>

            <style>{`
                .monitoring-page {
                    padding: var(--space-8);
                    max-width: 1400px;
                }

                .section {
                    margin-bottom: var(--space-8);
                }

                .section-title {
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-4) 0;
                }
            `}</style>
        </div>
    )
}
