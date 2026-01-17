import { Routes, Route } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { PageLayout } from './components/layout/PageLayout'
import { DashboardPage } from './features/dashboard'
import { DataSourcesPage, DatasetsPage } from './features/datasets'
import { JobsListPage, JobDetailPage } from './features/jobs'
import { ModelsPage } from './features/models'
import { MonitoringPage } from './features/monitoring'
import { MissionsPage, MissionDetailPage } from './features/missions'
import { SettingsPage } from './features/settings'

function App() {
    return (
        <ErrorBoundary>
            <Routes>
                <Route element={<PageLayout />}>
                    <Route path="/" element={<DashboardPage />} />
                    <Route path="/datasets" element={<DataSourcesPage />} />
                    <Route path="/datasets/generated" element={<DatasetsPage />} />
                    <Route path="/jobs" element={<JobsListPage />} />
                    <Route path="/jobs/new" element={<JobDetailPage />} />
                    <Route path="/jobs/:jobId" element={<JobDetailPage />} />
                    <Route path="/models" element={<ModelsPage />} />
                    <Route path="/monitoring" element={<MonitoringPage />} />
                    <Route path="/missions" element={<MissionsPage />} />
                    <Route path="/missions/:missionId" element={<MissionDetailPage />} />
                    <Route path="/settings" element={<SettingsPage />} />
                </Route>
            </Routes>
        </ErrorBoundary>
    )
}

export default App


