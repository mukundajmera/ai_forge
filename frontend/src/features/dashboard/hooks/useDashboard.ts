import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'

interface SystemHealth {
    status: 'healthy' | 'degraded' | 'error';
    ollama: {
        connected: boolean;
        version?: string;
        models: number;
    };
    system: {
        memory_used: number;
        memory_total: number;
        gpu?: string;
    };
    activeModel?: {
        name: string;
        version: string;
    };
}

interface ActiveJob {
    id: string;
    project: string;
    status: 'queued' | 'training' | 'completed' | 'failed';
    progress: number;
    currentEpoch?: number;
    totalEpochs?: number;
    loss?: number;
    eta?: string;
    startedAt: string;
}

interface RecentActivity {
    id: string;
    type: 'job_started' | 'job_completed' | 'job_failed' | 'model_deployed' | 'dataset_created';
    message: string;
    timestamp: string;
    metadata?: Record<string, unknown>;
}

interface DashboardData {
    health: SystemHealth;
    activeJobs: ActiveJob[];
    recentActivity: RecentActivity[];
    stats: {
        totalJobs: number;
        completedJobs: number;
        totalModels: number;
        totalDatasets: number;
    };
}

export function useDashboard() {
    return useQuery({
        queryKey: ['dashboard'],
        queryFn: () => api.get<DashboardData>('/dashboard'),
        refetchInterval: 5000, // Refresh every 5 seconds for live updates
    })
}

export function useSystemHealth() {
    return useQuery({
        queryKey: ['health'],
        queryFn: () => api.get<SystemHealth>('/health'),
        refetchInterval: 10000,
    })
}

export function useActiveJobs() {
    return useQuery({
        queryKey: ['jobs', 'active'],
        queryFn: () => api.get<ActiveJob[]>('/jobs?status=training,queued'),
        refetchInterval: 3000,
    })
}

export function useRecentActivity(limit = 10) {
    return useQuery({
        queryKey: ['activity', limit],
        queryFn: () => api.get<RecentActivity[]>(`/activity?limit=${limit}`),
        refetchInterval: 10000,
    })
}

export type { SystemHealth, ActiveJob, RecentActivity, DashboardData }
