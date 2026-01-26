// =============================================================================
// Mission Types - Antigravity Agent Integration
// =============================================================================

export type MissionStatus = 'active' | 'pending_approval' | 'completed' | 'failed' | 'approved' | 'rejected';
export type MissionType = 'retrain_suggestion' | 'deployment_approval' | 'quality_alert' | 'performance_drift';
export type MissionPriority = 'low' | 'medium' | 'high';
export type ArtifactType = 'chart' | 'report' | 'log' | 'dashboard';

export interface Artifact {
    id: string;
    missionId: string;
    type: ArtifactType;
    title: string;
    description?: string;
    url?: string;
    format?: string;
    size?: number;
    payload?: Record<string, unknown>;
    createdAt: string;
}

export interface RecommendedAction {
    type: 'start_training' | 'deploy_model' | 'rollback_model';
    params?: Record<string, unknown>;
    estimatedDuration?: string;
}

export interface MissionApproval {
    approvedAt?: string;
    approvedBy?: string;
    rejectedAt?: string;
    rejectedBy?: string;
    reason?: string;
}

export interface MissionReasoning {
    trigger?: string;
    analysis?: string;
    expectedOutcome?: string;
}

export interface Mission {
    id: string;
    title: string;
    description: string;
    status: MissionStatus;
    type: MissionType;
    priority?: MissionPriority;
    confidence: number; // 0-1 or 0-100
    reasoning?: MissionReasoning | string;
    createdAt: string;
    completedAt?: string;
    relatedJobIds: string[];
    relatedModelIds: string[];
    relatedDatasetIds?: string[];
    recommendedAction?: RecommendedAction;
    artifacts: Artifact[];
    approval?: MissionApproval;
}

export interface MissionsResponse {
    missions: Mission[];
    total: number;
    page?: number;
    pageSize?: number;
    pending?: number;
}
