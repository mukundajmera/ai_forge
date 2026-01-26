// =============================================================================
// Models Page - Model management and deployment
// =============================================================================

import { useState } from 'react';
import { useModels } from '@/lib/hooks';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Rocket, Check, Clock, AlertCircle, Server, Cpu } from 'lucide-react';
import type { Model } from '@/types';

interface ModelsPageProps {
    showDeploy?: boolean;
}

export function ModelsPage({ showDeploy: _showDeploy }: ModelsPageProps) {
    const { data: models, isLoading, error } = useModels();
    const [selectedModel, setSelectedModel] = useState<string | null>(null);

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="text-muted">Loading models...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="text-danger">Failed to load models</div>
            </div>
        );
    }

    const getStatusIcon = (status: Model['status']) => {
        switch (status) {
            case 'active':
                return <Check className="w-4 h-4 text-success" />;
            case 'candidate':
                return <Clock className="w-4 h-4 text-warning" />;
            case 'deprecated':
                return <AlertCircle className="w-4 h-4 text-danger" />;
            default:
                return <Server className="w-4 h-4 text-muted" />;
        }
    };

    const formatSize = (bytes: number) => {
        if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
        if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
        return `${(bytes / 1e3).toFixed(1)} KB`;
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold">Models</h1>
                    <p className="text-muted">Manage and deploy your fine-tuned models</p>
                </div>
                <Button variant="primary">
                    <Rocket className="w-4 h-4 mr-2" />
                    Deploy to Ollama
                </Button>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {models?.map((model) => (
                    <Card
                        key={model.id}
                        className={`p-4 cursor-pointer transition-colors ${
                            selectedModel === model.id ? 'border-accent' : ''
                        }`}
                        onClick={() => setSelectedModel(model.id)}
                    >
                        <div className="flex items-start justify-between mb-3">
                            <div>
                                <h3 className="font-medium">{model.name}</h3>
                                <p className="text-sm text-muted">v{model.version}</p>
                            </div>
                            {getStatusIcon(model.status)}
                        </div>

                        <div className="space-y-2 text-sm">
                            <div className="flex items-center justify-between">
                                <span className="text-muted">Base Model</span>
                                <span>{model.baseModel.split('/').pop()}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-muted">Size</span>
                                <span>{formatSize(model.size)}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-muted">Quantization</span>
                                <span className="uppercase">{model.quantization}</span>
                            </div>
                        </div>

                        {model.metrics && (
                            <div className="mt-4 pt-4 border-t border-subtle">
                                <div className="flex items-center gap-2 text-xs">
                                    <Cpu className="w-3 h-3" />
                                    <span>CodeBLEU: {(model.metrics.codeBleu * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        )}
                    </Card>
                ))}

                {(!models || models.length === 0) && (
                    <Card className="col-span-full p-8 text-center">
                        <Server className="w-12 h-12 mx-auto mb-4 text-muted" />
                        <h3 className="font-medium mb-2">No Models Yet</h3>
                        <p className="text-sm text-muted mb-4">
                            Complete a training job to create your first model
                        </p>
                        <Button variant="secondary" href="/jobs/new">
                            Start Training
                        </Button>
                    </Card>
                )}
            </div>
        </div>
    );
}

export default ModelsPage;
