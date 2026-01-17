// =============================================================================
// New Fine-Tune Dialog - 3-Step Wizard for Starting Training Jobs
// =============================================================================

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Dialog } from '@/components/ui/Dialog';
import { Button } from '@/components/ui/Button';
import { useDatasets, useStartFineTune } from '@/lib/hooks';
import {
    ChevronRight,
    ChevronLeft,
    Database,
    Cpu,
    Zap,
    Clock,
    AlertTriangle,
    Info
} from 'lucide-react';

// =============================================================================
// Schema & Types
// =============================================================================

const fineTuneSchema = z.object({
    projectName: z.string().min(1, 'Project name is required').max(50, 'Max 50 characters'),
    datasetId: z.string().min(1, 'Please select a dataset'),
    baseModel: z.enum(['Llama-3.2-3B', 'Llama-3.2-7B', 'Llama-3.2-13B']),
    preset: z.enum(['fast', 'balanced', 'thorough']),
    // Advanced options (override preset defaults)
    epochs: z.number().min(1).max(10).optional(),
    learningRate: z.number().min(0.00001).max(0.01).optional(),
    rank: z.number().min(8).max(256).optional(),
    batchSize: z.number().min(1).max(8).optional(),
});

type FineTuneFormData = z.infer<typeof fineTuneSchema>;

interface PresetConfig {
    epochs: number;
    lr: number;
    rank: number;
    batchSize: number;
    estimatedTime: string;
    description: string;
}

const PRESETS: Record<string, PresetConfig> = {
    fast: {
        epochs: 1,
        lr: 2e-4,
        rank: 32,
        batchSize: 2,
        estimatedTime: '15-20 min',
        description: 'Quick iteration, good for testing'
    },
    balanced: {
        epochs: 3,
        lr: 2e-4,
        rank: 64,
        batchSize: 2,
        estimatedTime: '45-60 min',
        description: 'Recommended for most use cases'
    },
    thorough: {
        epochs: 5,
        lr: 1e-4,
        rank: 128,
        batchSize: 2,
        estimatedTime: '2-3 hours',
        description: 'Maximum quality, longer training'
    },
};

const MODEL_INFO: Record<string, { memory: string; speed: string }> = {
    'Llama-3.2-3B': { memory: '6-8GB RAM', speed: 'Fastest' },
    'Llama-3.2-7B': { memory: '10-14GB RAM', speed: 'Balanced' },
    'Llama-3.2-13B': { memory: '16-20GB RAM', speed: 'Best quality' },
};

// =============================================================================
// Component
// =============================================================================

interface NewFineTuneDialogProps {
    open: boolean;
    onClose: () => void;
}

export function NewFineTuneDialog({ open, onClose }: NewFineTuneDialogProps) {
    const navigate = useNavigate();
    const [step, setStep] = useState(1);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const { data: datasets, isLoading: datasetsLoading } = useDatasets();
    const startFineTune = useStartFineTune();

    const {
        register,
        handleSubmit,
        watch,
        formState: { errors },
        reset,
    } = useForm<FineTuneFormData>({
        resolver: zodResolver(fineTuneSchema),
        defaultValues: {
            projectName: '',
            datasetId: '',
            baseModel: 'Llama-3.2-3B',
            preset: 'balanced',
        },
    });

    const selectedDatasetId = watch('datasetId');
    const selectedPreset = watch('preset');
    const selectedBaseModel = watch('baseModel');
    const projectName = watch('projectName');

    const dataset = datasets?.find(d => d.id === selectedDatasetId);
    const preset = PRESETS[selectedPreset];

    const handleClose = () => {
        reset();
        setStep(1);
        setShowAdvanced(false);
        onClose();
    };

    const onSubmit = async (data: FineTuneFormData) => {
        try {
            const config = {
                projectName: data.projectName,
                baseModel: data.baseModel,
                datasetId: data.datasetId,
                epochs: data.epochs || preset.epochs,
                learningRate: data.learningRate || preset.lr,
                rank: data.rank || preset.rank,
                batchSize: data.batchSize || preset.batchSize,
                method: 'pissa' as const,
            };

            const result = await startFineTune.mutateAsync(config);
            handleClose();
            navigate(`/jobs/${result.jobId}`);
        } catch (error) {
            console.error('Failed to start training:', error);
        }
    };

    const canProceed = () => {
        if (step === 1) return projectName && selectedDatasetId;
        if (step === 2) return selectedBaseModel;
        return true;
    };

    return (
        <Dialog isOpen={open} onClose={handleClose} title="New Fine-Tune" size="lg">
            <form onSubmit={handleSubmit(onSubmit)}>
                {/* Progress Indicator */}
                <div className="wizard-progress">
                    {[1, 2, 3].map((s) => (
                        <div
                            key={s}
                            className={`progress-step ${s === step ? 'active' : s < step ? 'completed' : ''
                                }`}
                        >
                            <div className="step-dot">
                                {s < step ? '✓' : s}
                            </div>
                            <span className="step-label">
                                {s === 1 && 'Dataset'}
                                {s === 2 && 'Config'}
                                {s === 3 && 'Review'}
                            </span>
                        </div>
                    ))}
                </div>

                {/* Step 1: Project & Dataset */}
                {step === 1 && (
                    <div className="wizard-step">
                        <h3 className="step-title">
                            <Database size={20} />
                            Select Training Data
                        </h3>

                        {/* Project Name */}
                        <div className="form-group">
                            <label className="form-label">Project Name</label>
                            <input
                                {...register('projectName')}
                                className="form-input"
                                placeholder="my-code-assistant"
                                autoFocus
                            />
                            {errors.projectName && (
                                <p className="form-error">{errors.projectName.message}</p>
                            )}
                        </div>

                        {/* Dataset Selection */}
                        <div className="form-group">
                            <label className="form-label">Training Dataset</label>
                            {datasetsLoading ? (
                                <div className="loading-placeholder">Loading datasets...</div>
                            ) : datasets && datasets.length > 0 ? (
                                <div className="radio-cards">
                                    {datasets.map((ds) => (
                                        <label
                                            key={ds.id}
                                            className={`radio-card ${selectedDatasetId === ds.id ? 'selected' : ''}`}
                                        >
                                            <input
                                                type="radio"
                                                {...register('datasetId')}
                                                value={ds.id}
                                            />
                                            <div className="card-content">
                                                <div className="card-title">{ds.name}</div>
                                                <div className="card-meta">
                                                    <span>{ds.exampleCount} examples</span>
                                                    <span className="separator">•</span>
                                                    <span>Quality: {(ds.qualityMetrics?.avgScore * 100 || 0).toFixed(0)}%</span>
                                                </div>
                                            </div>
                                        </label>
                                    ))}
                                </div>
                            ) : (
                                <div className="empty-state-inline">
                                    <p>No datasets available.</p>
                                    <Button
                                        intent="secondary"
                                        size="sm"
                                        onClick={() => {
                                            handleClose();
                                            navigate('/datasets');
                                        }}
                                    >
                                        Create Dataset
                                    </Button>
                                </div>
                            )}
                            {errors.datasetId && (
                                <p className="form-error">{errors.datasetId.message}</p>
                            )}
                        </div>

                        {/* Quality Warning */}
                        {dataset && (dataset.qualityMetrics?.avgScore || 0) < 0.5 && (
                            <div className="warning-banner">
                                <AlertTriangle size={16} />
                                <span>
                                    Low quality score ({((dataset.qualityMetrics?.avgScore || 0) * 100).toFixed(0)}%).
                                    Consider improving your data before training.
                                </span>
                            </div>
                        )}
                    </div>
                )}

                {/* Step 2: Model & Training Config */}
                {step === 2 && (
                    <div className="wizard-step">
                        <h3 className="step-title">
                            <Cpu size={20} />
                            Configure Training
                        </h3>

                        {/* Base Model Selection */}
                        <div className="form-group">
                            <label className="form-label">Base Model</label>
                            <div className="radio-cards">
                                {Object.entries(MODEL_INFO).map(([model, info]) => (
                                    <label
                                        key={model}
                                        className={`radio-card ${selectedBaseModel === model ? 'selected' : ''}`}
                                    >
                                        <input
                                            type="radio"
                                            {...register('baseModel')}
                                            value={model}
                                        />
                                        <div className="card-content">
                                            <div className="card-title">{model}</div>
                                            <div className="card-meta">
                                                <span>{info.memory}</span>
                                                <span className="separator">•</span>
                                                <span>{info.speed}</span>
                                            </div>
                                        </div>
                                    </label>
                                ))}
                            </div>
                        </div>

                        {/* Preset Selection */}
                        <div className="form-group">
                            <label className="form-label">Training Preset</label>
                            <div className="radio-cards">
                                {Object.entries(PRESETS).map(([key, config]) => (
                                    <label
                                        key={key}
                                        className={`radio-card ${selectedPreset === key ? 'selected' : ''}`}
                                    >
                                        <input
                                            type="radio"
                                            {...register('preset')}
                                            value={key}
                                        />
                                        <div className="card-content">
                                            <div className="card-title capitalize">{key}</div>
                                            <div className="card-description">{config.description}</div>
                                            <div className="card-meta">
                                                <span>{config.epochs} epochs</span>
                                                <span className="separator">•</span>
                                                <span>rank {config.rank}</span>
                                                <span className="separator">•</span>
                                                <span>~{config.estimatedTime}</span>
                                            </div>
                                        </div>
                                    </label>
                                ))}
                            </div>
                        </div>

                        {/* Advanced Options */}
                        <details
                            className="advanced-options"
                            open={showAdvanced}
                            onToggle={(e) => setShowAdvanced((e.target as HTMLDetailsElement).open)}
                        >
                            <summary>Advanced Options (Optional)</summary>
                            <div className="advanced-grid">
                                <div className="form-group">
                                    <label className="form-label">Epochs</label>
                                    <input
                                        type="number"
                                        {...register('epochs', { valueAsNumber: true })}
                                        className="form-input"
                                        placeholder={preset.epochs.toString()}
                                        min={1}
                                        max={10}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Learning Rate</label>
                                    <input
                                        type="number"
                                        step="0.00001"
                                        {...register('learningRate', { valueAsNumber: true })}
                                        className="form-input"
                                        placeholder={preset.lr.toString()}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Rank</label>
                                    <input
                                        type="number"
                                        {...register('rank', { valueAsNumber: true })}
                                        className="form-input"
                                        placeholder={preset.rank.toString()}
                                        min={8}
                                        max={256}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Batch Size</label>
                                    <input
                                        type="number"
                                        {...register('batchSize', { valueAsNumber: true })}
                                        className="form-input"
                                        placeholder={preset.batchSize.toString()}
                                        min={1}
                                        max={8}
                                    />
                                </div>
                            </div>
                        </details>
                    </div>
                )}

                {/* Step 3: Review & Confirm */}
                {step === 3 && (
                    <div className="wizard-step">
                        <h3 className="step-title">
                            <Zap size={20} />
                            Review & Start
                        </h3>

                        <div className="review-card">
                            <div className="review-row">
                                <span className="review-label">Project</span>
                                <span className="review-value">{projectName}</span>
                            </div>
                            <div className="review-row">
                                <span className="review-label">Dataset</span>
                                <span className="review-value">{dataset?.name || 'Unknown'}</span>
                            </div>
                            <div className="review-row">
                                <span className="review-label">Base Model</span>
                                <span className="review-value">{selectedBaseModel}</span>
                            </div>
                            <div className="review-row">
                                <span className="review-label">Epochs</span>
                                <span className="review-value">{watch('epochs') || preset.epochs}</span>
                            </div>
                            <div className="review-row">
                                <span className="review-label">Learning Rate</span>
                                <span className="review-value">{watch('learningRate') || preset.lr}</span>
                            </div>
                            <div className="review-row">
                                <span className="review-label">Rank</span>
                                <span className="review-value">{watch('rank') || preset.rank}</span>
                            </div>
                            <div className="review-row highlight">
                                <span className="review-label">
                                    <Clock size={14} />
                                    Estimated Time
                                </span>
                                <span className="review-value">{preset.estimatedTime}</span>
                            </div>
                        </div>

                        <div className="info-banner">
                            <Info size={16} />
                            <span>
                                Using <strong>PiSSA</strong> initialization for 3-5x faster convergence
                                compared to standard LoRA.
                            </span>
                        </div>
                    </div>
                )}

                {/* Navigation Buttons */}
                <div className="wizard-actions">
                    <Button
                        type="button"
                        intent="ghost"
                        onClick={() => step === 1 ? handleClose() : setStep(step - 1)}
                        icon={step > 1 ? <ChevronLeft size={16} /> : undefined}
                    >
                        {step === 1 ? 'Cancel' : 'Back'}
                    </Button>

                    {step < 3 ? (
                        <Button
                            type="button"
                            intent="primary"
                            onClick={() => setStep(step + 1)}
                            disabled={!canProceed()}
                            icon={<ChevronRight size={16} />}
                        >
                            Next
                        </Button>
                    ) : (
                        <Button
                            type="submit"
                            intent="primary"
                            loading={startFineTune.isPending}
                            icon={<Zap size={16} />}
                        >
                            Start Training
                        </Button>
                    )}
                </div>
            </form>

            <style>{`
                .wizard-progress {
                    display: flex;
                    justify-content: center;
                    gap: var(--space-8);
                    margin-bottom: var(--space-6);
                    padding-bottom: var(--space-6);
                    border-bottom: 1px solid var(--border-subtle);
                }

                .progress-step {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: var(--space-2);
                }

                .step-dot {
                    width: 32px;
                    height: 32px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    background: var(--bg-elevated);
                    border: 2px solid var(--border-subtle);
                    color: var(--text-tertiary);
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    transition: all 0.2s;
                }

                .progress-step.active .step-dot {
                    background: var(--accent-primary);
                    border-color: var(--accent-primary);
                    color: white;
                }

                .progress-step.completed .step-dot {
                    background: var(--status-success);
                    border-color: var(--status-success);
                    color: white;
                }

                .step-label {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }

                .progress-step.active .step-label {
                    color: var(--text-primary);
                }

                .wizard-step {
                    min-height: 300px;
                }

                .step-title {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    font-size: var(--text-lg);
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                    margin-bottom: var(--space-5);
                }

                .form-group {
                    margin-bottom: var(--space-4);
                }

                .form-label {
                    display: block;
                    font-size: var(--text-sm);
                    font-weight: var(--font-medium);
                    color: var(--text-secondary);
                    margin-bottom: var(--space-2);
                }

                .form-input {
                    width: 100%;
                    padding: var(--space-3) var(--space-4);
                    border: 1px solid var(--border-default);
                    border-radius: var(--radius-md);
                    background: var(--bg-elevated);
                    color: var(--text-primary);
                    font-size: var(--text-base);
                    transition: border-color 0.2s;
                }

                .form-input:focus {
                    outline: none;
                    border-color: var(--accent-primary);
                }

                .form-input::placeholder {
                    color: var(--text-tertiary);
                }

                .form-error {
                    margin-top: var(--space-1);
                    font-size: var(--text-sm);
                    color: var(--status-danger);
                }

                .radio-cards {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-3);
                }

                .radio-card {
                    display: flex;
                    align-items: flex-start;
                    gap: var(--space-3);
                    padding: var(--space-4);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    background: var(--bg-surface);
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .radio-card:hover {
                    border-color: var(--border-default);
                }

                .radio-card.selected {
                    border-color: var(--accent-primary);
                    background: rgba(99, 102, 241, 0.05);
                }

                .radio-card input[type="radio"] {
                    margin-top: var(--space-1);
                    accent-color: var(--accent-primary);
                }

                .card-content {
                    flex: 1;
                }

                .card-title {
                    font-weight: var(--font-semibold);
                    color: var(--text-primary);
                }

                .card-description {
                    font-size: var(--text-sm);
                    color: var(--text-secondary);
                    margin-top: var(--space-1);
                }

                .card-meta {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    margin-top: var(--space-2);
                    font-size: var(--text-sm);
                    color: var(--text-tertiary);
                }

                .separator {
                    color: var(--border-default);
                }

                .capitalize {
                    text-transform: capitalize;
                }

                .loading-placeholder,
                .empty-state-inline {
                    padding: var(--space-6);
                    text-align: center;
                    background: var(--bg-elevated);
                    border-radius: var(--radius-lg);
                    color: var(--text-secondary);
                }

                .empty-state-inline p {
                    margin-bottom: var(--space-3);
                }

                .warning-banner {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    padding: var(--space-3) var(--space-4);
                    background: var(--status-warning-bg);
                    border: 1px solid rgba(245, 158, 11, 0.2);
                    border-radius: var(--radius-md);
                    font-size: var(--text-sm);
                    color: var(--status-warning);
                }

                .info-banner {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    padding: var(--space-3) var(--space-4);
                    background: var(--status-info-bg);
                    border: 1px solid rgba(59, 130, 246, 0.2);
                    border-radius: var(--radius-md);
                    font-size: var(--text-sm);
                    color: var(--status-info);
                    margin-top: var(--space-4);
                }

                .advanced-options {
                    margin-top: var(--space-4);
                    padding: var(--space-4);
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                }

                .advanced-options summary {
                    cursor: pointer;
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .advanced-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: var(--space-4);
                    margin-top: var(--space-4);
                }

                .review-card {
                    background: var(--bg-surface);
                    border: 1px solid var(--border-subtle);
                    border-radius: var(--radius-lg);
                    overflow: hidden;
                }

                .review-row {
                    display: flex;
                    justify-content: space-between;
                    padding: var(--space-3) var(--space-4);
                    border-bottom: 1px solid var(--border-subtle);
                }

                .review-row:last-child {
                    border-bottom: none;
                }

                .review-row.highlight {
                    background: var(--bg-elevated);
                }

                .review-label {
                    display: flex;
                    align-items: center;
                    gap: var(--space-2);
                    color: var(--text-secondary);
                    font-size: var(--text-sm);
                }

                .review-value {
                    font-weight: var(--font-medium);
                    color: var(--text-primary);
                }

                .wizard-actions {
                    display: flex;
                    justify-content: space-between;
                    margin-top: var(--space-6);
                    padding-top: var(--space-6);
                    border-top: 1px solid var(--border-subtle);
                }

                @media (max-width: 640px) {
                    .advanced-grid {
                        grid-template-columns: 1fr;
                    }
                }
            `}</style>
        </Dialog>
    );
}
