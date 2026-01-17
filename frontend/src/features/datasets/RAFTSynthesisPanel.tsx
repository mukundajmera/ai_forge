import { useState } from 'react'
import { Sparkles, RefreshCw, Check, AlertTriangle, Zap } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Progress, QualityBar } from '@/components/ui/Progress'
import { Card, MetricCard } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import type { TrainingExample, QualityMetrics, GenerationJob } from '@/types'

interface RAFTSynthesisPanelProps {
    job?: GenerationJob | null;
    examples?: TrainingExample[];
    metrics?: QualityMetrics;
    onRegenerate?: () => void;
    onApprove?: () => void;
    isGenerating?: boolean;
}

export function RAFTSynthesisPanel({
    job,
    examples = [],
    metrics,
    onRegenerate,
    onApprove,
    isGenerating = false,
}: RAFTSynthesisPanelProps) {
    const [expandedExample, setExpandedExample] = useState<number | null>(null)

    const isComplete = job?.status === 'complete'
    const isFailed = job?.status === 'failed'
    const progress = job?.progress || 0

    return (
        <div className="raft-synthesis-panel">
            {/* Header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: 'var(--space-6)'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                    <div style={{
                        width: 40,
                        height: 40,
                        borderRadius: 'var(--radius-md)',
                        background: 'linear-gradient(135deg, var(--primary-500), var(--primary-600))',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                        <Sparkles size={20} color="white" />
                    </div>
                    <div>
                        <h2 style={{ fontSize: 'var(--text-lg)', fontWeight: 600 }}>
                            RAFT Synthesis
                        </h2>
                        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                            Generate training examples using Retrieval-Augmented Fine-Tuning
                        </p>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
                    {onRegenerate && (
                        <Button
                            variant="secondary"
                            icon={<RefreshCw size={16} />}
                            onClick={onRegenerate}
                            disabled={isGenerating}
                        >
                            Regenerate
                        </Button>
                    )}
                    {onApprove && isComplete && (
                        <Button
                            icon={<Check size={16} />}
                            onClick={onApprove}
                        >
                            Approve & Save
                        </Button>
                    )}
                </div>
            </div>

            {/* Progress section - shown when generating */}
            {(isGenerating || (job && !isComplete && !isFailed)) && (
                <Card style={{ marginBottom: 'var(--space-6)' }}>
                    <div style={{ marginBottom: 'var(--space-4)' }}>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            marginBottom: 'var(--space-2)'
                        }}>
                            <span style={{ fontWeight: 500 }}>
                                {job?.currentStep || 'Generating training examples...'}
                            </span>
                            <span style={{ color: 'var(--text-secondary)' }}>
                                {job?.examplesGenerated || 0} / {job?.totalExpected || '?'}
                            </span>
                        </div>
                        <Progress value={progress} />
                    </div>

                    {job?.status === 'running' && (
                        <p style={{
                            fontSize: 'var(--text-sm)',
                            color: 'var(--text-muted)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 'var(--space-2)'
                        }}>
                            <span className="spinner spinner-sm" />
                            Processing code blocks and generating Q&A pairs...
                        </p>
                    )}
                </Card>
            )}

            {/* Error state */}
            {isFailed && (
                <Card style={{
                    marginBottom: 'var(--space-6)',
                    borderColor: 'var(--error-500)',
                    background: 'rgba(239, 68, 68, 0.05)'
                }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 'var(--space-3)'
                    }}>
                        <AlertTriangle size={24} style={{ color: 'var(--error-500)' }} />
                        <div>
                            <h3 style={{ fontWeight: 600, marginBottom: 'var(--space-1)' }}>
                                Generation Failed
                            </h3>
                            <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--text-sm)' }}>
                                {job?.error || 'An error occurred during synthesis. Please try again.'}
                            </p>
                        </div>
                    </div>
                </Card>
            )}

            {/* Quality Metrics */}
            {metrics && isComplete && (
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: 'var(--space-4)',
                    marginBottom: 'var(--space-6)'
                }}>
                    <MetricCard
                        label="Total Examples"
                        value={job?.examplesGenerated || examples.length}
                        icon={<Zap size={16} />}
                    />
                    <MetricCard
                        label="Avg Quality"
                        value={metrics.avgScore.toFixed(2)}
                    />
                    <MetricCard
                        label="Diversity Score"
                        value={metrics.diversity.toFixed(2)}
                    />
                    <MetricCard
                        label="RAFT Distractor Quality"
                        value={metrics.raftDistractorQuality.toFixed(2)}
                    />
                </div>
            )}

            {/* Sample Examples */}
            {examples.length > 0 && (
                <div>
                    <h3 style={{
                        fontSize: 'var(--text-base)',
                        fontWeight: 600,
                        marginBottom: 'var(--space-4)'
                    }}>
                        Sample Generated Examples ({examples.length})
                    </h3>

                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-3)'
                    }}>
                        {examples.slice(0, 5).map((example, index) => (
                            <ExampleCard
                                key={example.id || index}
                                example={example}
                                index={index + 1}
                                isExpanded={expandedExample === index}
                                onToggle={() => setExpandedExample(
                                    expandedExample === index ? null : index
                                )}
                            />
                        ))}
                    </div>

                    {examples.length > 5 && (
                        <p style={{
                            textAlign: 'center',
                            marginTop: 'var(--space-4)',
                            color: 'var(--text-muted)',
                            fontSize: 'var(--text-sm)'
                        }}>
                            + {examples.length - 5} more examples
                        </p>
                    )}
                </div>
            )}

            {/* Warnings */}
            {isComplete && examples.length < 100 && (
                <div style={{
                    marginTop: 'var(--space-6)',
                    padding: 'var(--space-3)',
                    background: 'rgba(234, 179, 8, 0.1)',
                    border: '1px solid var(--warning-500)',
                    borderRadius: 'var(--radius-md)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    fontSize: 'var(--text-sm)',
                    color: 'var(--warning-500)'
                }}>
                    <AlertTriangle size={16} />
                    Small dataset ({examples.length} examples) may cause overfitting.
                    Consider adding more source files for 500+ examples.
                </div>
            )}
        </div>
    )
}

// Example card component
function ExampleCard({
    example,
    index,
    isExpanded,
    onToggle,
}: {
    example: TrainingExample;
    index: number;
    isExpanded: boolean;
    onToggle: () => void;
}) {
    return (
        <Card
            onClick={onToggle}
            style={{ cursor: 'pointer' }}
        >
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                marginBottom: isExpanded ? 'var(--space-4)' : 0
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                    <span style={{
                        width: 24,
                        height: 24,
                        borderRadius: 'var(--radius-full)',
                        background: 'var(--bg-overlay)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 'var(--text-xs)',
                        fontWeight: 600
                    }}>
                        {index}
                    </span>
                    <div>
                        <Badge variant="neutral" style={{ marginBottom: 'var(--space-2)' }}>
                            {example.questionType || 'purpose'}
                        </Badge>
                        <p style={{ fontWeight: 500 }}>Q: {example.question}</p>
                    </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                    <QualityBar score={example.qualityScore} segments={3} />
                    <span style={{
                        transform: isExpanded ? 'rotate(180deg)' : 'none',
                        transition: 'transform 0.2s',
                        color: 'var(--text-muted)'
                    }}>
                        ▼
                    </span>
                </div>
            </div>

            {isExpanded && (
                <div style={{
                    paddingTop: 'var(--space-4)',
                    borderTop: '1px solid var(--border)'
                }}>
                    {/* Context */}
                    <div style={{ marginBottom: 'var(--space-4)' }}>
                        <h4 style={{
                            fontSize: 'var(--text-xs)',
                            color: 'var(--text-muted)',
                            marginBottom: 'var(--space-2)',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em'
                        }}>
                            Context
                        </h4>
                        <pre style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: 'var(--text-xs)',
                            background: 'var(--bg-overlay)',
                            padding: 'var(--space-3)',
                            borderRadius: 'var(--radius-md)',
                            overflow: 'auto',
                            maxHeight: 150
                        }}>
                            {example.context || 'No context provided'}
                        </pre>
                    </div>

                    {/* Answer */}
                    <div>
                        <h4 style={{
                            fontSize: 'var(--text-xs)',
                            color: 'var(--text-muted)',
                            marginBottom: 'var(--space-2)',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em'
                        }}>
                            Answer
                        </h4>
                        <p style={{
                            fontSize: 'var(--text-sm)',
                            lineHeight: 1.6,
                            color: 'var(--text-secondary)'
                        }}>
                            {example.answer}
                        </p>
                    </div>

                    {/* Reasoning if present */}
                    {example.reasoning && (
                        <div style={{ marginTop: 'var(--space-4)' }}>
                            <h4 style={{
                                fontSize: 'var(--text-xs)',
                                color: 'var(--text-muted)',
                                marginBottom: 'var(--space-2)',
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em'
                            }}>
                                Chain of Thought
                            </h4>
                            <p style={{
                                fontSize: 'var(--text-sm)',
                                lineHeight: 1.6,
                                color: 'var(--text-muted)',
                                fontStyle: 'italic'
                            }}>
                                {example.reasoning}
                            </p>
                        </div>
                    )}
                </div>
            )}
        </Card>
    )
}
