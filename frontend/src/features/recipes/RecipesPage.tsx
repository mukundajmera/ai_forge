import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
    BookOpen,
    Code2,
    FileText,
    MessageSquare,
    GraduationCap,
    ChevronRight,
    Cpu,
    Zap,
    Settings2,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { EmptyState } from '@/components/ui/EmptyState'
import { useRecipes, useRecipe } from './hooks/useRecipes'
import type { TaskType, HardwareProfile } from '@/types'

const taskTypeIcons: Record<TaskType, React.ReactNode> = {
    instruction_tuning: <GraduationCap size={24} />,
    domain_adaptation: <FileText size={24} />,
    code_specialization: <Code2 size={24} />,
    qa_finetuning: <MessageSquare size={24} />,
}

const taskTypeLabels: Record<TaskType, string> = {
    instruction_tuning: 'Instruction Tuning',
    domain_adaptation: 'Domain Adaptation',
    code_specialization: 'Code Specialization',
    qa_finetuning: 'QA Fine-Tuning',
}

const hardwareOptions: { value: HardwareProfile; label: string; description: string }[] = [
    { value: 'low', label: '8GB', description: 'M1/M2 base' },
    { value: 'medium', label: '16GB', description: 'M1/M2 Pro' },
    { value: 'high', label: '32GB+', description: 'M1/M2 Max/Ultra' },
]

export function RecipesPage() {
    const [selectedRecipeId, setSelectedRecipeId] = useState<string | null>(null)
    const [hardware, setHardware] = useState<HardwareProfile>('medium')
    const [taskFilter, setTaskFilter] = useState<string>('all')

    const { data: recipes = [], isLoading, error } = useRecipes(
        taskFilter !== 'all' ? { task_type: taskFilter } : undefined
    )
    const { data: recipeDetail } = useRecipe(selectedRecipeId ?? undefined, hardware)

    const filteredRecipes = recipes

    return (
        <div className="recipes-page">
            <header className="page-header">
                <div>
                    <h1>Training Recipes</h1>
                    <p>Opinionated, hardware-aware training configurations</p>
                </div>
            </header>

            {/* Task Type Filter */}
            <div style={{
                display: 'flex',
                gap: 'var(--space-2)',
                marginBottom: 'var(--space-6)',
                flexWrap: 'wrap',
            }}>
                <button
                    className={`filter-chip ${taskFilter === 'all' ? 'active' : ''}`}
                    onClick={() => setTaskFilter('all')}
                >
                    All
                </button>
                {Object.entries(taskTypeLabels).map(([key, label]) => (
                    <button
                        key={key}
                        className={`filter-chip ${taskFilter === key ? 'active' : ''}`}
                        onClick={() => setTaskFilter(key)}
                    >
                        {label}
                    </button>
                ))}
            </div>

            {isLoading ? (
                <div style={{ padding: 'var(--space-12)', textAlign: 'center', color: 'var(--text-muted)' }}>
                    Loading recipes...
                </div>
            ) : error ? (
                <EmptyState
                    icon={<BookOpen size={48} />}
                    title="Failed to load recipes"
                    description={error instanceof Error ? error.message : 'Unknown error'}
                />
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: selectedRecipeId ? '1fr 1fr' : '1fr', gap: 'var(--space-6)' }}>
                    {/* Recipe Cards */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                        {filteredRecipes.map(recipe => (
                            <div
                                key={recipe.id}
                                onClick={() => setSelectedRecipeId(recipe.id === selectedRecipeId ? null : recipe.id)}
                                style={{
                                    background: recipe.id === selectedRecipeId ? 'var(--surface-3)' : 'var(--surface-2)',
                                    borderRadius: 'var(--radius-lg)',
                                    padding: 'var(--space-5)',
                                    border: recipe.id === selectedRecipeId ? '2px solid var(--accent)' : '1px solid var(--border)',
                                    cursor: 'pointer',
                                    transition: 'all 0.15s ease',
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-4)' }}>
                                    <div style={{
                                        width: 48,
                                        height: 48,
                                        borderRadius: 'var(--radius-md)',
                                        background: 'var(--surface-1)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        color: 'var(--accent)',
                                        flexShrink: 0,
                                    }}>
                                        {taskTypeIcons[recipe.taskType]}
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-1)' }}>
                                            <span style={{ fontWeight: 600, fontSize: 'var(--text-base)' }}>{recipe.name}</span>
                                            {recipe.isBuiltin && (
                                                <span className="badge-info" style={{ fontSize: 'var(--text-xs)' }}>Built-in</span>
                                            )}
                                        </div>
                                        <p style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                                            {recipe.description}
                                        </p>
                                        <div style={{ display: 'flex', gap: 'var(--space-1)', marginTop: 'var(--space-2)', flexWrap: 'wrap' }}>
                                            {recipe.tags.map(tag => (
                                                <span key={tag} className="badge-muted" style={{ fontSize: 'var(--text-xs)' }}>
                                                    {tag}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                    <ChevronRight size={18} style={{
                                        color: 'var(--text-muted)',
                                        transform: recipe.id === selectedRecipeId ? 'rotate(90deg)' : 'none',
                                        transition: 'transform 0.2s ease',
                                        flexShrink: 0,
                                    }} />
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Recipe Detail Panel */}
                    {selectedRecipeId && recipeDetail && (
                        <div style={{
                            background: 'var(--surface-2)',
                            borderRadius: 'var(--radius-lg)',
                            padding: 'var(--space-6)',
                            border: '1px solid var(--border)',
                            position: 'sticky',
                            top: 'var(--space-4)',
                            alignSelf: 'start',
                        }}>
                            <h3 style={{ margin: '0 0 var(--space-2) 0' }}>{recipeDetail.recipe.name}</h3>
                            <p style={{ margin: '0 0 var(--space-5) 0', color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
                                {recipeDetail.recipe.description}
                            </p>

                            {/* Hardware Selector */}
                            <div style={{ marginBottom: 'var(--space-5)' }}>
                                <label style={{ display: 'block', fontWeight: 500, marginBottom: 'var(--space-2)', fontSize: 'var(--text-sm)' }}>
                                    <Cpu size={14} style={{ display: 'inline', marginRight: 4 }} />
                                    Hardware Profile
                                </label>
                                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                                    {hardwareOptions.map(opt => (
                                        <button
                                            key={opt.value}
                                            onClick={() => setHardware(opt.value)}
                                            className={`filter-chip ${hardware === opt.value ? 'active' : ''}`}
                                            style={{ flex: 1, textAlign: 'center' }}
                                        >
                                            <strong>{opt.label}</strong>
                                            <br />
                                            <span style={{ fontSize: 'var(--text-xs)', opacity: 0.7 }}>{opt.description}</span>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Adjusted Defaults */}
                            <div style={{ marginBottom: 'var(--space-5)' }}>
                                <h4 style={{ margin: '0 0 var(--space-3) 0', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                    <Settings2 size={16} />
                                    Recommended Settings
                                </h4>
                                <div style={{
                                    display: 'grid',
                                    gridTemplateColumns: '1fr 1fr',
                                    gap: 'var(--space-2)',
                                }}>
                                    {[
                                        { label: 'Epochs', value: recipeDetail.adjustedDefaults.epochs },
                                        { label: 'Learning Rate', value: recipeDetail.adjustedDefaults.learningRate },
                                        { label: 'Rank', value: recipeDetail.adjustedDefaults.rank },
                                        { label: 'Batch Size', value: recipeDetail.adjustedDefaults.batchSize },
                                        { label: 'Grad Accum', value: recipeDetail.adjustedDefaults.gradientAccumulationSteps },
                                        { label: 'Seq Length', value: recipeDetail.adjustedDefaults.maxSeqLength },
                                        { label: 'Scheduler', value: recipeDetail.adjustedDefaults.scheduler },
                                        { label: 'PiSSA', value: recipeDetail.adjustedDefaults.usePissa ? 'Yes' : 'No' },
                                    ].map(item => (
                                        <div key={item.label} style={{
                                            background: 'var(--surface-1)',
                                            borderRadius: 'var(--radius-md)',
                                            padding: 'var(--space-2) var(--space-3)',
                                        }}>
                                            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{item.label}</div>
                                            <div style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{String(item.value)}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Supported Models */}
                            <div style={{ marginBottom: 'var(--space-5)' }}>
                                <h4 style={{ margin: '0 0 var(--space-2) 0', fontSize: 'var(--text-sm)' }}>Supported Models</h4>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
                                    {recipeDetail.recipe.supportedModels.map(model => (
                                        <span key={model} style={{ fontSize: 'var(--text-sm)', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                                            {model}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            {/* Eval Suite */}
                            {recipeDetail.recipe.evalSuite.metrics.length > 0 && (
                                <div style={{ marginBottom: 'var(--space-5)' }}>
                                    <h4 style={{ margin: '0 0 var(--space-2) 0', display: 'flex', alignItems: 'center', gap: 'var(--space-2)', fontSize: 'var(--text-sm)' }}>
                                        <Zap size={14} />
                                        Evaluation Metrics
                                    </h4>
                                    <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
                                        {recipeDetail.recipe.evalSuite.metrics.map(metric => (
                                            <span key={metric} className="badge-info" style={{ fontSize: 'var(--text-xs)' }}>
                                                {metric.replace(/_/g, ' ')}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <Link to={`/jobs/new?recipe=${selectedRecipeId}`}>
                                <Button intent="primary" style={{ width: '100%' }}>
                                    Use This Recipe
                                </Button>
                            </Link>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
