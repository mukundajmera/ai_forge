import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Dialog } from '@/components/ui/Dialog'
import { Button } from '@/components/ui/Button'
import { useDataSources, useGenerateDataset } from './hooks'
import { Check, Loader2, Database } from 'lucide-react'
import type { DataSource } from '@/types'

const generateSchema = z.object({
    name: z.string().min(3, 'Name must be at least 3 characters'),
    sourceIds: z.array(z.string()).min(1, 'Select at least one data source'),
    format: z.string(),
    questionsPerBlock: z.number().min(1).max(20),
})

type GenerateFormData = z.infer<typeof generateSchema>

interface GenerateDatasetDialogProps {
    open: boolean
    onClose: () => void
}

export function GenerateDatasetDialog({ open, onClose }: GenerateDatasetDialogProps) {
    const { data: sources, isLoading: sourcesLoading } = useDataSources()
    const generateDataset = useGenerateDataset()

    const {
        register,
        handleSubmit,
        setValue,
        watch,
        formState: { errors, isSubmitting },
        reset,
    } = useForm<GenerateFormData>({
        resolver: zodResolver(generateSchema),
        defaultValues: {
            name: '',
            sourceIds: [],
            format: 'alpaca',
            questionsPerBlock: 5,
        },
    })

    const selectedSourceIds = watch('sourceIds')

    const toggleSource = (id: string) => {
        const current = selectedSourceIds
        if (current.includes(id)) {
            setValue('sourceIds', current.filter(s => s !== id))
        } else {
            setValue('sourceIds', [...current, id])
        }
    }

    const onSubmit = async (data: GenerateFormData) => {
        try {
            await generateDataset.mutateAsync(data)
            reset()
            onClose()
        } catch (error) {
            console.error('Failed to generate dataset:', error)
        }
    }

    const readySources = sources?.filter((s: DataSource) => s.status === 'ready') || []

    return (
        <Dialog
            isOpen={open}
            onClose={onClose}
            title="Generate Training Dataset"
            size="lg"
        >
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium mb-1">Dataset Name</label>
                        <input
                            {...register('name')}
                            className="w-full px-3 py-2 bg-[var(--bg-input)] border border-[var(--border-input)] rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--primary-500)]"
                            placeholder="e.g. My Custom Dataset"
                        />
                        {errors.name && (
                            <p className="text-sm text-[var(--error-500)] mt-1">{errors.name.message}</p>
                        )}
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-2">Select Data Sources</label>
                        {sourcesLoading ? (
                            <div className="py-8 text-center text-[var(--text-muted)]">
                                <Loader2 className="animate-spin mx-auto mb-2" />
                                Loading sources...
                            </div>
                        ) : readySources.length === 0 ? (
                            <div className="p-4 border border-dashed border-[var(--border)] rounded-md text-center text-[var(--text-muted)]">
                                No ready data sources found. Add some first.
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[300px] overflow-y-auto p-1">
                                {readySources.map((source: DataSource) => {
                                    const isSelected = selectedSourceIds.includes(source.id)
                                    return (
                                        <div
                                            key={source.id}
                                            onClick={() => toggleSource(source.id)}
                                            className={`
                                                cursor-pointer p-3 rounded-lg border transition-all flex items-center gap-3
                                                ${isSelected
                                                    ? 'border-[var(--primary-500)] bg-[var(--primary-500-10)]'
                                                    : 'border-[var(--border)] hover:border-[var(--border-hover)]'
                                                }
                                            `}
                                        >
                                            <div className={`
                                                w-5 h-5 rounded border flex items-center justify-center
                                                ${isSelected
                                                    ? 'bg-[var(--primary-500)] border-[var(--primary-500)] text-white'
                                                    : 'border-[var(--text-muted)]'
                                                }
                                            `}>
                                                {isSelected && <Check size={12} />}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <div className="font-medium truncate">{source.name}</div>
                                                <div className="text-xs text-[var(--text-muted)] flex gap-2">
                                                    <span>{source.type}</span>
                                                    <span>•</span>
                                                    <span>{source.fileCount} files</span>
                                                </div>
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        )}
                        {errors.sourceIds && (
                            <p className="text-sm text-[var(--error-500)] mt-1">{errors.sourceIds.message}</p>
                        )}
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium mb-1">Format</label>
                            <select
                                {...register('format')}
                                className="w-full px-3 py-2 bg-[var(--bg-input)] border border-[var(--border-input)] rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--primary-500)]"
                            >
                                <option value="alpaca">Alpaca (Instruction)</option>
                                <option value="sharegpt">ShareGPT (Conversational)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1">Questions Per Block</label>
                            <input
                                type="number"
                                {...register('questionsPerBlock', { valueAsNumber: true })}
                                className="w-full px-3 py-2 bg-[var(--bg-input)] border border-[var(--border-input)] rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--primary-500)]"
                            />
                        </div>
                    </div>
                </div>

                <div className="flex justify-end gap-3 pt-4 border-t border-[var(--border)]">
                    <Button type="button" intent="ghost" onClick={onClose}>
                        Cancel
                    </Button>
                    <Button
                        type="submit"
                        disabled={isSubmitting || selectedSourceIds.length === 0}
                        icon={isSubmitting ? <Loader2 className="animate-spin" /> : <Database />}
                    >
                        {isSubmitting ? 'Starting Generation...' : 'Generate Dataset'}
                    </Button>
                </div>
            </form>
        </Dialog>
    )
}
