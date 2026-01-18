import { Dialog } from '@/components/ui/Dialog'
import { useDataSourceFiles } from './hooks'
import { Loader2, FileCode, FileText, AlertCircle } from 'lucide-react'
import { formatBytes } from '@/utils/formatters'
import type { ParsedFile } from '@/types'

interface SourceFilesDialogProps {
    sourceId: string
    sourceName: string
    onClose: () => void
}

export function SourceFilesDialog({ sourceId, sourceName, onClose }: SourceFilesDialogProps) {
    const { data: files, isLoading, error } = useDataSourceFiles(sourceId)

    const getFileIcon = (file: ParsedFile) => {
        if (file.type === 'code') return <FileCode size={16} className="text-blue-400" />
        return <FileText size={16} className="text-gray-400" />
    }

    return (
        <Dialog
            isOpen={true}
            onClose={onClose}
            title={`Files in ${sourceName}`}
            size="lg"
        >
            <div className="max-h-[400px] overflow-y-auto">
                {isLoading ? (
                    <div className="py-12 text-center text-[var(--text-muted)]">
                        <Loader2 className="animate-spin mx-auto mb-2" size={24} />
                        Loading files...
                    </div>
                ) : error ? (
                    <div className="py-8 text-center text-[var(--error-500)]">
                        <AlertCircle className="mx-auto mb-2" size={24} />
                        Failed to load files
                    </div>
                ) : !files || files.length === 0 ? (
                    <div className="py-8 text-center text-[var(--text-muted)]">
                        No files found in this data source.
                    </div>
                ) : (
                    <table className="table w-full">
                        <thead>
                            <tr>
                                <th>File</th>
                                <th>Type</th>
                                <th>Size</th>
                                <th>Chunks</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {files.map((file: ParsedFile) => (
                                <tr key={file.id}>
                                    <td>
                                        <div className="flex items-center gap-2">
                                            {getFileIcon(file)}
                                            <span className="font-mono text-xs truncate max-w-[200px]">
                                                {file.path.split('/').pop()}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="text-xs capitalize">{file.type}</td>
                                    <td className="text-xs">{formatBytes(file.size)}</td>
                                    <td className="text-xs">{file.chunksExtracted}</td>
                                    <td>
                                        <span className={`text-xs px-2 py-0.5 rounded ${file.parseStatus === 'success'
                                            ? 'bg-green-500/20 text-green-400'
                                            : file.parseStatus === 'failed'
                                                ? 'bg-red-500/20 text-red-400'
                                                : 'bg-yellow-500/20 text-yellow-400'
                                            }`}>
                                            {file.parseStatus}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </Dialog>
    )
}
