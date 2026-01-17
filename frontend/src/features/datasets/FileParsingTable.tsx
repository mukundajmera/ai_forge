import { CheckCircle, XCircle, Clock, Loader } from 'lucide-react'
import { QualityBar } from '@/components/ui/Progress'
import { TableSkeleton } from '@/components/ui/EmptyState'
import { formatBytes } from '@/utils/formatters'
import { FILE_TYPE_ICONS } from '@/utils/constants'
import type { ParsedFile } from '@/types'
import clsx from 'clsx'

interface FileParsingTableProps {
    files: ParsedFile[];
    isLoading?: boolean;
    onFileClick?: (file: ParsedFile) => void;
}

export function FileParsingTable({
    files,
    isLoading = false,
    onFileClick
}: FileParsingTableProps) {
    const getStatusIcon = (status: ParsedFile['parseStatus']) => {
        switch (status) {
            case 'success':
                return <CheckCircle size={16} className="status-icon success" />
            case 'failed':
                return <XCircle size={16} className="status-icon error" />
            case 'parsing':
                return <Loader size={16} className="status-icon parsing spinning" />
            case 'pending':
                return <Clock size={16} className="status-icon pending" />
        }
    }

    const getFileIcon = (file: ParsedFile) => {
        // Try to get icon from extension
        const ext = '.' + file.filename.split('.').pop()?.toLowerCase()
        return FILE_TYPE_ICONS[ext] || '📄'
    }

    const getLanguageLabel = (file: ParsedFile) => {
        if (file.type === 'code' && file.language) {
            return file.language.charAt(0).toUpperCase() + file.language.slice(1)
        }
        return file.type.charAt(0).toUpperCase() + file.type.slice(1)
    }

    if (isLoading) {
        return <TableSkeleton rows={5} cols={7} />
    }

    const completedCount = files.filter(f => f.parseStatus === 'success').length
    const failedCount = files.filter(f => f.parseStatus === 'failed').length
    const totalChunks = files.reduce((acc, f) => acc + f.chunksExtracted, 0)
    const avgQuality = files.length > 0
        ? files.reduce((acc, f) => acc + f.qualityScore, 0) / files.length
        : 0

    return (
        <div className="file-parsing-table">
            {/* Summary header */}
            <div className="parsing-summary" style={{
                display: 'flex',
                gap: 'var(--space-6)',
                padding: 'var(--space-4)',
                background: 'var(--bg-surface)',
                borderRadius: 'var(--radius-lg)',
                marginBottom: 'var(--space-4)',
                border: '1px solid var(--border)'
            }}>
                <div className="summary-stat">
                    <span className="stat-value">{files.length}</span>
                    <span className="stat-label">Files</span>
                </div>
                <div className="summary-stat">
                    <span className="stat-value" style={{ color: 'var(--success-500)' }}>
                        {completedCount}
                    </span>
                    <span className="stat-label">Parsed</span>
                </div>
                {failedCount > 0 && (
                    <div className="summary-stat">
                        <span className="stat-value" style={{ color: 'var(--error-500)' }}>
                            {failedCount}
                        </span>
                        <span className="stat-label">Failed</span>
                    </div>
                )}
                <div className="summary-stat">
                    <span className="stat-value">{totalChunks}</span>
                    <span className="stat-label">Chunks</span>
                </div>
                <div className="summary-stat">
                    <QualityBar score={avgQuality} showValue />
                    <span className="stat-label">Avg Quality</span>
                </div>
            </div>

            {/* Table */}
            <div className="table-container">
                <table className="table">
                    <thead>
                        <tr>
                            <th>Filename</th>
                            <th>Type</th>
                            <th>Size</th>
                            <th>Status</th>
                            <th>Chunks</th>
                            <th>Quality</th>
                        </tr>
                    </thead>
                    <tbody>
                        {files.map((file) => (
                            <tr
                                key={file.id}
                                onClick={() => onFileClick?.(file)}
                                style={{ cursor: onFileClick ? 'pointer' : 'default' }}
                                className={clsx(file.parseStatus === 'failed' && 'row-error')}
                            >
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                        <span style={{ fontSize: '1.25rem' }}>{getFileIcon(file)}</span>
                                        <div>
                                            <div style={{ fontWeight: 500 }}>{file.filename}</div>
                                            {file.path !== file.filename && (
                                                <div style={{
                                                    fontSize: 'var(--text-xs)',
                                                    color: 'var(--text-muted)',
                                                    fontFamily: 'var(--font-mono)'
                                                }}>
                                                    {file.path}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <span style={{ color: 'var(--text-secondary)' }}>
                                        {getLanguageLabel(file)}
                                    </span>
                                </td>
                                <td style={{ color: 'var(--text-secondary)' }}>
                                    {formatBytes(file.size)}
                                </td>
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                                        {getStatusIcon(file.parseStatus)}
                                        {file.parseStatus === 'failed' && file.error && (
                                            <span
                                                className="tooltip"
                                                data-tooltip={file.error}
                                                style={{
                                                    fontSize: 'var(--text-xs)',
                                                    color: 'var(--error-500)',
                                                    maxWidth: 200,
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                    whiteSpace: 'nowrap'
                                                }}
                                            >
                                                {file.error}
                                            </span>
                                        )}
                                    </div>
                                </td>
                                <td>
                                    {file.parseStatus === 'success' ? (
                                        <span>{file.chunksExtracted}</span>
                                    ) : (
                                        <span style={{ color: 'var(--text-muted)' }}>-</span>
                                    )}
                                </td>
                                <td>
                                    {file.parseStatus === 'success' ? (
                                        <QualityBar score={file.qualityScore} segments={5} />
                                    ) : (
                                        <span style={{ color: 'var(--text-muted)' }}>-</span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <style>{`
        .summary-stat {
          display: flex;
          flex-direction: column;
          gap: var(--space-1);
        }
        
        .stat-value {
          font-size: var(--text-xl);
          font-weight: 700;
        }
        
        .stat-label {
          font-size: var(--text-xs);
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        
        .status-icon.success {
          color: var(--success-500);
        }
        
        .status-icon.error {
          color: var(--error-500);
        }
        
        .status-icon.parsing {
          color: var(--info-500);
        }
        
        .status-icon.pending {
          color: var(--text-muted);
        }
        
        .spinning {
          animation: spin 1s linear infinite;
        }
        
        .row-error {
          background-color: rgba(239, 68, 68, 0.05);
        }
        
        .row-error:hover td {
          background-color: rgba(239, 68, 68, 0.1) !important;
        }
      `}</style>
        </div>
    )
}
