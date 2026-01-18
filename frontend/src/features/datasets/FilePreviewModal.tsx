import { useState } from 'react'
import { Code, FileText, AlertCircle, Ban } from 'lucide-react'
import { Dialog } from '@/components/ui/Dialog'
import { Button } from '@/components/ui/Button'
import { QualityBar } from '@/components/ui/Progress'
import { Badge } from '@/components/ui/Badge'
import { formatBytes } from '@/utils/formatters'
import type { ParsedFile, CodeChunk, FilePreview } from '@/types'

interface FilePreviewModalProps {
    file: ParsedFile | null;
    preview?: FilePreview;
    isOpen: boolean;
    onClose: () => void;
    onExclude?: (fileId: string) => void;
    onReparse?: (fileId: string) => void;
}

export function FilePreviewModal({
    file,
    preview,
    isOpen,
    onClose,
    onExclude,
    onReparse,
}: FilePreviewModalProps) {
    const [activeView, setActiveView] = useState<'preview' | 'chunks'>('preview')

    if (!file) return null

    const qualityLevel = file.qualityScore >= 0.7 ? 'HIGH' : file.qualityScore >= 0.5 ? 'MEDIUM' : 'LOW'

    return (
        <Dialog
            isOpen={isOpen}
            onClose={onClose}
            title=""
            size="xl"
            footer={
                <>
                    {onExclude && (
                        <Button
                            intent="ghost"
                            icon={<Ban size={16} />}
                            onClick={() => {
                                onExclude(file.id)
                                onClose()
                            }}
                        >
                            Exclude File
                        </Button>
                    )}
                    {onReparse && (
                        <Button
                            intent="secondary"
                            onClick={() => onReparse(file.id)}
                        >
                            Re-parse with Different Settings
                        </Button>
                    )}
                    <Button onClick={onClose}>Close</Button>
                </>
            }
        >
            {/* Custom header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-3)',
                marginBottom: 'var(--space-6)'
            }}>
                <span style={{ fontSize: '1.5rem' }}>
                    {file.type === 'code' ? '🐍' : file.type === 'markdown' ? '📄' : '📕'}
                </span>
                <div>
                    <h2 style={{ fontSize: 'var(--text-lg)', fontWeight: 600 }}>{file.filename}</h2>
                    <span style={{
                        fontSize: 'var(--text-sm)',
                        color: 'var(--text-muted)',
                        fontFamily: 'var(--font-mono)'
                    }}>
                        {file.path}
                    </span>
                </div>
            </div>

            {/* Two-column layout */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '280px 1fr',
                gap: 'var(--space-6)',
                minHeight: 400
            }}>
                {/* Left: Metadata */}
                <div className="file-metadata" style={{
                    background: 'var(--bg-elevated)',
                    borderRadius: 'var(--radius-lg)',
                    padding: 'var(--space-4)'
                }}>
                    <h3 style={{
                        fontSize: 'var(--text-sm)',
                        fontWeight: 600,
                        marginBottom: 'var(--space-4)',
                        color: 'var(--text-secondary)'
                    }}>
                        File Metadata
                    </h3>

                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-3)'
                    }}>
                        <MetadataRow label="Language" value={file.language || file.type} />
                        <MetadataRow label="Size" value={formatBytes(file.size)} />
                        <MetadataRow label="Status" value={
                            <Badge variant={file.parseStatus === 'success' ? 'success' : 'danger'}>
                                {file.parseStatus}
                            </Badge>
                        } />

                        {file.metadata && (
                            <>
                                {file.metadata.functions !== undefined && (
                                    <MetadataRow label="Functions" value={file.metadata.functions} />
                                )}
                                {file.metadata.classes !== undefined && (
                                    <MetadataRow label="Classes" value={file.metadata.classes} />
                                )}
                                {file.metadata.docstrings !== undefined && (
                                    <MetadataRow label="Docstrings" value={file.metadata.docstrings} />
                                )}
                                {file.metadata.pages !== undefined && (
                                    <MetadataRow label="Pages" value={file.metadata.pages} />
                                )}
                            </>
                        )}

                        <div style={{ marginTop: 'var(--space-4)' }}>
                            <div style={{
                                fontSize: 'var(--text-xs)',
                                color: 'var(--text-muted)',
                                marginBottom: 'var(--space-2)'
                            }}>
                                Quality Score
                            </div>
                            <QualityBar score={file.qualityScore} segments={10} />
                            <div style={{
                                fontSize: 'var(--text-sm)',
                                fontWeight: 600,
                                marginTop: 'var(--space-2)',
                                color: file.qualityScore >= 0.7
                                    ? 'var(--quality-high)'
                                    : file.qualityScore >= 0.5
                                        ? 'var(--quality-medium)'
                                        : 'var(--quality-low)'
                            }}>
                                {qualityLevel}
                            </div>
                        </div>

                        <div style={{ marginTop: 'var(--space-4)' }}>
                            <div style={{
                                fontSize: 'var(--text-xs)',
                                color: 'var(--text-muted)',
                                marginBottom: 'var(--space-2)'
                            }}>
                                Extracted Chunks
                            </div>
                            <div style={{ fontSize: 'var(--text-2xl)', fontWeight: 700 }}>
                                {file.chunksExtracted}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right: Content Preview */}
                <div className="file-content" style={{
                    display: 'flex',
                    flexDirection: 'column'
                }}>
                    {/* Tabs */}
                    <div className="tabs" style={{ marginBottom: 'var(--space-4)' }}>
                        <button
                            className={`tab ${activeView === 'preview' ? 'active' : ''}`}
                            onClick={() => setActiveView('preview')}
                        >
                            <FileText size={14} />
                            Preview
                        </button>
                        <button
                            className={`tab ${activeView === 'chunks' ? 'active' : ''}`}
                            onClick={() => setActiveView('chunks')}
                        >
                            <Code size={14} />
                            Chunks ({file.chunksExtracted})
                        </button>
                    </div>

                    {/* Content */}
                    <div style={{
                        flex: 1,
                        background: 'var(--bg-elevated)',
                        borderRadius: 'var(--radius-lg)',
                        padding: 'var(--space-4)',
                        overflow: 'auto',
                        maxHeight: 500
                    }}>
                        {activeView === 'preview' ? (
                            <pre style={{
                                fontFamily: 'var(--font-mono)',
                                fontSize: 'var(--text-sm)',
                                lineHeight: 1.6,
                                margin: 0,
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word'
                            }}>
                                {preview?.content || `// Content preview not available\n// File: ${file.filename}`}
                            </pre>
                        ) : (
                            <ChunksList chunks={preview?.chunks || []} />
                        )}
                    </div>
                </div>
            </div>

            {/* Error display */}
            {file.error && (
                <div style={{
                    marginTop: 'var(--space-4)',
                    padding: 'var(--space-3)',
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid var(--error-500)',
                    borderRadius: 'var(--radius-md)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    color: 'var(--error-500)'
                }}>
                    <AlertCircle size={16} />
                    {file.error}
                </div>
            )}
        </Dialog>
    )
}

function MetadataRow({ label, value }: { label: string; value: React.ReactNode }) {
    return (
        <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 'var(--text-sm)'
        }}>
            <span style={{ color: 'var(--text-muted)' }}>{label}</span>
            <span style={{ fontWeight: 500 }}>{value}</span>
        </div>
    )
}

function ChunksList({ chunks }: { chunks: CodeChunk[] }) {
    if (chunks.length === 0) {
        return (
            <div style={{
                textAlign: 'center',
                color: 'var(--text-muted)',
                padding: 'var(--space-8)'
            }}>
                No chunks extracted
            </div>
        )
    }

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--space-3)'
        }}>
            {chunks.map((chunk, index) => (
                <div
                    key={chunk.id}
                    style={{
                        padding: 'var(--space-3)',
                        background: 'var(--bg-overlay)',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--border)'
                    }}
                >
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 'var(--space-2)'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                            <span style={{
                                fontSize: 'var(--text-xs)',
                                color: 'var(--text-muted)',
                                background: 'var(--bg-surface)',
                                padding: '2px 6px',
                                borderRadius: 'var(--radius-sm)'
                            }}>
                                {index + 1}
                            </span>
                            <span style={{ fontWeight: 500 }}>{chunk.name}</span>
                            <Badge variant="info">{chunk.type}</Badge>
                        </div>
                        <QualityBar score={chunk.qualityScore} segments={3} />
                    </div>
                    <div style={{
                        fontSize: 'var(--text-xs)',
                        color: 'var(--text-muted)'
                    }}>
                        {chunk.tokens} tokens
                        {chunk.docstring && ' • has docstring'}
                    </div>
                </div>
            ))}
        </div>
    )
}
