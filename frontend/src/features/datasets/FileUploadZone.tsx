import { useCallback, useRef, useState, DragEvent } from 'react'
import { Upload, File, X, AlertCircle, CheckCircle } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Progress } from '@/components/ui/Progress'
import { formatBytes, getFileExtension, getFileType } from '@/utils/formatters'
import { ALL_SUPPORTED_EXTENSIONS, UPLOAD_LIMITS, FILE_TYPE_ICONS } from '@/utils/constants'
import type { UploadProgress } from '@/types'
import clsx from 'clsx'

interface FileUploadZoneProps {
    onFilesSelected: (files: File[]) => void;
    uploadProgress?: UploadProgress[];
    onRemoveFile?: (fileId: string) => void;
    isUploading?: boolean;
    maxFiles?: number;
    maxTotalSize?: number;
}

export function FileUploadZone({
    onFilesSelected,
    uploadProgress = [],
    onRemoveFile,
    isUploading = false,
    maxFiles = UPLOAD_LIMITS.maxFiles,
    maxTotalSize = UPLOAD_LIMITS.maxTotalSize,
}: FileUploadZoneProps) {
    const [isDragging, setIsDragging] = useState(false)
    const [validationError, setValidationError] = useState<string | null>(null)
    const inputRef = useRef<HTMLInputElement>(null)

    const validateFiles = useCallback((files: File[]): { valid: File[]; errors: string[] } => {
        const valid: File[] = []
        const errors: string[] = []
        let totalSize = uploadProgress.reduce((acc, p) => acc + p.size, 0)

        for (const file of files) {
            const ext = getFileExtension(file.name)

            // Check extension
            if (!ALL_SUPPORTED_EXTENSIONS.includes(ext)) {
                errors.push(`${file.name}: Unsupported file type`)
                continue
            }

            // Check individual file size
            if (file.size > UPLOAD_LIMITS.maxFileSize) {
                errors.push(`${file.name}: File too large (max ${formatBytes(UPLOAD_LIMITS.maxFileSize)})`)
                continue
            }

            // Check total size
            if (totalSize + file.size > maxTotalSize) {
                errors.push(`${file.name}: Would exceed total size limit`)
                continue
            }

            // Check file count
            if (uploadProgress.length + valid.length >= maxFiles) {
                errors.push(`Maximum ${maxFiles} files allowed`)
                break
            }

            valid.push(file)
            totalSize += file.size
        }

        return { valid, errors }
    }, [uploadProgress, maxFiles, maxTotalSize])

    const handleFiles = useCallback((files: FileList | File[]) => {
        const fileArray = Array.from(files)
        const { valid, errors } = validateFiles(fileArray)

        if (errors.length > 0) {
            setValidationError(errors[0])
            setTimeout(() => setValidationError(null), 5000)
        }

        if (valid.length > 0) {
            onFilesSelected(valid)
        }
    }, [validateFiles, onFilesSelected])

    const handleDragOver = useCallback((e: DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(true)
    }, [])

    const handleDragLeave = useCallback((e: DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)
    }, [])

    const handleDrop = useCallback((e: DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)

        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files)
        }
    }, [handleFiles])

    const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFiles(e.target.files)
        }
        // Reset input so same file can be selected again
        e.target.value = ''
    }, [handleFiles])

    const getStatusIcon = (status: UploadProgress['status']) => {
        switch (status) {
            case 'complete':
                return <CheckCircle size={16} className="status-icon success" />
            case 'error':
                return <AlertCircle size={16} className="status-icon error" />
            case 'uploading':
                return <span className="spinner spinner-sm" />
            default:
                return null
        }
    }

    const getFileIcon = (filename: string) => {
        const ext = getFileExtension(filename)
        return FILE_TYPE_ICONS[ext] || '📄'
    }

    return (
        <div className="file-upload-zone">
            {/* Drop zone */}
            <div
                className={clsx('drop-zone', isDragging && 'active')}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
            >
                <Upload className="drop-zone-icon" size={48} />
                <h4 style={{ marginBottom: 'var(--space-2)' }}>
                    Drop files here or click to browse
                </h4>
                <p style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
                    Supports: .py, .js, .ts, .java, .go, .rs, .md, .txt, .pdf
                </p>
                <p style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)', marginTop: 'var(--space-2)' }}>
                    Max {maxFiles} files, {formatBytes(maxTotalSize)} total
                </p>

                <input
                    ref={inputRef}
                    type="file"
                    multiple
                    accept={ALL_SUPPORTED_EXTENSIONS.join(',')}
                    onChange={handleInputChange}
                    style={{ display: 'none' }}
                />
            </div>

            {/* Validation error */}
            {validationError && (
                <div className="validation-error" style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-2)',
                    padding: 'var(--space-3)',
                    marginTop: 'var(--space-4)',
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid var(--error-500)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--error-500)',
                    fontSize: 'var(--text-sm)'
                }}>
                    <AlertCircle size={16} />
                    {validationError}
                </div>
            )}

            {/* Selected files list */}
            {uploadProgress.length > 0 && (
                <div className="selected-files" style={{ marginTop: 'var(--space-6)' }}>
                    <h4 style={{
                        fontSize: 'var(--text-sm)',
                        marginBottom: 'var(--space-3)',
                        color: 'var(--text-secondary)'
                    }}>
                        Selected Files ({uploadProgress.length})
                    </h4>

                    <div className="file-list" style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-2)',
                        maxHeight: 300,
                        overflowY: 'auto'
                    }}>
                        {uploadProgress.map(file => (
                            <div
                                key={file.fileId}
                                className="file-item"
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 'var(--space-3)',
                                    padding: 'var(--space-3)',
                                    background: 'var(--bg-elevated)',
                                    borderRadius: 'var(--radius-md)',
                                    border: '1px solid var(--border)'
                                }}
                            >
                                <span style={{ fontSize: '1.25rem' }}>
                                    {getFileIcon(file.filename)}
                                </span>

                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        marginBottom: file.status === 'uploading' ? 'var(--space-1)' : 0
                                    }}>
                                        <span style={{
                                            fontSize: 'var(--text-sm)',
                                            overflow: 'hidden',
                                            textOverflow: 'ellipsis',
                                            whiteSpace: 'nowrap'
                                        }}>
                                            {file.filename}
                                        </span>
                                        <span style={{
                                            fontSize: 'var(--text-xs)',
                                            color: 'var(--text-muted)',
                                            marginLeft: 'var(--space-2)'
                                        }}>
                                            {formatBytes(file.size)}
                                        </span>
                                    </div>

                                    {file.status === 'uploading' && (
                                        <Progress value={file.progress} size="sm" />
                                    )}

                                    {file.error && (
                                        <span style={{
                                            fontSize: 'var(--text-xs)',
                                            color: 'var(--error-500)'
                                        }}>
                                            {file.error}
                                        </span>
                                    )}
                                </div>

                                {getStatusIcon(file.status)}

                                {!isUploading && onRemoveFile && (
                                    <button
                                        className="btn btn-ghost btn-icon"
                                        onClick={() => onRemoveFile(file.fileId)}
                                        style={{ padding: 'var(--space-1)' }}
                                    >
                                        <X size={14} />
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <style>{`
        .status-icon.success {
          color: var(--success-500);
        }
        .status-icon.error {
          color: var(--error-500);
        }
      `}</style>
        </div>
    )
}
