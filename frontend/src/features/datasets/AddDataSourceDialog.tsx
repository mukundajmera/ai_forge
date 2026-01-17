import { useState, useCallback } from 'react'
import { GitBranch, Upload, FolderOpen, Search, Filter } from 'lucide-react'
import { Dialog } from '@/components/ui/Dialog'
import { Button } from '@/components/ui/Button'
import { Input, Textarea } from '@/components/ui/Input'
import { Tabs, TabPanel } from '@/components/ui/Tabs'
import { FileUploadZone } from './FileUploadZone'
import { useAddDataSource, useUploadFiles } from './hooks'
import { DEFAULT_INCLUDE_PATTERNS, DEFAULT_EXCLUDE_PATTERNS } from '@/utils/constants'
import type { UploadProgress } from '@/types'
import { generateId } from '@/utils/formatters'

interface AddDataSourceDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onSuccess?: () => void;
}

type TabId = 'git' | 'upload' | 'local';

const tabs = [
    { id: 'git' as const, label: 'Git Repository', icon: <GitBranch size={16} /> },
    { id: 'upload' as const, label: 'Upload Files', icon: <Upload size={16} /> },
    { id: 'local' as const, label: 'Local Folder', icon: <FolderOpen size={16} /> },
]

export function AddDataSourceDialog({ isOpen, onClose, onSuccess }: AddDataSourceDialogProps) {
    const [activeTab, setActiveTab] = useState<TabId>('upload')

    // Git form state
    const [gitUrl, setGitUrl] = useState('')
    const [gitBranch, setGitBranch] = useState('main')
    const [gitName, setGitName] = useState('')

    // Local folder state
    const [localPath, setLocalPath] = useState('')
    const [localName, setLocalName] = useState('')

    // Shared filter state
    const [includePatterns, setIncludePatterns] = useState(DEFAULT_INCLUDE_PATTERNS.join('\n'))
    const [excludePatterns, setExcludePatterns] = useState(DEFAULT_EXCLUDE_PATTERNS.join('\n'))
    const [showFilters, setShowFilters] = useState(false)

    // Upload state
    const [pendingFiles, setPendingFiles] = useState<File[]>([])
    const [uploadProgressList, setUploadProgressList] = useState<UploadProgress[]>([])

    const addDataSourceMutation = useAddDataSource()
    const { upload, isUploading, uploadProgress } = useUploadFiles({
        onSuccess: () => {
            onSuccess?.()
            handleClose()
        }
    })

    const handleClose = () => {
        // Reset state
        setGitUrl('')
        setGitBranch('main')
        setGitName('')
        setLocalPath('')
        setLocalName('')
        setPendingFiles([])
        setUploadProgressList([])
        setShowFilters(false)
        onClose()
    }

    const handleFilesSelected = useCallback((files: File[]) => {
        setPendingFiles(prev => [...prev, ...files])
        setUploadProgressList(prev => [
            ...prev,
            ...files.map(f => ({
                fileId: generateId(),
                filename: f.name,
                size: f.size,
                progress: 0,
                status: 'pending' as const,
            }))
        ])
    }, [])

    const handleRemoveFile = useCallback((fileId: string) => {
        setUploadProgressList(prev => prev.filter(p => p.fileId !== fileId))
        // Also remove from pending files by matching index
        const index = uploadProgressList.findIndex(p => p.fileId === fileId)
        if (index !== -1) {
            setPendingFiles(prev => prev.filter((_, i) => i !== index))
        }
    }, [uploadProgressList])

    const handleSubmit = async () => {
        const config = {
            includePatterns: includePatterns.split('\n').filter(Boolean),
            excludePatterns: excludePatterns.split('\n').filter(Boolean),
            fileTypes: [],
        }

        switch (activeTab) {
            case 'git':
                await addDataSourceMutation.mutateAsync({
                    type: 'git',
                    url: gitUrl,
                    branch: gitBranch || 'main',
                    name: gitName || undefined,
                    config,
                })
                onSuccess?.()
                handleClose()
                break

            case 'upload':
                if (pendingFiles.length > 0) {
                    upload(pendingFiles)
                }
                break

            case 'local':
                await addDataSourceMutation.mutateAsync({
                    type: 'local',
                    path: localPath,
                    name: localName || undefined,
                    config,
                })
                onSuccess?.()
                handleClose()
                break
        }
    }

    const isSubmitDisabled = () => {
        switch (activeTab) {
            case 'git':
                return !gitUrl || addDataSourceMutation.isPending
            case 'upload':
                return pendingFiles.length === 0 || isUploading
            case 'local':
                return !localPath || addDataSourceMutation.isPending
            default:
                return true
        }
    }

    const getSubmitLabel = () => {
        switch (activeTab) {
            case 'git':
                return 'Clone & Parse'
            case 'upload':
                return `Upload ${pendingFiles.length} File${pendingFiles.length !== 1 ? 's' : ''}`
            case 'local':
                return 'Scan & Parse'
            default:
                return 'Submit'
        }
    }

    return (
        <Dialog
            isOpen={isOpen}
            onClose={handleClose}
            title="Add Data Source"
            size="lg"
            footer={
                <>
                    <Button variant="secondary" onClick={handleClose}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleSubmit}
                        disabled={isSubmitDisabled()}
                        loading={addDataSourceMutation.isPending || isUploading}
                    >
                        {getSubmitLabel()}
                    </Button>
                </>
            }
        >
            <Tabs tabs={tabs} activeTab={activeTab} onChange={(id) => setActiveTab(id as TabId)} />

            {/* Git Repository Tab */}
            <TabPanel value="git" activeValue={activeTab}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                    <Input
                        label="Repository URL or Path"
                        placeholder="https://github.com/user/repo or /path/to/local/repo"
                        value={gitUrl}
                        onChange={(e) => setGitUrl(e.target.value)}
                        icon={<GitBranch size={16} />}
                    />

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                        <Input
                            label="Branch"
                            placeholder="main"
                            value={gitBranch}
                            onChange={(e) => setGitBranch(e.target.value)}
                        />
                        <Input
                            label="Name (optional)"
                            placeholder="Auto-detect from URL"
                            value={gitName}
                            onChange={(e) => setGitName(e.target.value)}
                        />
                    </div>

                    <FilterSection
                        showFilters={showFilters}
                        setShowFilters={setShowFilters}
                        includePatterns={includePatterns}
                        setIncludePatterns={setIncludePatterns}
                        excludePatterns={excludePatterns}
                        setExcludePatterns={setExcludePatterns}
                    />
                </div>
            </TabPanel>

            {/* Upload Files Tab */}
            <TabPanel value="upload" activeValue={activeTab}>
                <FileUploadZone
                    onFilesSelected={handleFilesSelected}
                    uploadProgress={isUploading ? uploadProgress : uploadProgressList}
                    onRemoveFile={handleRemoveFile}
                    isUploading={isUploading}
                />
            </TabPanel>

            {/* Local Folder Tab */}
            <TabPanel value="local" activeValue={activeTab}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                    <Input
                        label="Folder Path"
                        placeholder="/path/to/your/project"
                        value={localPath}
                        onChange={(e) => setLocalPath(e.target.value)}
                        icon={<FolderOpen size={16} />}
                        hint="Enter the absolute path to your project folder"
                    />

                    <Input
                        label="Name (optional)"
                        placeholder="Auto-detect from path"
                        value={localName}
                        onChange={(e) => setLocalName(e.target.value)}
                    />

                    <FilterSection
                        showFilters={showFilters}
                        setShowFilters={setShowFilters}
                        includePatterns={includePatterns}
                        setIncludePatterns={setIncludePatterns}
                        excludePatterns={excludePatterns}
                        setExcludePatterns={setExcludePatterns}
                    />
                </div>
            </TabPanel>
        </Dialog>
    )
}

// Filter section component
function FilterSection({
    showFilters,
    setShowFilters,
    includePatterns,
    setIncludePatterns,
    excludePatterns,
    setExcludePatterns,
}: {
    showFilters: boolean;
    setShowFilters: (show: boolean) => void;
    includePatterns: string;
    setIncludePatterns: (patterns: string) => void;
    excludePatterns: string;
    setExcludePatterns: (patterns: string) => void;
}) {
    return (
        <div>
            <button
                className="btn btn-ghost"
                onClick={() => setShowFilters(!showFilters)}
                style={{
                    padding: 'var(--space-2) 0',
                    gap: 'var(--space-2)',
                    color: 'var(--text-secondary)'
                }}
            >
                <Filter size={14} />
                {showFilters ? 'Hide' : 'Show'} File Filters
            </button>

            {showFilters && (
                <div style={{
                    marginTop: 'var(--space-4)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 'var(--space-4)'
                }}>
                    <Textarea
                        label="Include Patterns (one per line)"
                        placeholder="src/**/*.py&#10;lib/**/*.js"
                        value={includePatterns}
                        onChange={(e) => setIncludePatterns(e.target.value)}
                        rows={3}
                    />

                    <Textarea
                        label="Exclude Patterns (one per line)"
                        placeholder="**/node_modules/**&#10;**/__pycache__/**"
                        value={excludePatterns}
                        onChange={(e) => setExcludePatterns(e.target.value)}
                        rows={4}
                    />
                </div>
            )}
        </div>
    )
}
