import { useState, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { UploadProgress, ParsingJob } from '@/types'
import { generateId } from '@/utils/formatters'

interface UseUploadFilesOptions {
    onSuccess?: (job: ParsingJob) => void;
    onError?: (error: Error) => void;
}

export function useUploadFiles(options: UseUploadFilesOptions = {}) {
    const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([])
    const [isUploading, setIsUploading] = useState(false)
    const queryClient = useQueryClient()

    const uploadMutation = useMutation({
        mutationFn: async (files: File[]) => {
            setIsUploading(true)

            // Initialize progress for all files
            const initialProgress: UploadProgress[] = files.map(file => ({
                fileId: generateId(),
                filename: file.name,
                size: file.size,
                progress: 0,
                status: 'pending',
            }))
            setUploadProgress(initialProgress)

            // Upload files
            const result = await api.uploadFiles(files, {
                name: 'Upload',
                type: 'upload',
                includePatterns: [],
                excludePatterns: [],
                fileTypes: []
            }, (progress) => {
                setUploadProgress(prev => prev.map(p => ({
                    ...p,
                    progress,
                    status: progress < 100 ? 'uploading' : 'complete',
                })))
            })

            return result
        },
        onSuccess: (data) => {
            setIsUploading(false)
            setUploadProgress(prev => prev.map(p => ({
                ...p,
                status: 'complete',
                progress: 100,
            })))
            queryClient.invalidateQueries({ queryKey: ['data-sources'] })
            options.onSuccess?.(data as unknown as ParsingJob)
        },
        onError: (error: Error) => {
            setIsUploading(false)
            setUploadProgress(prev => prev.map(p => ({
                ...p,
                status: 'error',
                error: error.message,
            })))
            options.onError?.(error)
        },
    })

    const upload = useCallback((files: File[]) => {
        uploadMutation.mutate(files)
    }, [uploadMutation])

    const reset = useCallback(() => {
        setUploadProgress([])
        setIsUploading(false)
    }, [])

    const removeFile = useCallback((fileId: string) => {
        setUploadProgress(prev => prev.filter(p => p.fileId !== fileId))
    }, [])

    return {
        upload,
        reset,
        removeFile,
        uploadProgress,
        isUploading,
        error: uploadMutation.error,
    }
}
