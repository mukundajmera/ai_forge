import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { ParsingJob } from '@/types'
import { POLLING_INTERVALS } from '@/utils/constants'

interface UseParsingStatusOptions {
    enabled?: boolean;
    onComplete?: (job: ParsingJob) => void;
    onError?: (error: Error) => void;
}

export function useParsingStatus(
    jobId: string | null,
    options: UseParsingStatusOptions = {}
) {
    const { enabled = true, onComplete, onError } = options

    return useQuery({
        queryKey: ['parsing', jobId],
        queryFn: async () => {
            if (!jobId) throw new Error('No job ID');
            const job = await api.get<ParsingJob>(`/parsing/${jobId}`);

            // Call callbacks based on status
            if (job.status === 'complete') {
                onComplete?.(job);
            } else if (job.status === 'failed') {
                onError?.(new Error(job.error || 'Parsing failed'));
            }

            return job;
        },
        enabled: enabled && !!jobId,
        refetchInterval: (query) => {
            const data = query.state.data;
            // Stop polling when complete or failed
            if (data?.status === 'complete' || data?.status === 'failed') {
                return false;
            }
            return POLLING_INTERVALS.parsing;
        },
    })
}
