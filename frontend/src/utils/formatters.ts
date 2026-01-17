/**
 * Format bytes to human-readable string
 */
export function formatBytes(bytes: number, decimals = 1): string {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

/**
 * Format number with comma separators
 */
export function formatNumber(num: number): string {
    return new Intl.NumberFormat().format(num);
}

/**
 * Format date to relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: string | Date): string {
    const now = new Date();
    const then = new Date(date);
    const diffMs = now.getTime() - then.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffSecs < 60) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return then.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: diffDays > 365 ? 'numeric' : undefined,
    });
}

/**
 * Format date to full date string
 */
export function formatDate(date: string | Date): string {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

/**
 * Format duration in seconds to human-readable
 */
export function formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds}s`;

    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;

    if (mins < 60) {
        return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
    }

    const hours = Math.floor(mins / 60);
    const remainingMins = mins % 60;

    return remainingMins > 0 ? `${hours}h ${remainingMins}m` : `${hours}h`;
}

/**
 * Format quality score with color classification
 */
export function formatQualityScore(score: number): {
    value: string;
    level: 'high' | 'medium' | 'low';
    color: string;
} {
    const level = score >= 0.7 ? 'high' : score >= 0.5 ? 'medium' : 'low';
    const colors = {
        high: 'var(--quality-high)',
        medium: 'var(--quality-medium)',
        low: 'var(--quality-low)',
    };

    return {
        value: score.toFixed(2),
        level,
        color: colors[level],
    };
}

/**
 * Get file extension from filename
 */
export function getFileExtension(filename: string): string {
    const lastDot = filename.lastIndexOf('.');
    return lastDot !== -1 ? filename.slice(lastDot).toLowerCase() : '';
}

/**
 * Get file type from extension
 */
export function getFileType(filename: string): 'code' | 'markdown' | 'text' | 'pdf' | 'unknown' {
    const ext = getFileExtension(filename);

    const codeExts = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.h'];
    const mdExts = ['.md', '.mdx'];
    const textExts = ['.txt'];
    const pdfExts = ['.pdf'];

    if (codeExts.includes(ext)) return 'code';
    if (mdExts.includes(ext)) return 'markdown';
    if (textExts.includes(ext)) return 'text';
    if (pdfExts.includes(ext)) return 'pdf';
    return 'unknown';
}

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number): string {
    if (str.length <= maxLength) return str;
    return str.slice(0, maxLength - 3) + '...';
}

/**
 * Truncate path from the middle
 */
export function truncatePath(path: string, maxLength = 50): string {
    if (path.length <= maxLength) return path;

    const parts = path.split('/');
    if (parts.length <= 2) return truncate(path, maxLength);

    const first = parts[0];
    const last = parts.slice(-2).join('/');

    if (first.length + last.length + 4 > maxLength) {
        return truncate(path, maxLength);
    }

    return `${first}/.../${last}`;
}

/**
 * Pluralize a word based on count
 */
export function pluralize(count: number, singular: string, plural?: string): string {
    return count === 1 ? singular : (plural || `${singular}s`);
}

/**
 * Generate a random ID
 */
export function generateId(): string {
    return Math.random().toString(36).substring(2, 11);
}
