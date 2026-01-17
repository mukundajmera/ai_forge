// API base URL
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// File upload limits
export const UPLOAD_LIMITS = {
    maxFiles: 100,
    maxTotalSize: 500 * 1024 * 1024, // 500MB
    maxFileSize: 50 * 1024 * 1024, // 50MB per file
};

// Supported file types
export const SUPPORTED_FILE_TYPES = {
    code: ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.h'],
    markdown: ['.md', '.mdx'],
    text: ['.txt'],
    pdf: ['.pdf'],
};

export const ALL_SUPPORTED_EXTENSIONS = [
    ...SUPPORTED_FILE_TYPES.code,
    ...SUPPORTED_FILE_TYPES.markdown,
    ...SUPPORTED_FILE_TYPES.text,
    ...SUPPORTED_FILE_TYPES.pdf,
];

// File type icons (using emoji as fallback)
export const FILE_TYPE_ICONS: Record<string, string> = {
    '.py': '🐍',
    '.js': '📜',
    '.jsx': '⚛️',
    '.ts': '📘',
    '.tsx': '⚛️',
    '.java': '☕',
    '.go': '🐹',
    '.rs': '🦀',
    '.cpp': '⚙️',
    '.c': '⚙️',
    '.h': '📋',
    '.md': '📄',
    '.mdx': '📄',
    '.txt': '📝',
    '.pdf': '📕',
};

// Language display names
export const LANGUAGE_NAMES: Record<string, string> = {
    python: 'Python',
    javascript: 'JavaScript',
    typescript: 'TypeScript',
    java: 'Java',
    go: 'Go',
    rust: 'Rust',
    cpp: 'C++',
    c: 'C',
};

// Status colors
export const STATUS_COLORS = {
    syncing: 'info',
    parsing: 'info',
    ready: 'success',
    error: 'error',
    pending: 'neutral',
    generating: 'info',
    complete: 'success',
    failed: 'error',
} as const;

// Quality thresholds
export const QUALITY_THRESHOLDS = {
    high: 0.7,
    medium: 0.5,
    low: 0, // anything below medium
};

// Validation warnings
export const VALIDATION_WARNINGS = {
    smallDataset: {
        threshold: 100,
        message: '⚠️ Small dataset may cause overfitting. Recommend 500+ examples.',
        severity: 'warning' as const,
    },
    lowQuality: {
        threshold: 0.5,
        message: '⚠️ Low quality data. Consider adding more documentation or examples.',
        severity: 'warning' as const,
    },
    noDocstrings: {
        message: '⚠️ No docstrings found. Model may lack context. Add code comments.',
        severity: 'warning' as const,
    },
    singleFileType: {
        message: '💡 Tip: Include .md docs for better context understanding.',
        severity: 'info' as const,
    },
    largeDataset: {
        threshold: 10000,
        message: '💡 Large dataset detected. Training may take 2-3 hours.',
        severity: 'info' as const,
    },
};

// Polling intervals (ms)
export const POLLING_INTERVALS = {
    parsing: 2000,
    generation: 2000,
    sync: 3000,
};

// Default filter patterns
export const DEFAULT_INCLUDE_PATTERNS = ['src/**/*', 'lib/**/*', 'docs/**/*'];
export const DEFAULT_EXCLUDE_PATTERNS = [
    '**/node_modules/**',
    '**/__pycache__/**',
    '**/venv/**',
    '**/.venv/**',
    '**/.git/**',
    '**/dist/**',
    '**/build/**',
    '**/*.min.js',
    '**/coverage/**',
    '**/.next/**',
];
