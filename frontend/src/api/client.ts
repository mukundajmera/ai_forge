const API_BASE = '/api';

type RequestMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

interface RequestOptions {
    method?: RequestMethod;
    body?: unknown;
    headers?: Record<string, string>;
}

class ApiError extends Error {
    constructor(
        message: string,
        public status: number,
        public data?: unknown
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
    const { method = 'GET', body, headers = {} } = options;

    const config: RequestInit = {
        method,
        headers: {
            'Content-Type': 'application/json',
            ...headers,
        },
    };

    if (body && method !== 'GET') {
        config.body = JSON.stringify(body);
    }

    const response = await fetch(`${API_BASE}${endpoint}`, config);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiError(
            errorData.detail || `HTTP error ${response.status}`,
            response.status,
            errorData
        );
    }

    // Handle empty responses
    const text = await response.text();
    if (!text) return {} as T;

    return JSON.parse(text);
}

// File upload helper
async function uploadFiles(
    endpoint: string,
    files: File[],
    onProgress?: (progress: number) => void
): Promise<{ jobId: string; files: Array<{ id: string; filename: string }> }> {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();

        files.forEach(file => {
            formData.append('files', file);
        });

        xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable && onProgress) {
                const progress = Math.round((event.loaded / event.total) * 100);
                onProgress(progress);
            }
        });

        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(JSON.parse(xhr.responseText));
            } else {
                reject(new ApiError('Upload failed', xhr.status));
            }
        });

        xhr.addEventListener('error', () => {
            reject(new ApiError('Network error', 0));
        });

        xhr.open('POST', `${API_BASE}${endpoint}`);
        xhr.send(formData);
    });
}

// Export the API client
export const api = {
    get: <T>(endpoint: string) => request<T>(endpoint, { method: 'GET' }),
    post: <T>(endpoint: string, body?: unknown) => request<T>(endpoint, { method: 'POST', body }),
    put: <T>(endpoint: string, body?: unknown) => request<T>(endpoint, { method: 'PUT', body }),
    patch: <T>(endpoint: string, body?: unknown) => request<T>(endpoint, { method: 'PATCH', body }),
    delete: <T>(endpoint: string) => request<T>(endpoint, { method: 'DELETE' }),
    upload: uploadFiles,
};

export { ApiError };
