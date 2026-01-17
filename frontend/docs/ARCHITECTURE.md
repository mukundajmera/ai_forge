# AI Forge Architecture

A technical overview of the AI Forge system design, data flow, and implementation details.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Frontend Architecture](#frontend-architecture)
3. [Backend Integration](#backend-integration)
4. [State Management](#state-management)
5. [Real-Time Updates](#real-time-updates)
6. [Component Patterns](#component-patterns)
7. [Testing Architecture](#testing-architecture)
8. [Performance Optimizations](#performance-optimizations)
9. [Security Considerations](#security-considerations)
10. [Deployment Architecture](#deployment-architecture)

---

## System Overview

AI Forge is a full-stack application for local LLM fine-tuning with four main components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AI Forge System                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐       │
│  │  React UI    │      │   FastAPI    │      │   Training   │       │
│  │  (Frontend)  │─────▶│   Backend    │─────▶│   Engine     │       │
│  └──────────────┘      └──────────────┘      └──────────────┘       │
│         │                     │                     │                │
│         │                     │                     ▼                │
│         │                     │              ┌──────────────┐       │
│         │                     │              │   MLX /      │       │
│         │                     │              │   Unsloth    │       │
│         │                     │              └──────────────┘       │
│         │                     │                     │                │
│         │                     ▼                     ▼                │
│         │              ┌──────────────┐      ┌──────────────┐       │
│         │              │  Antigravity │      │    Ollama    │       │
│         └─────────────▶│    Agent     │      │   (Serving)  │       │
│                        └──────────────┘      └──────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **React UI** | User interface, visualization, user interactions |
| **FastAPI Backend** | API, orchestration, data pipeline, job management |
| **Training Engine** | PiSSA/QLoRA training, MLX/Unsloth integration |
| **Antigravity Agent** | Intelligent suggestions, repo monitoring |
| **Ollama** | Model serving, inference |

---

## Frontend Architecture

### Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Framework | React 18 | UI rendering with hooks and concurrent features |
| Language | TypeScript 5 | Type safety across codebase |
| Build | Vite | Fast development server and bundler |
| Styling | CSS + Tailwind | Design system with utility classes |
| State | React Query | Server state management and caching |
| Forms | React Hook Form + Zod | Form handling and validation |
| Charts | Recharts | Data visualization |
| Testing | Vitest + Playwright | Unit, integration, and E2E testing |
| Mocking | MSW | API mocking for tests |

### Directory Structure

```
frontend/
├── src/
│   ├── app/                    # Application shell
│   │   ├── App.tsx            # Root component
│   │   ├── AppRoutes.tsx      # Route definitions
│   │   └── AppLayout.tsx      # Layout wrapper
│   │
│   ├── components/             # Shared components
│   │   ├── ui/                # Design system primitives
│   │   │   ├── Button.tsx
│   │   │   ├── Badge.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Progress.tsx
│   │   │   └── Toast.tsx
│   │   │
│   │   └── layout/            # Layout components
│   │       ├── Sidebar.tsx
│   │       ├── Header.tsx
│   │       └── PageLayout.tsx
│   │
│   ├── features/              # Feature modules
│   │   ├── dashboard/
│   │   │   ├── DashboardPage.tsx
│   │   │   └── components/
│   │   │
│   │   ├── datasets/
│   │   │   ├── DatasetsPage.tsx
│   │   │   └── components/
│   │   │
│   │   ├── jobs/
│   │   │   ├── JobsListPage.tsx
│   │   │   ├── JobDetailPage.tsx
│   │   │   ├── NewJobWizard.tsx
│   │   │   └── components/
│   │   │
│   │   ├── models/
│   │   │   ├── ModelsPage.tsx
│   │   │   └── components/
│   │   │
│   │   └── missions/
│   │       ├── MissionsPage.tsx
│   │       ├── MissionDetailPage.tsx
│   │       └── components/
│   │
│   ├── lib/                   # Utilities and services
│   │   ├── api/
│   │   │   ├── client.ts      # API client class
│   │   │   └── index.ts       # Exports
│   │   │
│   │   ├── hooks/
│   │   │   ├── useJobs.ts
│   │   │   ├── useModels.ts
│   │   │   ├── useMissions.ts
│   │   │   ├── useDatasets.ts
│   │   │   └── useSystem.ts
│   │   │
│   │   ├── types/
│   │   │   ├── index.ts       # All TypeScript interfaces
│   │   │   └── api.ts         # API response types
│   │   │
│   │   └── utils/
│   │       ├── format.ts      # Formatting utilities
│   │       ├── validation.ts  # Zod schemas
│   │       └── test-utils.tsx # Testing utilities
│   │
│   ├── mocks/                 # MSW mock handlers
│   │   ├── handlers.ts
│   │   ├── server.ts          # Node.js server
│   │   └── browser.ts         # Browser worker
│   │
│   ├── styles/                # Global styles
│   │   ├── index.css
│   │   └── tokens.css         # Design tokens
│   │
│   ├── main.tsx               # Entry point
│   └── setupTests.ts          # Test setup
│
├── e2e/                       # Playwright E2E tests
│   ├── fixtures.ts
│   ├── training-flow.spec.ts
│   ├── mission-approval.spec.ts
│   └── navigation.spec.ts
│
├── docs/                      # Documentation
│   ├── USER_GUIDE.md
│   ├── ARCHITECTURE.md
│   └── API.md
│
├── public/                    # Static assets
├── package.json
├── vite.config.ts
├── vitest.config.ts
├── playwright.config.ts
└── tsconfig.json
```

---

## Backend Integration

### API Client Design

The API client provides a type-safe interface to the backend:

```typescript
// lib/api/client.ts

class APIClient {
    private baseUrl: string;
    private abortControllers: Map<string, AbortController>;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
        this.abortControllers = new Map();
    }

    // Generic request method with error handling
    private async request<T>(
        endpoint: string,
        options?: RequestInit
    ): Promise<T> {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options?.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new APIError(response.status, error.message, error.detail);
        }

        return response.json();
    }

    // Type-safe methods for each endpoint
    async getJobs(): Promise<TrainingJob[]> {
        return this.request('/jobs');
    }

    async getJob(id: string): Promise<TrainingJob> {
        return this.request(`/jobs/${id}`);
    }

    async startFineTune(config: FineTuneConfig): Promise<{ jobId: string }> {
        return this.request('/v1/fine-tune', {
            method: 'POST',
            body: JSON.stringify(config),
        });
    }

    // ... more methods
}

// Singleton instance
export const apiClient = new APIClient(
    import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
);
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs` | GET | List all training jobs |
| `/jobs/:id` | GET | Get job details |
| `/jobs/:id/metrics` | GET | Get job metrics |
| `/jobs/:id/logs` | GET | Get job logs |
| `/v1/fine-tune` | POST | Start new training |
| `/jobs/:id` | DELETE | Cancel job |
| `/models` | GET | List all models |
| `/models/:id/deploy` | POST | Deploy model to Ollama |
| `/missions` | GET | List all missions |
| `/missions/:id/approve` | POST | Approve mission |
| `/missions/:id/reject` | POST | Reject mission |
| `/datasets` | GET | List datasets |
| `/data-sources` | GET | List data sources |
| `/status` | GET | Get system status |
| `/health` | GET | Health check |

### Error Handling

```typescript
// Backend error response format
interface APIErrorResponse {
    status: number;
    message: string;
    detail?: Record<string, unknown>;
}

// Frontend error class
class APIError extends Error {
    status: number;
    detail?: Record<string, unknown>;

    constructor(status: number, message: string, detail?: Record<string, unknown>) {
        super(message);
        this.status = status;
        this.detail = detail;
    }
}
```

---

## State Management

### Server State (React Query)

All server data is managed through React Query hooks:

```typescript
// lib/hooks/useJobs.ts

export function useJobs() {
    return useQuery({
        queryKey: ['jobs'],
        queryFn: () => apiClient.getJobs(),
        refetchInterval: 5000,  // Poll every 5s
        staleTime: 3000,
    });
}

export function useJob(id: string | undefined) {
    return useQuery({
        queryKey: ['jobs', id],
        queryFn: () => apiClient.getJob(id!),
        enabled: !!id,
        refetchInterval: (query) => {
            // Poll more frequently for running jobs
            return query.state.data?.status === 'running' ? 3000 : 10000;
        },
    });
}

export function useStartFineTune() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (config: FineTuneConfig) => 
            apiClient.startFineTune(config),
        onSuccess: () => {
            // Invalidate jobs cache to refresh list
            queryClient.invalidateQueries({ queryKey: ['jobs'] });
        },
    });
}
```

### Query Key Structure

```typescript
// Hierarchical query keys for cache management
const queryKeys = {
    jobs: ['jobs'] as const,
    job: (id: string) => ['jobs', id] as const,
    jobMetrics: (id: string) => ['jobs', id, 'metrics'] as const,
    jobLogs: (id: string) => ['jobs', id, 'logs'] as const,

    models: ['models'] as const,
    activeModel: ['models', 'active'] as const,

    missions: ['missions'] as const,
    mission: (id: string) => ['missions', id] as const,

    datasets: ['datasets'] as const,
    dataSources: ['dataSources'] as const,

    system: ['system'] as const,
    health: ['system', 'health'] as const,
};
```

### Client State

Local UI state uses React hooks:

```typescript
// Modal state
const [isOpen, setIsOpen] = useState(false);

// Form state with React Hook Form
const form = useForm<FineTuneFormData>({
    resolver: zodResolver(fineTuneSchema),
    defaultValues: {
        projectName: '',
        baseModel: 'Llama-3.2-3B',
        epochs: 3,
    },
});

// Filter state
const [statusFilter, setStatusFilter] = useState<JobStatus | 'all'>('all');
const [searchQuery, setSearchQuery] = useState('');
```

---

## Real-Time Updates

### Polling Strategy

| Data Type | Interval | Condition |
|-----------|----------|-----------|
| Jobs List | 5s | Always when on page |
| Job Detail (running) | 3s | When job.status === 'running' |
| Job Detail (completed) | 10s | When job is not running |
| System Status | 3s | Always |
| Missions | 10s | Always |
| Models | 10s | Always |

### Log Streaming (Future)

For real-time logs, use Server-Sent Events:

```typescript
// Future implementation
function useJobLogs(jobId: string) {
    useEffect(() => {
        const eventSource = new EventSource(
            `${API_BASE}/jobs/${jobId}/logs/stream`
        );

        eventSource.onmessage = (event) => {
            const log = JSON.parse(event.data);
            setLogs(prev => [...prev, log]);
        };

        eventSource.onerror = () => {
            eventSource.close();
        };

        return () => eventSource.close();
    }, [jobId]);
}
```

---

## Component Patterns

### Container/Presentational Pattern

```typescript
// Container (Page) - handles data fetching
function JobsListPage() {
    const { data: jobs, isLoading, error } = useJobs();
    const [filter, setFilter] = useState('all');

    const filteredJobs = useMemo(() => 
        filterJobs(jobs, filter),
        [jobs, filter]
    );

    return (
        <PageLayout title="Training Jobs">
            <JobsFilter value={filter} onChange={setFilter} />
            <JobsTable jobs={filteredJobs} loading={isLoading} />
        </PageLayout>
    );
}

// Presentational - pure render
function JobsTable({ jobs, loading }: JobsTableProps) {
    if (loading) return <TableSkeleton />;

    return (
        <Table>
            {jobs.map(job => (
                <JobRow key={job.id} job={job} />
            ))}
        </Table>
    );
}
```

### Compound Components

```typescript
// Compound component pattern for complex UI
<Dialog>
    <DialogHeader>
        <DialogTitle>Deploy Model</DialogTitle>
        <DialogClose />
    </DialogHeader>
    <DialogContent>
        <MetricComparison current={currentModel} new={newModel} />
    </DialogContent>
    <DialogActions>
        <Button intent="ghost" onClick={onCancel}>Cancel</Button>
        <Button intent="primary" onClick={onDeploy}>Deploy</Button>
    </DialogActions>
</Dialog>
```

### Custom Hooks

```typescript
// Encapsulate complex logic in hooks
function useJobActions(jobId: string) {
    const cancelMutation = useCancelJob();
    const exportMutation = useExportModel();

    const canCancel = useCallback((job: TrainingJob) => 
        job.status === 'running' || job.status === 'queued',
        []
    );

    const canExport = useCallback((job: TrainingJob) => 
        job.status === 'completed',
        []
    );

    return {
        cancel: () => cancelMutation.mutate(jobId),
        export: () => exportMutation.mutate(jobId),
        canCancel,
        canExport,
        isCancelling: cancelMutation.isPending,
        isExporting: exportMutation.isPending,
    };
}
```

---

## Testing Architecture

### Test Pyramid

```
        ╱╲
       ╱  ╲           E2E Tests (10%)
      ╱────╲          - Critical user flows
     ╱      ╲         - Real browser
    ╱────────╲
   ╱          ╲       Integration Tests (30%)
  ╱────────────╲      - Page components
 ╱              ╲     - API mocking with MSW
╱────────────────╲
                      Unit Tests (60%)
                      - Components
                      - Hooks
                      - Utilities
```

### Test Configuration

```typescript
// vitest.config.ts
export default defineConfig({
    test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: ['./src/setupTests.ts'],
        include: ['src/**/*.{test,spec}.{ts,tsx}'],
        coverage: {
            provider: 'v8',
            reporter: ['text', 'html', 'lcov'],
            thresholds: {
                lines: 80,
                functions: 80,
                branches: 75,
                statements: 80,
            },
        },
    },
});
```

### MSW Handlers

```typescript
// mocks/handlers.ts
export const handlers = [
    http.get('/jobs', () => {
        return HttpResponse.json(mockJobs);
    }),

    http.post('/v1/fine-tune', async ({ request }) => {
        const body = await request.json();
        return HttpResponse.json({ jobId: 'job-new' }, { status: 201 });
    }),

    // Error scenario handlers for testing
    http.get('/jobs/error', () => {
        return HttpResponse.json(
            { message: 'Server error' },
            { status: 500 }
        );
    }),
];
```

### Test Utilities

```typescript
// utils/test-utils.tsx
export function renderWithProviders(
    ui: ReactElement,
    { initialEntries = ['/'], queryClient = createTestQueryClient() } = {}
) {
    return {
        ...render(
            <QueryClientProvider client={queryClient}>
                <MemoryRouter initialEntries={initialEntries}>
                    {ui}
                </MemoryRouter>
            </QueryClientProvider>
        ),
        queryClient,
    };
}
```

---

## Performance Optimizations

### Code Splitting

```typescript
// Route-based code splitting
const JobsListPage = lazy(() => import('./features/jobs/JobsListPage'));
const JobDetailPage = lazy(() => import('./features/jobs/JobDetailPage'));
const MissionsPage = lazy(() => import('./features/missions/MissionsPage'));

function AppRoutes() {
    return (
        <Suspense fallback={<PageSkeleton />}>
            <Routes>
                <Route path="/jobs" element={<JobsListPage />} />
                <Route path="/jobs/:id" element={<JobDetailPage />} />
                <Route path="/missions" element={<MissionsPage />} />
            </Routes>
        </Suspense>
    );
}
```

### Memoization

```typescript
// Memoize expensive computations
const filteredJobs = useMemo(() => {
    return jobs
        .filter(job => statusFilter === 'all' || job.status === statusFilter)
        .filter(job => job.projectName.includes(searchQuery));
}, [jobs, statusFilter, searchQuery]);

// Memoize components that receive object props
const JobRow = memo(function JobRow({ job }: { job: TrainingJob }) {
    return <tr>...</tr>;
});
```

### Virtualization

```typescript
// For large lists, use virtualization
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualizedJobsList({ jobs }: { jobs: TrainingJob[] }) {
    const parentRef = useRef<HTMLDivElement>(null);

    const virtualizer = useVirtualizer({
        count: jobs.length,
        getScrollElement: () => parentRef.current,
        estimateSize: () => 64,
    });

    return (
        <div ref={parentRef} style={{ height: 600, overflow: 'auto' }}>
            <div style={{ height: virtualizer.getTotalSize() }}>
                {virtualizer.getVirtualItems().map(virtualRow => (
                    <JobRow key={jobs[virtualRow.index].id} 
                            job={jobs[virtualRow.index]} />
                ))}
            </div>
        </div>
    );
}
```

---

## Security Considerations

### XSS Prevention

- React escapes content by default
- Use `DOMPurify` for any user-generated HTML
- Never use `dangerouslySetInnerHTML` with unsanitized content

### API Security

- CORS configured on backend
- HTTPS in production
- Input validation with Zod schemas

### Environment Variables

- Secrets never in code
- `.env` files in `.gitignore`
- Environment-specific configs

---

## Deployment Architecture

### Development

```
npm run dev
    │
    ▼
Vite Dev Server (localhost:5173)
    │
    ├── Hot Module Replacement
    └── Proxy to Backend API
              │
              ▼
        FastAPI (localhost:8000)
```

### Production

```
npm run build
    │
    ▼
dist/
    │
    ▼
Static Hosting (Vercel / Netlify / S3 + CloudFront)
              │
              ▼
        FastAPI Backend (Docker / K8s)
```

### Environment Configuration

| Environment | API URL | Features |
|-------------|---------|----------|
| Development | `localhost:8000` | HMR, MSW mocks available |
| Staging | `api.staging.example.com` | Full backend, test data |
| Production | `api.example.com` | Full backend, real data |

---

## Future Enhancements

### Planned Features

- **Multi-user authentication**: JWT-based auth
- **Team collaboration**: Shared jobs, models, datasets
- **Experiment tracking**: MLflow-style comparison
- **Advanced visualizations**: Embeddings, attention maps
- **Mobile app**: React Native version

### Technical Improvements

- **WebSocket support**: Real-time log streaming
- **Offline support**: Service worker caching
- **Performance monitoring**: Core Web Vitals tracking
- **Accessibility audit**: WCAG 2.1 AAA compliance
