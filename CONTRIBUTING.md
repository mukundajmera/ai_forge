# Contributing to AI Forge

Thank you for your interest in contributing to AI Forge! This guide will help you get started with development and ensure your contributions meet our quality standards.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Development Setup](#development-setup)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Getting Help](#getting-help)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please read our expectations:

- **Be Respectful**: Treat everyone with respect and consideration
- **Be Constructive**: Provide helpful, actionable feedback
- **Be Inclusive**: Welcome newcomers and help them learn
- **Be Professional**: Focus on the work, not the person

---

## Development Setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Node.js | 18 or higher |
| npm | 9 or higher |
| Git | 2.30+ |

### Recommended Tools

- **Editor**: VS Code
- **Terminal**: iTerm2 (macOS), Windows Terminal (Windows)
- **Browser**: Chrome with React DevTools

### VS Code Extensions

Install these extensions for the best experience:

- ESLint
- Prettier - Code formatter
- Tailwind CSS IntelliSense
- TypeScript Vue Plugin (Volar)
- GitLens
- Error Lens

### Initial Setup

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-forge.git
cd ai-forge/frontend

# 3. Add upstream remote
git remote add upstream https://github.com/yourorg/ai-forge.git

# 4. Install dependencies
npm install

# 5. Copy environment file
cp .env.example .env

# 6. Start development server
npm run dev
```

### Verify Setup

```bash
# Run all checks
npm run lint
npm run type-check
npm test

# If all pass, you're ready to contribute!
```

---

## Development Workflow

### Branch Strategy

```
main ─────────────────────────────────────────▶
  │
  └─ develop ─────────────────────────────────▶
       │
       ├─ feature/add-export-dialog
       │
       ├─ fix/memory-leak-in-chart
       │
       └─ docs/update-api-reference
```

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code, always stable |
| `develop` | Integration branch for next release |
| `feature/*` | New features and enhancements |
| `fix/*` | Bug fixes |
| `docs/*` | Documentation updates |
| `refactor/*` | Code refactoring |

### Making Changes

1. **Create a branch**
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/my-feature
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Run checks locally**
   ```bash
   npm run lint
   npm run type-check
   npm test
   ```

4. **Commit with conventional format**
   ```bash
   git add .
   git commit -m "feat: add model export dialog"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/my-feature
   ```

6. **Open a Pull Request**
   - Go to GitHub
   - Click "Compare & pull request"
   - Fill out the PR template
   - Request review

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, missing semicolons, etc. |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |

**Examples:**

```bash
# Feature
git commit -m "feat(jobs): add job cancellation button"

# Bug fix
git commit -m "fix(charts): resolve memory leak in loss curve"

# Documentation
git commit -m "docs: update API reference for missions"

# With body
git commit -m "feat(datasets): add drag-and-drop upload

Implements HTML5 drag-and-drop API for file uploads.
Supports multiple files and folder uploads.

Closes #123"
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your branch
git checkout develop
git merge upstream/develop

# Rebase your feature branch
git checkout feature/my-feature
git rebase develop
```

---

## Code Standards

### TypeScript

```typescript
// ✅ Good: Explicit types
interface JobProps {
    job: TrainingJob;
    onCancel: (id: string) => void;
}

// ❌ Bad: Using any
function processData(data: any) { ... }

// ✅ Good: Use unknown and type guards
function processData(data: unknown) {
    if (isTrainingJob(data)) {
        return data.id;
    }
}

// ✅ Good: Interface for props
interface ButtonProps {
    children: React.ReactNode;
    onClick: () => void;
    disabled?: boolean;
}

// ✅ Good: Type for unions
type JobStatus = 'queued' | 'running' | 'completed' | 'failed';

// ✅ Good: Explicit return types for functions
function calculateProgress(current: number, total: number): number {
    return Math.round((current / total) * 100);
}
```

### React Components

```typescript
// ✅ Good: Functional component with typed props
interface JobCardProps {
    job: TrainingJob;
    selected?: boolean;
    onSelect: (id: string) => void;
}

export function JobCard({ job, selected = false, onSelect }: JobCardProps) {
    const handleClick = useCallback(() => {
        onSelect(job.id);
    }, [job.id, onSelect]);

    return (
        <Card 
            className={clsx('job-card', selected && 'job-card--selected')}
            onClick={handleClick}
        >
            <JobCardHeader job={job} />
            <JobCardBody job={job} />
        </Card>
    );
}

// ✅ Good: Memoize expensive components
export const JobCard = memo(function JobCard({ job, onSelect }: JobCardProps) {
    // ...
});

// ❌ Bad: Inline object creation in JSX
<MyComponent style={{ color: 'red' }} />

// ✅ Good: Extract to constant or useMemo
const style = useMemo(() => ({ color: 'red' }), []);
<MyComponent style={style} />
```

### Styling

```typescript
// ✅ Good: Tailwind classes with clsx for conditionals
<button className={clsx(
    'px-4 py-2 rounded-md font-medium',
    'bg-blue-600 text-white hover:bg-blue-700',
    disabled && 'opacity-50 cursor-not-allowed'
)}>
    Submit
</button>

// ✅ Good: Use design tokens from CSS variables
<div className="bg-bg-primary text-text-primary">

// ❌ Bad: Inline styles
<div style={{ backgroundColor: '#1a1a2e' }}>

// ❌ Bad: Magic numbers
<div className="mt-[13px]">

// ✅ Good: Use spacing scale
<div className="mt-3">
```

### File Organization

```typescript
// ✅ Good: One component per file
// Button.tsx
export function Button() { ... }

// ✅ Good: Related items can share a file
// Badge.tsx
export function Badge() { ... }
export function StatusBadge() { ... }
export function ModelStatusBadge() { ... }

// ✅ Good: Index files for cleaner imports
// components/ui/index.ts
export { Button } from './Button';
export { Badge, StatusBadge } from './Badge';
export { Card, StatCard } from './Card';
```

---

## Testing Guidelines

### Testing Philosophy

1. **Test behavior, not implementation**
2. **Write tests that break when features break**
3. **Prefer integration over unit tests for pages**
4. **Cover edge cases and error states**

### Test Structure

```typescript
describe('ComponentName', () => {
    // Group related tests
    describe('when rendering', () => {
        it('displays the title', () => { ... });
        it('shows loading state initially', () => { ... });
    });

    describe('when user interacts', () => {
        it('calls onClick when button is clicked', () => { ... });
        it('disables button during loading', () => { ... });
    });

    describe('error handling', () => {
        it('shows error message when API fails', () => { ... });
        it('provides retry option', () => { ... });
    });
});
```

### Writing Good Tests

```typescript
// ✅ Good: Test user behavior with accessible queries
it('shows error message when form is invalid', async () => {
    render(<LoginForm />);
    
    // Find by role (accessible)
    const submitButton = screen.getByRole('button', { name: 'Submit' });
    fireEvent.click(submitButton);
    
    // Assert on visible text
    expect(screen.getByText('Email is required')).toBeInTheDocument();
});

// ❌ Bad: Testing implementation details
it('sets hasError state to true', () => {
    const { result } = renderHook(() => useForm());
    act(() => result.current.validate());
    expect(result.current.hasError).toBe(true); // Implementation detail
});

// ✅ Good: Descriptive test names
it('displays "No jobs found" when search returns empty results', () => { ... });

// ❌ Bad: Vague test names
it('works correctly', () => { ... });
it('handles edge case', () => { ... });
```

### Test Coverage Requirements

| Type | Minimum Coverage |
|------|-----------------|
| **New code** | 80% |
| **UI components** | 75% |
| **Hooks** | 85% |
| **Utilities** | 90% |
| **Critical paths** | 95% |

### Running Tests

```bash
# Run all unit tests
npm test

# Run tests in watch mode
npm run test:ui

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- Button.test.tsx

# Run E2E tests
npm run test:e2e

# Run E2E with UI
npm run test:e2e:ui
```

---

## Pull Request Process

### PR Checklist

Before submitting, ensure:

- [ ] All tests pass (`npm test`)
- [ ] No TypeScript errors (`npm run type-check`)
- [ ] No linting errors (`npm run lint`)
- [ ] Coverage is maintained or improved
- [ ] Documentation is updated (if applicable)
- [ ] Commit messages follow conventional format
- [ ] PR description is complete

### PR Template

When you open a PR, please fill out:

```markdown
## Description
Brief description of the changes and motivation.

## Type of Change
- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would break existing functionality)
- [ ] 📝 Documentation update
- [ ] 🔧 Refactoring (no functional changes)

## Testing
Describe how you tested these changes:
- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E tests
- [ ] Manual testing

## Screenshots (if UI changes)
| Before | After |
|--------|-------|
| image  | image |

## Checklist
- [ ] Tests pass locally
- [ ] No new TypeScript errors
- [ ] No new linting warnings
- [ ] Coverage maintained
- [ ] Documentation updated
```

### Review Process

1. **Automated Checks**: CI runs lint, type-check, and tests
2. **Code Review**: At least 1 approval required
3. **Discussion**: Address any feedback
4. **Merge**: Squash and merge to `develop`

### Review Guidelines for Reviewers

- Be constructive and specific
- Explain the "why" behind suggestions
- Approve promptly when ready
- Use "Request Changes" sparingly

---

## Project Structure

### Adding New Features

1. **Create feature folder**
   ```
   src/features/my-feature/
   ├── MyFeaturePage.tsx
   ├── components/
   │   └── FeatureCard.tsx
   └── index.ts
   ```

2. **Add hooks** (if needed)
   ```
   src/lib/hooks/useMyFeature.ts
   ```

3. **Add types** (if needed)
   ```
   src/lib/types/index.ts  # Add to existing file
   ```

4. **Add route**
   ```typescript
   // src/app/AppRoutes.tsx
   <Route path="/my-feature" element={<MyFeaturePage />} />
   ```

5. **Write tests**
   ```
   src/features/my-feature/MyFeaturePage.test.tsx
   ```

### Adding New Components

1. **Check existing components** in `src/components/ui/`
2. **Create component file**
   ```
   src/components/ui/NewComponent.tsx
   ```
3. **Export from index**
   ```typescript
   // src/components/ui/index.ts
   export { NewComponent } from './NewComponent';
   ```
4. **Write tests**
   ```
   src/components/ui/NewComponent.test.tsx
   ```

---

## Getting Help

### Resources

- **Documentation**: Check `docs/` folder
- **Examples**: Look at existing code
- **Issues**: Browse GitHub Issues

### Asking Questions

- **GitHub Discussions**: General questions
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat (link in README)

### Reporting Bugs

When reporting bugs, include:

1. **Description**: What happened?
2. **Expected**: What should happen?
3. **Steps to Reproduce**: Numbered steps
4. **Environment**: OS, browser, Node version
5. **Logs**: Console output, error messages
6. **Screenshots**: If visual issue

### Security Issues

For security vulnerabilities, **do not** open a public issue. Instead:

1. Email security@yourorg.com
2. Include detailed description
3. Wait for response before disclosure

---

## License

By contributing to AI Forge, you agree that your contributions will be licensed under the MIT License.

---

## Thank You!

We appreciate your contribution to AI Forge. Every bug fix, feature, and documentation improvement helps make the project better for everyone.

Happy coding! 🚀
