# 🔥 AI Forge UI

> A world-class web interface for local LLM fine-tuning with PiSSA, QLoRA, and Ollama deployment.

[![CI](https://github.com/yourorg/ai-forge-ui/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/yourorg/ai-forge-ui/actions/workflows/frontend-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)

## ✨ Features

- **📊 Real-Time Training Monitoring** - Live loss curves, metrics, and log streaming
- **🤖 AI Agent Integration** - Repo Guardian suggests optimal retraining times
- **📁 Smart Data Ingestion** - Upload code/docs, auto-parse with AST, generate training data
- **⚡ PiSSA + QLoRA** - 3-5x faster convergence than standard LoRA
- **🚀 One-Click Deployment** - Export to GGUF, deploy to Ollama instantly
- **🎯 Confidence-Driven UX** - Deploy decisions backed by CodeBLEU, HumanEval metrics

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm or pnpm
- AI Forge Backend running (see [backend setup](docs/DEPLOYMENT.md))

### Installation

```bash
# Clone repo
git clone https://github.com/yourorg/ai-forge-ui.git
cd ai-forge-ui/frontend

# Install dependencies
npm install

# Set environment variables
cp .env.example .env
# Edit .env with your API base URL

# Start dev server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [User Guide](frontend/docs/USER_GUIDE.md) | How to use AI Forge |
| [Architecture](frontend/docs/ARCHITECTURE.md) | System design and data flow |
| [API Reference](frontend/docs/API.md) | Backend API documentation |
| [Contributing](CONTRIBUTING.md) | Development setup and guidelines |
| [Deployment](docs/DEPLOYMENT.md) | Production deployment guide |

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   React UI      │─────▶│  FastAPI Backend │─────▶│  MLX / Unsloth  │
│  (This Repo)    │      │                  │      │   (Training)    │
└─────────────────┘      └──────────────────┘      └─────────────────┘
        │                         │                          │
        │                         ▼                          ▼
        │                  ┌─────────────┐          ┌──────────────┐
        │                  │  Antigravity│          │    Ollama    │
        └─────────────────▶│   Agent     │          │  (Serving)   │
                           └─────────────┘          └──────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, TypeScript, Vite |
| Styling | Tailwind CSS, CSS Variables |
| State | React Query (TanStack Query) |
| Charts | Recharts |
| Testing | Vitest, Playwright, MSW |
| Backend | FastAPI, Python 3.11 |
| Training | MLX, Unsloth, PiSSA, QLoRA |
| Serving | Ollama |

## 🧪 Testing

```bash
# Unit tests
npm test

# Unit tests with coverage
npm run test:coverage

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e

# E2E with UI
npm run test:e2e:ui
```

### Test Coverage Goals

- **Unit Tests**: 80%+ coverage
- **Integration Tests**: All page components
- **E2E Tests**: Critical user flows

## 📦 Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Type check
npm run type-check

# Lint
npm run lint
```

## 🎨 Design System

AI Forge uses a custom design system with:

- **Dark-first palette** with semantic color tokens
- **Accessible components** (WCAG 2.1 AA compliant)
- **Consistent spacing** using an 8px grid
- **Typography scale** using Inter font family

See [Design System](frontend/docs/DESIGN_SYSTEM.md) for full documentation.

## 📁 Project Structure

```
ai_forge/
├── frontend/                 # React UI (this README)
│   ├── src/
│   │   ├── app/             # App shell, routing
│   │   ├── components/      # Shared UI components
│   │   ├── features/        # Feature modules
│   │   ├── lib/             # Utilities, hooks, API
│   │   └── mocks/           # MSW handlers
│   ├── e2e/                 # Playwright tests
│   └── docs/                # Documentation
├── ai_forge/                # Python backend
├── training/                # Training engine
├── judge/                   # Model evaluation
├── conductor/               # Orchestration
└── antigravity_agent/       # Repo Guardian agent
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_ENABLE_MOCKS` | Enable MSW mocking | `false` |
| `VITE_LOG_LEVEL` | Logging level | `info` |

### Feature Flags

Feature flags can be configured in `.env`:

```bash
VITE_FEATURE_MISSIONS=true
VITE_FEATURE_MULTIMODEL=false
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup
- Code standards
- Pull request guidelines
- Testing requirements

### Quick Contribution Steps

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'feat: add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

MIT © [Your Organization]

---

<p align="center">
  Made with ❤️ for the AI/ML community
</p>
