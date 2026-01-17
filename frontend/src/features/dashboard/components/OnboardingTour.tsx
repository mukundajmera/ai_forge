// =============================================================================
// Onboarding Tour - First-time user experience guide
// =============================================================================

import { useState } from 'react';
import { Dialog } from '@/components/ui/Dialog';
import { Button } from '@/components/ui/Button';
import {
    ChevronRight,
    ChevronLeft,
    X,
    Upload,
    Zap,
    TrendingUp,
    Rocket,
} from 'lucide-react';

const STORAGE_KEY = 'ai_forge_onboarding_completed';

interface TourStep {
    title: string;
    description: string;
    icon: React.ReactNode;
    highlight?: string;
}

const TOUR_STEPS: TourStep[] = [
    {
        title: 'Welcome to AI Forge',
        description:
            'Your local LLM fine-tuning control plane. Fine-tune models on your own hardware with complete privacy.',
        icon: <Zap className="tour-icon" size={48} />,
    },
    {
        title: 'Upload Your Data',
        description:
            'Start by uploading code, documentation, or PDFs. AI Forge will parse and prepare them for training using RAFT synthesis.',
        icon: <Upload className="tour-icon" size={48} />,
        highlight: '/datasets',
    },
    {
        title: 'Start Training',
        description:
            'Choose your base model, configure parameters, and let PiSSA+QLoRA work its magic. Monitor training in real-time.',
        icon: <TrendingUp className="tour-icon" size={48} />,
        highlight: '/jobs',
    },
    {
        title: 'Deploy to Ollama',
        description:
            'Once training completes, export models directly to Ollama. Query your custom model via CLI, API, or web UI.',
        icon: <Rocket className="tour-icon" size={48} />,
        highlight: '/models',
    },
];

export function OnboardingTour() {
    const [isOpen, setIsOpen] = useState(() => {
        try {
            return !localStorage.getItem(STORAGE_KEY);
        } catch {
            return true;
        }
    });
    const [step, setStep] = useState(0);

    const handleComplete = () => {
        try {
            localStorage.setItem(STORAGE_KEY, 'true');
        } catch {
            // Ignore localStorage errors
        }
        setIsOpen(false);
    };

    const nextStep = () => {
        if (step < TOUR_STEPS.length - 1) {
            setStep(step + 1);
        } else {
            handleComplete();
        }
    };

    const prevStep = () => {
        if (step > 0) {
            setStep(step - 1);
        }
    };

    const currentStep = TOUR_STEPS[step];
    const isFirstStep = step === 0;
    const isLastStep = step === TOUR_STEPS.length - 1;

    if (!isOpen) return null;

    return (
        <Dialog isOpen={isOpen} onClose={handleComplete} title="Welcome" size="lg">
            <div className="tour-container">
                {/* Skip button */}
                <button className="tour-skip" onClick={handleComplete}>
                    <X size={20} />
                </button>

                {/* Content */}
                <div className="tour-content">
                    <div className="tour-icon-wrapper">{currentStep.icon}</div>

                    <h2 className="tour-title">{currentStep.title}</h2>
                    <p className="tour-description">{currentStep.description}</p>
                </div>

                {/* Progress dots */}
                <div className="tour-dots">
                    {TOUR_STEPS.map((_, idx) => (
                        <button
                            key={idx}
                            className={`tour-dot ${idx === step ? 'active' : ''}`}
                            onClick={() => setStep(idx)}
                            aria-label={`Go to step ${idx + 1}`}
                        />
                    ))}
                </div>

                {/* Navigation */}
                <div className="tour-navigation">
                    {!isFirstStep ? (
                        <Button
                            intent="ghost"
                            icon={<ChevronLeft size={16} />}
                            onClick={prevStep}
                        >
                            Back
                        </Button>
                    ) : (
                        <Button intent="ghost" onClick={handleComplete}>
                            Skip Tour
                        </Button>
                    )}

                    <Button
                        intent="primary"
                        onClick={nextStep}
                    >
                        {isLastStep ? "Let's Go!" : 'Next'}
                        {!isLastStep && <ChevronRight size={16} />}
                    </Button>
                </div>
            </div>

            <style>{`
                .tour-container {
                    position: relative;
                    padding: var(--space-2);
                }

                .tour-skip {
                    position: absolute;
                    top: 0;
                    right: 0;
                    padding: var(--space-2);
                    background: transparent;
                    border: none;
                    color: var(--text-tertiary);
                    cursor: pointer;
                    transition: color 0.2s;
                }

                .tour-skip:hover {
                    color: var(--text-primary);
                }

                .tour-content {
                    text-align: center;
                    padding: var(--space-8) var(--space-4);
                }

                .tour-icon-wrapper {
                    display: flex;
                    justify-content: center;
                    margin-bottom: var(--space-6);
                }

                .tour-icon {
                    color: var(--accent-primary);
                }

                .tour-title {
                    font-size: var(--text-2xl);
                    font-weight: var(--font-bold);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-3) 0;
                }

                .tour-description {
                    font-size: var(--text-base);
                    color: var(--text-secondary);
                    line-height: 1.6;
                    max-width: 400px;
                    margin: 0 auto;
                }

                .tour-dots {
                    display: flex;
                    justify-content: center;
                    gap: var(--space-2);
                    margin-bottom: var(--space-6);
                }

                .tour-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: var(--bg-elevated);
                    border: 1px solid var(--border-subtle);
                    cursor: pointer;
                    transition: all 0.2s;
                    padding: 0;
                }

                .tour-dot:hover {
                    background: var(--border-default);
                }

                .tour-dot.active {
                    background: var(--accent-primary);
                    border-color: var(--accent-primary);
                    transform: scale(1.25);
                }

                .tour-navigation {
                    display: flex;
                    justify-content: space-between;
                    gap: var(--space-3);
                }

                .tour-navigation button {
                    min-width: 100px;
                }
            `}</style>
        </Dialog>
    );
}

/**
 * Hook to reset onboarding tour (for testing or settings)
 */
export function useResetOnboarding() {
    return () => {
        try {
            localStorage.removeItem(STORAGE_KEY);
            window.location.reload();
        } catch {
            // Ignore errors
        }
    };
}
