// =============================================================================
// Keyboard Shortcuts Dialog - Display available shortcuts
// =============================================================================

import { Dialog } from './Dialog';

interface KeyboardShortcutsDialogProps {
    isOpen: boolean;
    onClose: () => void;
}

interface Shortcut {
    keys: string[];
    action: string;
    category: string;
}

const SHORTCUTS: Shortcut[] = [
    // Navigation
    { keys: ['⌘', 'D'], action: 'Go to Dashboard', category: 'Navigation' },
    { keys: ['⌘', 'J'], action: 'Go to Jobs', category: 'Navigation' },
    { keys: ['⌘', 'M'], action: 'Go to Models', category: 'Navigation' },

    // Actions
    { keys: ['⌘', 'N'], action: 'New Fine-Tune', category: 'Actions' },
    { keys: ['⌘', 'K'], action: 'Open Search', category: 'Actions' },

    // General
    { keys: ['ESC'], action: 'Close Dialog', category: 'General' },
    { keys: ['?'], action: 'Show Shortcuts', category: 'General' },
];

export function KeyboardShortcutsDialog({ isOpen, onClose }: KeyboardShortcutsDialogProps) {
    // Group shortcuts by category
    const groupedShortcuts = SHORTCUTS.reduce((acc, shortcut) => {
        if (!acc[shortcut.category]) {
            acc[shortcut.category] = [];
        }
        acc[shortcut.category].push(shortcut);
        return acc;
    }, {} as Record<string, Shortcut[]>);

    return (
        <Dialog isOpen={isOpen} onClose={onClose} title="Keyboard Shortcuts">
            <div className="shortcuts-content">
                {Object.entries(groupedShortcuts).map(([category, shortcuts]) => (
                    <div key={category} className="shortcuts-category">
                        <h4 className="category-title">{category}</h4>
                        <div className="shortcuts-list">
                            {shortcuts.map((shortcut) => (
                                <div key={shortcut.action} className="shortcut-row">
                                    <span className="shortcut-action">{shortcut.action}</span>
                                    <div className="shortcut-keys">
                                        {shortcut.keys.map((key, idx) => (
                                            <span key={idx}>
                                                <kbd className="shortcut-key">{key}</kbd>
                                                {idx < shortcut.keys.length - 1 && (
                                                    <span className="key-separator">+</span>
                                                )}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}

                <p className="shortcuts-hint">
                    Press <kbd className="shortcut-key">?</kbd> anywhere to show this dialog
                </p>
            </div>

            <style>{`
                .shortcuts-content {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-6);
                }

                .shortcuts-category {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .category-title {
                    font-size: var(--text-sm);
                    font-weight: var(--font-semibold);
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    margin: 0;
                }

                .shortcuts-list {
                    display: flex;
                    flex-direction: column;
                    gap: var(--space-2);
                }

                .shortcut-row {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: var(--space-2) 0;
                }

                .shortcut-action {
                    font-size: var(--text-sm);
                    color: var(--text-primary);
                }

                .shortcut-keys {
                    display: flex;
                    align-items: center;
                    gap: var(--space-1);
                }

                .shortcut-key {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 24px;
                    height: 24px;
                    padding: 0 var(--space-2);
                    background: var(--bg-elevated);
                    border: 1px solid var(--border-default);
                    border-radius: var(--radius-sm);
                    font-family: var(--font-mono);
                    font-size: var(--text-xs);
                    color: var(--text-primary);
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                }

                .key-separator {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                }

                .shortcuts-hint {
                    font-size: var(--text-xs);
                    color: var(--text-tertiary);
                    text-align: center;
                    margin: 0;
                    padding-top: var(--space-4);
                    border-top: 1px solid var(--border-subtle);
                }
            `}</style>
        </Dialog>
    );
}
