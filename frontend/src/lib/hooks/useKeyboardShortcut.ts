// =============================================================================
// Keyboard Shortcut Hook - Handle Cmd/Ctrl + Key combinations
// =============================================================================

import { useEffect, useCallback } from 'react';

type KeyboardShortcutHandler = () => void;

interface ShortcutOptions {
    /** Whether to use metaKey (Cmd on Mac) or ctrlKey */
    useCtrl?: boolean;
    /** Whether to prevent default browser behavior */
    preventDefault?: boolean;
    /** Whether the shortcut is currently enabled */
    enabled?: boolean;
}

/**
 * Hook for handling keyboard shortcuts with Cmd/Ctrl modifier
 * 
 * @example
 * useKeyboardShortcut('n', () => setShowDialog(true)); // Cmd+N opens dialog
 * useKeyboardShortcut('s', handleSave, { useCtrl: true }); // Ctrl+S saves
 */
export function useKeyboardShortcut(
    key: string,
    callback: KeyboardShortcutHandler,
    options: ShortcutOptions = {}
) {
    const {
        useCtrl = false,
        preventDefault = true,
        enabled = true
    } = options;

    const handleKeyDown = useCallback(
        (event: KeyboardEvent) => {
            if (!enabled) return;

            const modifierPressed = useCtrl ? event.ctrlKey : event.metaKey;

            if (
                modifierPressed &&
                event.key.toLowerCase() === key.toLowerCase() &&
                !event.shiftKey &&
                !event.altKey
            ) {
                if (preventDefault) {
                    event.preventDefault();
                }
                callback();
            }
        },
        [key, callback, useCtrl, preventDefault, enabled]
    );

    useEffect(() => {
        if (!enabled) return;

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown, enabled]);
}

/**
 * Hook for handling multiple keyboard shortcuts
 * 
 * @example
 * useKeyboardShortcuts({
 *   'n': () => setShowNewDialog(true),
 *   's': handleSave,
 *   'Escape': handleClose,
 * });
 */
export function useKeyboardShortcuts(
    shortcuts: Record<string, KeyboardShortcutHandler>,
    options: ShortcutOptions = {}
) {
    const {
        useCtrl = false,
        preventDefault = true,
        enabled = true
    } = options;

    const handleKeyDown = useCallback(
        (event: KeyboardEvent) => {
            if (!enabled) return;

            const key = event.key.toLowerCase();
            const handler = shortcuts[key] || shortcuts[event.key];

            if (!handler) return;

            // Escape doesn't need modifier
            if (key === 'escape') {
                handler();
                return;
            }

            const modifierPressed = useCtrl ? event.ctrlKey : event.metaKey;

            if (modifierPressed && !event.shiftKey && !event.altKey) {
                if (preventDefault) {
                    event.preventDefault();
                }
                handler();
            }
        },
        [shortcuts, useCtrl, preventDefault, enabled]
    );

    useEffect(() => {
        if (!enabled) return;

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown, enabled]);
}
