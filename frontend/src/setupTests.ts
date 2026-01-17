// =============================================================================
// Test Setup - Global test configuration and mocks
// =============================================================================

import '@testing-library/jest-dom';
import { afterAll, afterEach, beforeAll } from 'vitest';
import { cleanup } from '@testing-library/react';
import { server } from './mocks/server';

// Establish API mocking before all tests
beforeAll(() => {
    server.listen({ onUnhandledRequest: 'warn' });
});

// Reset handlers after each test
afterEach(() => {
    cleanup();
    server.resetHandlers();
});

// Clean up after all tests
afterAll(() => {
    server.close();
});

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => { },
        removeListener: () => { },
        addEventListener: () => { },
        removeEventListener: () => { },
        dispatchEvent: () => false,
    }),
});

// Mock IntersectionObserver
class MockIntersectionObserver {
    observe = () => null;
    disconnect = () => null;
    unobserve = () => null;
}
Object.defineProperty(window, 'IntersectionObserver', {
    writable: true,
    configurable: true,
    value: MockIntersectionObserver,
});

// Mock ResizeObserver
class MockResizeObserver {
    observe = () => null;
    disconnect = () => null;
    unobserve = () => null;
}
Object.defineProperty(window, 'ResizeObserver', {
    writable: true,
    configurable: true,
    value: MockResizeObserver,
});

// Suppress console errors during tests (optional - comment out for debugging)
// const originalError = console.error;
// console.error = (...args) => {
//     if (args[0]?.includes?.('Warning:')) return;
//     originalError.call(console, ...args);
// };
