import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
    plugins: [react()],
    test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: ['./src/setupTests.ts'],
        include: ['src/**/*.{test,spec}.{ts,tsx}'],
        exclude: ['node_modules/', 'e2e/'],
        coverage: {
            provider: 'v8',
            reporter: ['text', 'json', 'html', 'lcov'],
            reportsDirectory: './coverage',
            exclude: [
                'node_modules/',
                'src/setupTests.ts',
                'src/mocks/**',
                '**/*.test.{ts,tsx}',
                '**/*.spec.{ts,tsx}',
                '**/*.d.ts',
                '**/index.ts',
            ],
            thresholds: {
                lines: 80,
                functions: 80,
                branches: 75,
                statements: 80,
            },
        },
        reporters: ['default', 'html'],
        outputFile: {
            html: './test-results/index.html',
        },
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
});
