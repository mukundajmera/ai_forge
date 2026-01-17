import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
    plugins: [react()],
    test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: ['./src/setupTests.ts'],
        include: ['src/**/*.integration.test.{ts,tsx}'],
        exclude: ['node_modules/', 'e2e/'],
        testTimeout: 15000,
        hookTimeout: 15000,
        reporters: ['default'],
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
});
