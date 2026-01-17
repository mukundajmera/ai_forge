// =============================================================================
// MSW Browser - Browser environment for development mode
// =============================================================================

import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';

// Create and export the MSW browser worker with all handlers
export const worker = setupWorker(...handlers);
