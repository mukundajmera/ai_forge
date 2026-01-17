// =============================================================================
// MSW Server - Node.js environment for unit/integration tests
// =============================================================================

import { setupServer } from 'msw/node';
import { handlers } from './handlers';

// Create and export the MSW server with all handlers
export const server = setupServer(...handlers);
