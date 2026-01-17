// =============================================================================
// E2E Test Fixtures - Shared test data and helpers
// =============================================================================

import { test as base } from '@playwright/test';

// =============================================================================
// Custom Test Fixture
// =============================================================================

interface TestFixtures {
    mockAPI: void;
}

export const test = base.extend<TestFixtures>({
    // Mock API fixture - can be used to set up mock data before tests
    mockAPI: async ({ page }, use) => {
        // Could add route interception here if needed
        await use();
    },
});

export { expect } from '@playwright/test';

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Wait for the page to be fully loaded
 */
export async function waitForPageLoad(page: import('@playwright/test').Page) {
    await page.waitForLoadState('networkidle');
}

/**
 * Navigate with loading wait
 */
export async function navigateAndWait(
    page: import('@playwright/test').Page,
    path: string
) {
    await page.goto(path);
    await waitForPageLoad(page);
}

/**
 * Select an option from a dropdown/select
 */
export async function selectOption(
    page: import('@playwright/test').Page,
    selector: string,
    value: string
) {
    await page.click(selector);
    await page.click(`text=${value}`);
}

// =============================================================================
// Test Data
// =============================================================================

export const testData = {
    jobs: {
        running: {
            id: 'job-1',
            project: 'my-project',
            status: 'running',
            progress: 78,
        },
        completed: {
            id: 'job-2',
            project: 'api-helper',
            status: 'completed',
            progress: 100,
        },
    },
    missions: {
        pending: {
            id: 'mission-1',
            title: 'Recommend retrain for myproject',
            status: 'pending_approval',
        },
    },
    models: {
        active: {
            id: 'model-1',
            name: 'myproject-v1',
            status: 'active',
        },
    },
    datasets: {
        sample: {
            id: 'ds-1',
            name: 'test-dataset',
            exampleCount: 1500,
        },
    },
};

// =============================================================================
// Page Object Helpers
// =============================================================================

export const selectors = {
    // Navigation
    sidebar: '.sidebar',
    sidebarLink: (name: string) => `.sidebar a:has-text("${name}")`,

    // Jobs page
    newFineTuneButton: 'a:has-text("New Fine-Tune")',
    jobsTable: '.table',
    jobRow: (id: string) => `tr:has-text("${id}")`,
    filterTab: (name: string) => `.filter-tab:has-text("${name}")`,

    // Missions page
    approveButton: 'button:has-text("Approve")',
    rejectButton: 'button:has-text("Reject")',
    rejectionReasonInput: 'textarea[placeholder*="reason"]',
    rejectMissionButton: 'button:has-text("Reject Mission")',

    // Common
    loadingSpinner: '.animate-spin',
    emptyState: '.empty-state',
    errorMessage: '.error',
    toast: '.toast',
};
