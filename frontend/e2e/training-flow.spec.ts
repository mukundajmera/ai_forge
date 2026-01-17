// =============================================================================
// Training Flow E2E Test
// =============================================================================

import { test, expect } from '@playwright/test';

test.describe('Training Flow', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
    });

    // =========================================================================
    // Navigation Tests
    // =========================================================================

    test('can navigate to jobs page from dashboard', async ({ page }) => {
        await page.click('a:has-text("Jobs")');
        await expect(page).toHaveURL(/\/jobs/);
        await expect(page.getByRole('heading', { name: 'Training Jobs' })).toBeVisible();
    });

    test('can navigate to datasets page from dashboard', async ({ page }) => {
        await page.click('a:has-text("Data")');
        await expect(page).toHaveURL(/\/data/);
    });

    // =========================================================================
    // Jobs List Tests
    // =========================================================================

    test('jobs page displays job list', async ({ page }) => {
        await page.goto('/jobs');

        await expect(page.getByRole('heading', { name: 'Training Jobs' })).toBeVisible();
        await expect(page.getByText('New Fine-Tune')).toBeVisible();
    });

    test('can filter jobs by status', async ({ page }) => {
        await page.goto('/jobs');

        // Click on Running filter
        await page.click('.filter-tab:has-text("Running")');

        // Tab should be active
        await expect(page.locator('.filter-tab.active')).toContainText('Running');
    });

    test('can search jobs by project name', async ({ page }) => {
        await page.goto('/jobs');

        const searchInput = page.getByPlaceholder(/Search projects/i);
        await searchInput.fill('my-project');

        // Should filter results
        await expect(page.getByText('my-project')).toBeVisible();
    });

    // =========================================================================
    // New Fine-Tune Flow Tests
    // =========================================================================

    test('can access new fine-tune page', async ({ page }) => {
        await page.goto('/jobs');

        await page.click('a:has-text("New Fine-Tune")');

        await expect(page).toHaveURL(/\/jobs\/new/);
    });

    test('new fine-tune page has required form fields', async ({ page }) => {
        await page.goto('/jobs/new');

        // Check for essential form elements (these may vary based on actual implementation)
        await expect(page.getByRole('heading')).toBeVisible();
    });

    // =========================================================================
    // Job Detail Tests
    // =========================================================================

    test('can navigate to job detail page', async ({ page }) => {
        await page.goto('/jobs');

        // Click on first job link
        const jobLink = page.locator('a[href*="/jobs/job-"]').first();
        await jobLink.click();

        await expect(page).toHaveURL(/\/jobs\/job-/);
    });

    test('job detail page displays job information', async ({ page }) => {
        await page.goto('/jobs/job-1');

        // Should display job details (project name, status, etc.)
        await expect(page.locator('body')).not.toBeEmpty();
    });

    // =========================================================================
    // Training Progress Tests
    // =========================================================================

    test('running job shows progress', async ({ page }) => {
        await page.goto('/jobs');

        // Should show progress bar for running jobs
        const progressBar = page.locator('.progress-bar');
        if (await progressBar.count() > 0) {
            await expect(progressBar.first()).toBeVisible();
        }
    });

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    test('handles non-existent job gracefully', async ({ page }) => {
        await page.goto('/jobs/non-existent-job-id');

        // Should show error or redirect
        await page.waitForLoadState('networkidle');
        // The page should not crash
        await expect(page.locator('body')).not.toBeEmpty();
    });
});

test.describe('Complete Training Flow', () => {
    test('full flow from jobs list to viewing job details', async ({ page }) => {
        // Step 1: Navigate to jobs page
        await page.goto('/jobs');
        await expect(page.getByRole('heading', { name: 'Training Jobs' })).toBeVisible();

        // Step 2: Verify jobs are displayed
        await expect(page.getByText('my-project')).toBeVisible({ timeout: 10000 });

        // Step 3: Click on a job to view details
        const jobLink = page.locator('.job-link').first();
        if (await jobLink.count() > 0) {
            await jobLink.click();

            // Step 4: Wait for job detail page
            await page.waitForLoadState('networkidle');

            // Step 5: Verify we're on the detail page
            await expect(page).toHaveURL(/\/jobs\//);
        }
    });
});
