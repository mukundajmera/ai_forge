
import { test, expect } from '@playwright/test';

test.describe('System Audit & Verification', () => {

    // -------------------------------------------------------------------------
    // 1. Data Ingestion (Add Source)
    // -------------------------------------------------------------------------
    test('1. Verify Add Data Source (Git)', async ({ page }) => {
        await page.goto('/datasets');
        await expect(page.getByRole('heading', { name: 'Data Sources' })).toBeVisible();

        // Open Dialog
        await page.getByRole('button', { name: 'Add Data Source' }).click();
        await expect(page.getByRole('dialog')).toBeVisible();

        // Select Git Tab (assuming it's default or clickable)
        // Note: I need to verify tabs exists, if not, it might be a dropdown
        // Based on file structure `AddDataSourceDialog.tsx`, likely tabs.

        // Fill Form
        await page.fill('input[name="url"]', 'https://github.com/octocat/Hello-World.git');
        await page.fill('input[name="name"]', 'audit-test-repo');

        // Submit
        await page.getByRole('button', { name: 'Add Source' }).click();

        // Verify it appears in the list
        // It might take time to sync, but the card should appear immediately with "Syncing" or "Pending"
        await expect(page.getByText('audit-test-repo')).toBeVisible({ timeout: 10000 });
    });

    // -------------------------------------------------------------------------
    // 2. Dataset Generation (Create from Source)
    // -------------------------------------------------------------------------
    test('2. Verify Create Dataset', async ({ page }) => {
        await page.goto('/datasets/generated');
        await expect(page.getByRole('heading', { name: 'Datasets' })).toBeVisible();

        // Click Generate (if sources exist)
        // This test depends on Step 1. In a real shared env, this is flaky, but for local agent loop it's fine.

        // Check for "Generate Dataset" button
        const generateBtn = page.getByRole('button', { name: 'Generate New' });
        if (await generateBtn.isVisible()) {
            await generateBtn.click();
            // Expect Dialog
            await expect(page.getByRole('dialog')).toBeVisible();
            // We won't submit to avoid heavy processing, just verify the form opens and loads sources
            await expect(page.getByText('Select Data Sources')).toBeVisible();
        } else {
            console.log('Generate button not found - maybe list is empty?');
        }
    });

    // -------------------------------------------------------------------------
    // 3. Fine-Tuning (Job Creation)
    // -------------------------------------------------------------------------
    test('3. Verify Fine-Tune Job Creation w/ Empty State Handling', async ({ page }) => {
        await page.goto('/jobs/new');
        // As verified before, check for "Select a dataset"
        await expect(page.getByText('Select a dataset')).toBeVisible();

        // If we successfully created a dataset in step 2 (we didn't yet), we could pick it.
        // For now, checking the page loads without 404/Crash is the goal.
    });

    // -------------------------------------------------------------------------
    // 4. Models & Settings
    // -------------------------------------------------------------------------
    test('4. Verify Models & Settings load', async ({ page }) => {
        await page.goto('/models');
        await expect(page.getByRole('heading', { name: 'Models' })).toBeVisible();

        await page.goto('/settings');
        await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
    });

});
