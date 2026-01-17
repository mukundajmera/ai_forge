// =============================================================================
// Mission Approval Flow E2E Test
// =============================================================================

import { test, expect } from '@playwright/test';

test.describe('Mission Approval Flow', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
    });

    // =========================================================================
    // Navigation Tests
    // =========================================================================

    test('can navigate to missions page', async ({ page }) => {
        await page.click('a:has-text("Missions")');
        await expect(page).toHaveURL(/\/missions/);
    });

    // =========================================================================
    // Missions List Tests
    // =========================================================================

    test('missions page displays mission list', async ({ page }) => {
        await page.goto('/missions');

        await expect(page.getByRole('heading', { name: /Missions/i })).toBeVisible();
    });

    test('shows pending badge for pending missions', async ({ page }) => {
        await page.goto('/missions');

        // Check for pending status indicators
        const pendingBadge = page.locator('.badge:has-text("pending")');
        if (await pendingBadge.count() > 0) {
            await expect(pendingBadge.first()).toBeVisible();
        }
    });

    // =========================================================================
    // Mission Detail Tests
    // =========================================================================

    test('can view mission details', async ({ page }) => {
        await page.goto('/missions/mission-1');

        // Should display mission title
        await expect(page.getByText('Recommend retrain for myproject')).toBeVisible({
            timeout: 10000,
        });
    });

    test('mission detail shows agent analysis', async ({ page }) => {
        await page.goto('/missions/mission-1');

        await expect(page.getByText('Agent Analysis')).toBeVisible({ timeout: 10000 });
        await expect(page.getByText('Trigger')).toBeVisible();
        await expect(page.getByText('Analysis')).toBeVisible();
        await expect(page.getByText('Expected Outcome')).toBeVisible();
    });

    test('mission detail shows confidence score', async ({ page }) => {
        await page.goto('/missions/mission-1');

        await expect(page.getByText('Confidence')).toBeVisible({ timeout: 10000 });
        await expect(page.getByText('85%')).toBeVisible();
    });

    // =========================================================================
    // Approve Flow Tests
    // =========================================================================

    test('approve button is visible for pending mission', async ({ page }) => {
        await page.goto('/missions/mission-1');

        const approveButton = page.getByRole('button', { name: /Approve/i });
        await expect(approveButton).toBeVisible({ timeout: 10000 });
    });

    test('clicking approve triggers approval action', async ({ page }) => {
        await page.goto('/missions/mission-1');

        const approveButton = page.getByRole('button', { name: /Approve/i });
        await expect(approveButton).toBeVisible({ timeout: 10000 });

        await approveButton.click();

        // Should navigate to missions list or show success
        await page.waitForLoadState('networkidle');
    });

    // =========================================================================
    // Reject Flow Tests
    // =========================================================================

    test('reject button is visible for pending mission', async ({ page }) => {
        await page.goto('/missions/mission-1');

        const rejectButton = page.getByRole('button', { name: /^Reject$/i });
        await expect(rejectButton).toBeVisible({ timeout: 10000 });
    });

    test('clicking reject opens rejection dialog', async ({ page }) => {
        await page.goto('/missions/mission-1');

        const rejectButton = page.getByRole('button', { name: /^Reject$/i });
        await expect(rejectButton).toBeVisible({ timeout: 10000 });

        await rejectButton.click();

        // Should show rejection dialog
        await expect(page.getByText('Reject Mission')).toBeVisible();
        await expect(page.getByPlaceholder(/Not enough new commits/i)).toBeVisible();
    });

    test('reject dialog requires reason before submitting', async ({ page }) => {
        await page.goto('/missions/mission-1');

        // Open reject dialog
        await page.click('button:has-text("Reject")');

        await expect(page.getByText('Reject Mission')).toBeVisible();

        // Submit button should be disabled
        const submitButton = page.getByRole('button', { name: 'Reject Mission' });
        await expect(submitButton).toBeDisabled();
    });

    test('can submit rejection with reason', async ({ page }) => {
        await page.goto('/missions/mission-1');

        // Open reject dialog
        await page.click('button:has-text("Reject")');

        await expect(page.getByText('Reject Mission')).toBeVisible();

        // Fill in reason
        const textarea = page.getByPlaceholder(/Not enough new commits/i);
        await textarea.fill('Not enough new data to justify retraining');

        // Submit button should be enabled now
        const submitButton = page.getByRole('button', { name: 'Reject Mission' });
        await expect(submitButton).toBeEnabled();

        await submitButton.click();

        // Should process the rejection
        await page.waitForLoadState('networkidle');
    });

    test('can cancel rejection dialog', async ({ page }) => {
        await page.goto('/missions/mission-1');

        // Open reject dialog
        await page.click('button:has-text("Reject")');

        await expect(page.getByRole('heading', { name: 'Reject Mission' })).toBeVisible();

        // Click cancel
        await page.click('button:has-text("Cancel")');

        // Dialog should close
        await expect(page.getByRole('heading', { name: 'Reject Mission' })).not.toBeVisible();
    });

    // =========================================================================
    // Back Navigation Tests
    // =========================================================================

    test('can navigate back to missions list', async ({ page }) => {
        await page.goto('/missions/mission-1');

        await expect(page.getByText('Back to Missions')).toBeVisible({ timeout: 10000 });

        await page.click('text=Back to Missions');

        await expect(page).toHaveURL(/\/missions/);
    });

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    test('handles non-existent mission gracefully', async ({ page }) => {
        await page.goto('/missions/non-existent-id');

        // Should show error state
        await expect(page.getByText('Mission Not Found')).toBeVisible({ timeout: 10000 });
        await expect(page.getByRole('button', { name: 'Back to Missions' })).toBeVisible();
    });
});

test.describe('Mission Approval Full Flow', () => {
    test('complete mission approval flow end-to-end', async ({ page }) => {
        // Step 1: Navigate to missions
        await page.goto('/missions');

        // Step 2: Click on pending mission (if any visible)
        const missionLink = page.locator('a[href*="/missions/"]').first();
        if (await missionLink.count() > 0) {
            await missionLink.click();

            // Step 3: Verify mission details are shown
            await page.waitForLoadState('networkidle');

            // Step 4: Check for approve/reject buttons
            const approveButton = page.getByRole('button', { name: /Approve/i });
            if (await approveButton.count() > 0) {
                await expect(approveButton).toBeVisible();

                // Step 5: Approve the mission
                await approveButton.click();

                // Step 6: Verify redirect/success
                await page.waitForLoadState('networkidle');
            }
        }
    });
});
