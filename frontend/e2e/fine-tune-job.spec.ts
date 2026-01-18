
import { test, expect } from '@playwright/test';

test('Verify Fine-Tune Job Creation', async ({ page }) => {
    // 1. Go to New Job
    await page.goto('/jobs/new');

    // 2. Step 1: Dataset
    // Assuming "audit-dataset-01" exists from previous step, OR we pick any available
    // We'll pick the first card
    await expect(page.getByText('Select Training Data')).toBeVisible();
    await page.locator('.radio-card').first().click(); // Select first dataset
    await page.getByRole('button', { name: 'Next' }).click();

    // 3. Step 2: Config
    await expect(page.getByText('Configure Training')).toBeVisible();
    // Default model (Llama 3B) and Preset (Balanced) should be selected
    await page.getByRole('button', { name: 'Next' }).click();

    // 4. Step 3: Review
    await expect(page.getByText('Review & Start')).toBeVisible();
    await page.getByRole('button', { name: 'Start Training' }).click();

    // 5. Verify Redirect
    // Should go to /jobs/:id
    await expect(page).toHaveURL(/\/jobs\/.+/);
    await expect(page.getByText('Job Status')).toBeVisible({ timeout: 10000 });
});
