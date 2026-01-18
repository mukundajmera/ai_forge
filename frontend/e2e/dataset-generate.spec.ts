
import { test, expect } from '@playwright/test';

test('Verify Dataset Generation', async ({ page }) => {
    // 1. Go to Generated Datasets
    await page.goto('/datasets/generated');

    // 2. Open Dialog
    await page.getByRole('button', { name: 'Generate Dataset' }).click();
    await expect(page.getByRole('dialog')).toBeVisible();

    // 3. Select Source
    // Assuming steps: Step 1 Select Source
    await page.getByText('Upload 2026-01-18').first().click();

    // 4. Next Step (Config)
    await page.getByRole('button', { name: 'Next' }).click();

    // 5. Fill Name
    await page.getByPlaceholder('my-python-dataset').fill('audit-dataset-01');

    // 6. Generate
    await page.getByRole('button', { name: 'Generate Dataset' }).click();

    // 7. Verify Success
    await expect(page.getByRole('dialog')).toBeHidden();
    await expect(page.getByText('audit-dataset-01')).toBeVisible({ timeout: 10000 });
});
