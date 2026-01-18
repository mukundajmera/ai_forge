
import { test, expect } from '@playwright/test';

test('Verify Data Sources Load', async ({ page }) => {
    // 1. Go to Data Sources page
    await page.goto('/datasets');

    // 2. Wait for header
    await expect(page.getByRole('heading', { name: 'Data Sources' })).toBeVisible();

    // 3. Verify we see at least one "Upload" item (since curl showed 3)
    // We check for the text "Upload 2026-01-18" or the class of a card
    // Assuming cards have some text content
    await expect(page.getByText('Upload 2026-01-18').first()).toBeVisible({ timeout: 5000 });
});
