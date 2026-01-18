
import { test, expect } from '@playwright/test';

test('Verify Add Git Data Source', async ({ page }) => {
    // 1. Go to Data Sources
    await page.goto('/datasets');

    // 2. Open Dialog
    await page.getByRole('button', { name: 'Add Data Source' }).click();
    await expect(page.getByRole('dialog')).toBeVisible();

    // 3. Fill Git Form (Default Tab)
    // Assuming Git tab is active or we click it
    await page.getByPlaceholder('https://github.com/user/repo').fill('https://github.com/m-ajmera/ai-forge-test-repo');
    await page.getByPlaceholder('main').fill('main');

    // 4. Submit
    await page.getByRole('button', { name: 'Clone & Parse' }).click();

    // 5. Verify Mutation/Success
    // Should close dialog and show new item
    await expect(page.getByRole('dialog')).toBeHidden();

    // 6. Check list for the new item (might take a moment to appear if not optimistic)
    // We search for "ai-forge-test-repo" or derived name
    await expect(page.getByText('ai-forge-test-repo')).toBeVisible({ timeout: 10000 });
});
