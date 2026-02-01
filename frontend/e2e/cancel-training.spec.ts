import { test, expect } from './fixtures';

test.describe('Training Cancellation', () => {
    // Mock data for a running job
    const runningJob = {
        job_id: 'job-running-123',
        status: 'training',
        progress: 45.0,
        config: {
            base_model: 'llama3:8b',
            epochs: 3
        },
        data_path: 'dataset.json',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        loss: 1.2
    };

    const cancelledJob = {
        ...runningJob,
        status: 'cancelled',
        progress: 45.0,
        error: 'Job cancelled'
    };

    test('can cancel a running training job', async ({ page }) => {
        // Mock the GET request for specific job
        await page.route('**/v1/fine-tune/job-running-123', async route => {
            const method = route.request().method();
            if (method === 'DELETE') {
                await route.fulfill({
                    status: 200,
                    contentType: 'application/json',
                    body: JSON.stringify({ message: "Job job-running-123 cancelled" })
                });
                return;
            }

            // If we have already clicked cancel, return cancelled state on next poll
            // Note: In real Playwright tests, controlling polling is tricky.
            // We'll rely on the DELETE interception to verify the action.
            // And we can update the mock for subsequent GETs if we could store state.
            // Simple approach: Initially return running.

            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify(runningJob)
            });
        });

        // Navigate to the job detail page
        await page.goto('/jobs/job-running-123');

        // Verify job is running
        await expect(page.getByText('training', { exact: false })).toBeVisible();

        // Locate the Stop/Cancel button
        // Assuming the button has text "Stop" or "Cancel"
        const stopButton = page.getByRole('button', { name: /Stop|Cancel/i });
        await expect(stopButton).toBeVisible();

        // Handle confirmation dialog
        page.on('dialog', dialog => dialog.accept());

        // Setup promise to verify DELETE request
        const deleteRequestPromise = page.waitForRequest(request =>
            request.url().includes('/v1/fine-tune/job-running-123') &&
            request.method() === 'DELETE'
        );

        // Click Stop
        await stopButton.click();

        // Wait for the DELETE request to be made
        const deleteRequest = await deleteRequestPromise;
        expect(deleteRequest).toBeTruthy();

        // Mock the update to cancelled state
        await page.unroute('**/v1/fine-tune/job-running-123');
        await page.route('**/v1/fine-tune/job-running-123', async route => {
            if (route.request().method() === 'DELETE') {
                await route.fulfill({ status: 200, body: JSON.stringify({ message: "Cancelled" }) });
                return;
            }
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify(cancelledJob)
            });
        });

        // Wait for UI to update (polling)
        // This might take a second depending on poll interval
        // Verify redirection to jobs list
        await expect(page).toHaveURL(/.*\/jobs$/);

        // Optional: We could verify a toast or the list update if we mocked the list endpoint too.
        // For now, the successful DELETE call and redirect is sufficient proof of UI wiring.
    });
});
