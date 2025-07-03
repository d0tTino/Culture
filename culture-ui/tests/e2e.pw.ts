import { test, expect } from '@playwright/test';

test.skip(true, 'e2e tests are skipped in this environment');

const tabSelector = '.flexlayout__tab';

// Ensure sidebar links are visible and DockManager layout persists after reload

test('sidebar and dock layout persist', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');

  await expect(page.getByRole('link', { name: 'Home' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Mission Overview' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Agent Data' })).toBeVisible();

  const tabs = page.locator(tabSelector);
  const initialCount = await tabs.count();
  await expect(tabs).toHaveCount(initialCount);

  await page.reload();
  await page.waitForLoadState('networkidle');

  await expect(page.locator(tabSelector)).toHaveCount(initialCount);
});
