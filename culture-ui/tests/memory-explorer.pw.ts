import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

test.beforeEach(async ({ page }) => {
  await page.route('/api/agents/agent-1/semantic_summaries', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ summaries: ['test summary'] }),
    })
  })
})

test('memory explorer loads', async ({ page }) => {
  await page.goto('/memory')
  await expect(page.getByRole('heading', { name: 'Memory Explorer' })).toBeVisible()
  await expect(page.getByText('test summary')).toBeVisible()
})
