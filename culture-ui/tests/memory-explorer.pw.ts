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

test('memory explorer reloads summaries when agent changes', async ({ page }) => {
  await page.goto('/memory')
  const summaries = page.getByTestId('summaries')
  await expect(page.getByRole('heading', { name: 'Memory Explorer' })).toBeVisible()
  await expect(summaries).toContainText('test summary')

  await page.route('/api/agents/agent-2/semantic_summaries', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ summaries: ['new summary'] }),
    })
  })

  await page.getByLabel('agent-select').fill('agent-2')

  await expect(summaries).toContainText('new summary')
  await expect(summaries).not.toContainText('test summary')
})
