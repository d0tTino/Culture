import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

test('renders map and fetches memory data', async ({ page }) => {
  await page.route('/api/map', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ agents: { 'agent-1': {} } }),
    })
  })

  await page.route('/api/memory/agent-1', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        semantic: ['semantic mem'],
        episodic: ['episodic mem'],
      }),
    })
  })

  await page.goto('/memory')

  await expect(page.getByRole('heading', { name: 'Memory Explorer' })).toBeVisible()
  await expect(page.getByLabel('agent-select')).toHaveValue('agent-1')
  await expect(page.getByTestId('summaries')).toContainText('semantic mem')
  await expect(page.getByTestId('memories')).toContainText('episodic mem')
})

