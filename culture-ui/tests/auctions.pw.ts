import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

test.beforeEach(async ({ page }) => {
  await page.route('/api/auctions', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        auctions: [
          {
            id: 1,
            item: 'sword',
            status: 'open',
            winner_id: null,
            bids: [
              { agent_id: 'A', amount: 3 },
              { agent_id: 'B', amount: 2 },
            ],
          },
        ],
      }),
    })
  })
})

test('auctions page loads', async ({ page }) => {
  await page.goto('/auctions')
  await expect(page.getByRole('heading', { name: 'Auctions' })).toBeVisible()
  await expect(page.getByTestId('auction-table')).toContainText('sword')
  await expect(page.getByTestId('auction-table')).toContainText('A:3')
})
