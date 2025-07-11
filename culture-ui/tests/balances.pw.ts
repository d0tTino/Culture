import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

test.beforeEach(async ({ page }) => {
  await page.route('/api/token_balances', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        agents: { A: { ip: 1, du: 2, tokens: { TOK: 3 } } },
      }),
    })
  })
})

test('balances page loads', async ({ page }) => {
  await page.goto('/balances')
  await expect(page.getByRole('heading', { name: 'Token Balances' })).toBeVisible()
  await expect(page.getByTestId('balance-table')).toContainText('A')
  await expect(page.getByTestId('balance-table')).toContainText('TOK:3')
})
