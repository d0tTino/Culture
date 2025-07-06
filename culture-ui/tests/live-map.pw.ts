import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

test.beforeEach(async ({ page }) => {
  await page.route('/api/agents/agent-1/semantic_summaries', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ summaries: ['p1'] }),
    })
  })
})

test('live map widget loads and updates', async ({ page }) => {
  await page.addInitScript(() => {
    class MockEventSource extends EventTarget {
      static instance: MockEventSource
      url: string
      constructor(url: string) {
        super()
        this.url = url
        MockEventSource.instance = this
      }
      close() {}
    }
    // @ts-expect-error override
    window.EventSource = MockEventSource as unknown as typeof EventSource
  })

  await page.goto('/live-map')
  await expect(page.getByRole('heading', { name: 'Live Map' })).toBeVisible()

  await page.evaluate(() => {
    const es = (window as unknown as { EventSource: { instance: EventTarget } }).EventSource
      .instance
    es.dispatchEvent(
      new MessageEvent('message', {
        data: '{"data":{"world_map":{"agents":{"agent-1":[3,4]}}}}',
      }),
    )
  })

  await expect(page.getByTestId('map-display')).toContainText('agent-1: 3, 4')
  await expect(page.getByTestId('summaries')).toContainText('p1')
})
