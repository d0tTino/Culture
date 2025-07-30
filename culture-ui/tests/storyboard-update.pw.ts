import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

// Verify Storyboard updates when events stream via /stream/events

test.beforeEach(async ({ page }) => {
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
    // @ts-expect-error override EventSource
    window.EventSource = MockEventSource as unknown as typeof EventSource
  })

  await page.route('/api/agent_stats', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ agents: {} }),
    })
  })
})

test('storyboard receives mood update', async ({ page }) => {
  await page.goto('/storyboard')
  await expect(page.getByRole('heading', { name: 'Storyboard' })).toBeVisible()

  // ensure SSE connected to correct endpoint
  expect(
    await page.evaluate(
      () => (window as unknown as { EventSource: { instance: { url: string } } }).EventSource.instance.url,
    ),
  ).toBe('/stream/events')

  await page.evaluate(() => {
    const es = (window as unknown as { EventSource: { instance: EventTarget } }).EventSource.instance
    es.dispatchEvent(
      new MessageEvent('message', {
        data: '{"data":{"world_map":{"agents":{"agent-1":[1,2]}},"agents":[{"agent_id":"agent-1","mood":0.5}]}}',
      }),
    )
  })

  await expect(page.getByTestId('map-info')).toContainText('mood 0.5')
})
