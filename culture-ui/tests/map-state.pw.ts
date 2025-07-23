import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

// mock EventSource

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
    // @ts-expect-error override
    window.EventSource = MockEventSource as unknown as typeof EventSource
  })
})

test('map state updates via SSE', async ({ page }) => {
  await page.goto('/map-state')
  await expect(page.getByRole('heading', { name: 'Map State' })).toBeVisible()

  await page.evaluate(() => {
    const es = (
      window as unknown as { EventSource: { instance: EventTarget } }
    ).EventSource.instance
    es.dispatchEvent(
      new MessageEvent('message', {
        data: '{"data":{"world_map":{"agents":{"agent-1":[3,4]}}}}',
      })
    )
  })

  await expect(page.getByTestId('map-state')).toContainText('agent-1: 3, 4')
})
