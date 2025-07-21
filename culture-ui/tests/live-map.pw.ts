import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')


test('live map widget updates from map events', async ({ page }) => {
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

  await page.goto('/')
  await expect(page.getByTestId('live-map')).toBeVisible()

  await page.evaluate(() => {
    const es = (window as unknown as { EventSource: { instance: EventTarget } }).EventSource
      .instance
    es.dispatchEvent(
      new MessageEvent('message', {
        data: '{"type":"map_change","data":{"world_map":{"agents":{"agent-1":[5,6]}}}}',
      }),
    )
  })

  await expect(page.getByTestId('live-map')).toContainText('agent-1: 5, 6')
})
