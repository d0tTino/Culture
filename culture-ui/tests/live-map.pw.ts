import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')


test('world map page updates from events', async ({ page }) => {
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

  await page.goto('/world-map')
  await expect(page.getByRole('heading', { name: 'World Map' })).toBeVisible()

  await page.evaluate(() => {
    const es = (window as unknown as { EventSource: { instance: EventTarget } }).EventSource
      .instance
    es.dispatchEvent(
      new MessageEvent('message', {
        data: '{"data":{"world_map":{"agents":{"agent-1":[3,4]},"resources":{"[0,0]":{"wood":2}}}}}',
      }),
    )
  })

  await expect(page.getByTestId('world-map')).toContainText('agent-1: 3, 4')
  await expect(page.getByTestId('world-map')).toContainText('[0,0]: wood:2')
})
