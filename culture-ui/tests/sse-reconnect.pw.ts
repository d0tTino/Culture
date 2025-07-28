import { test, expect } from '@playwright/test'

test.describe('sse reconnect', () => {
  test.skip(true, 'e2e tests are skipped in this environment')

// Verify WebSocket fallback when SSE connection fails

test.beforeEach(async ({ page }) => {
  await page.route('/api/agents/agent-1/semantic_summaries', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ summaries: [] }),
    })
  })

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
    class MockWebSocket extends EventTarget {
      static instance: MockWebSocket
      url: string
      constructor(url: string) {
        super()
        this.url = url
        MockWebSocket.instance = this
      }
      close() {}
      send() {}
    }
    // @ts-expect-error override globals
    window.EventSource = MockEventSource as unknown as typeof EventSource
    // @ts-expect-error override globals
    window.WebSocket = MockWebSocket as unknown as typeof WebSocket
  })
})

test('fallback to websocket on SSE error', async ({ page }) => {
  await page.goto('/memory')
  await expect(page.getByRole('heading', { name: 'Memory Explorer' })).toBeVisible()

  // SSE should connect initially
  expect(
    await page.evaluate(
      () =>
        (window as unknown as { EventSource: { instance: { url: string } } }).EventSource
          .instance.url,
    ),
  ).toBe('/stream/events')

  await page.evaluate(() => {
    const es = (window as unknown as { EventSource: { instance: EventTarget } }).EventSource
      .instance
    es.dispatchEvent(new Event('error'))
  })

  expect(
    await page.evaluate(
      () =>
        (window as unknown as { WebSocket: { instance: { url: string } } }).WebSocket.instance.url,
    ),
  ).toBe('/ws/events')

  await page.evaluate(() => {
    const ws = (window as unknown as { WebSocket: { instance: WebSocket } }).WebSocket.instance
    ws.dispatchEvent(
      new MessageEvent('message', {
        data: '{"type":"breakpoint_hit","data":{"tags":["fallback"]}}',
      }),
    )
  })

  await expect(page.getByTestId('toast')).toContainText('fallback')
})
})
