import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

// Verify live updates from /stream/events render in Memory Explorer and Storyboard

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

  await page.route('/api/agents/agent-1/semantic_summaries', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ summaries: ['test summary'] }),
    })
  })
})


test('stream events update pages', async ({ page }) => {
  await page.goto('/memory')
  await expect(page.getByRole('heading', { name: 'Memory Explorer' })).toBeVisible()

  // confirm SSE connected to correct endpoint
  expect(await page.evaluate(() => (window as any).EventSource.instance.url)).toBe('/stream/events')

  await page.evaluate(() => {
    const es = (window as any).EventSource.instance
    es.dispatchEvent(
      new MessageEvent('message', {
        data: '{"type":"breakpoint_hit","data":{"tags":["nsfw"]}}',
      }),
    )
  })

  await expect(page.getByTestId('toast')).toContainText('nsfw')

  await page.goto('/storyboard')
  await expect(page.getByRole('heading', { name: 'Storyboard' })).toBeVisible()

  await page.evaluate(() => {
    const ws = (window as any).WebSocket.instance
    ws.dispatchEvent(
      new MessageEvent('message', {
        data: '{"data":{"world_map":{"agents":{"agent-1":[1,2]}}}}',
      }),
    )
    ws.dispatchEvent(
      new MessageEvent('message', {
        data: '{"type":"memory_prune","data":{"step":1}}',
      }),
    )
  })

  await expect(page.getByTestId('map-info')).toContainText('agent-1: 1, 2')
  await expect(page.getByTestId('memory-events')).toContainText('memory_prune (step 1)')
})
