import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

// mock SSE and initial fetch

test.beforeEach(async ({ page }) => {
  await page.route('/api/agents/agent-1/semantic_summaries', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ summaries: ['initial summary'] }),
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
    // @ts-expect-error override
    window.EventSource = MockEventSource as unknown as typeof EventSource
  })
})

test('agent memories update via SSE', async ({ page }) => {
  await page.goto('/agent-memories')
  await expect(page.getByRole('heading', { name: 'Agent Memories' })).toBeVisible()
  await expect(page.getByText('initial summary')).toBeVisible()

  await page.evaluate(() => {
    const es = (
      window as unknown as { EventSource: { instance: EventTarget } }
    ).EventSource.instance
    es.dispatchEvent(
      new MessageEvent('message', { data: '{"summaries":["new summary"]}' })
    )
  })

  await expect(page.getByTestId('agent-memories')).toContainText('new summary')
})
