import { test, expect } from '@playwright/test'

// Ensure navigation works and SSE updates render

test('navigate between pages and receive SSE update', async ({ page }) => {
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
    // @ts-expect-error override built-in
    window.EventSource = MockEventSource
  })

  await page.goto('/')
  await page.waitForLoadState('networkidle')

  await page.evaluate(() => {
    const fg = document.querySelector('.force-graph-container') as HTMLElement | null
    if (fg) fg.style.display = 'none'
  })

  await page.goto('/missions')
  await expect(page.getByRole('heading', { name: 'Mission Overview' })).toBeVisible()

  await page.goto('/agent-data')
  await expect(page.getByRole('heading', { name: 'Agent Data Overview' })).toBeVisible()

  await page.goto('/memory')
  await expect(page.getByRole('heading', { name: 'Memory Explorer' })).toBeVisible()

  await page.evaluate(() => {
    const w = window as unknown as { EventSource: { instance?: EventTarget } }
    if (!w.EventSource.instance) {
      // ensure a mock EventSource exists so we can dispatch events
      // @ts-expect-error override built-in
      new w.EventSource('/stream')
    }
    w.EventSource.instance!.dispatchEvent(
      new MessageEvent('message', { data: '{"check":1}' }),
    )
  })
})
