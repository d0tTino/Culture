import { test, expect } from '@playwright/test'

test.skip(true, 'e2e tests are skipped in this environment')

// Register a dummy widget via the backend and verify it renders in the dashboard

test('register widget through backend', async ({ page }) => {
  // mock backend registration endpoint
  await page.route('**/api/register_widget', async (route, request) => {
    const data = await request.postDataJSON()
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ widgets: [{ name: data.name, scriptUrl: '/dummy_widget.js' }] }),
    })
  })

  // backend widgets list includes the dummy widget
  await page.route('**/api/widgets', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ widgets: [{ name: 'DummyWidget', scriptUrl: '/dummy_widget.js' }] }),
    })
  })

  // remote widget script registers a simple component
  await page.route('**/dummy_widget.js', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/javascript',
      body: `import React from 'react';
import { registerWidget } from 'culture-ui/lib';
const Dummy = () => <div data-testid="dummy-widget">Dummy Widget</div>;
registerWidget('DummyWidget', Dummy);
`,
    })
  })

  await page.goto('/')

  // register widget via backend and load it
  await page.evaluate(async () => {
    const mod = await import('culture-ui/lib')
    await mod.registerWidgetBackend({ name: 'DummyWidget', scriptUrl: '/dummy_widget.js' })
    await mod.loadRemoteWidgets()
  })

  await expect(page.getByTestId('dummy-widget')).toBeVisible()
})
