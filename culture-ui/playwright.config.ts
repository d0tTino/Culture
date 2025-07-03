import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.pw.ts',
  use: {
    baseURL: 'http://localhost:4175',
  },
  webServer: {
    command:
      'pnpm --filter culture-ui preview --port 4175 --strictPort --host 127.0.0.1',
    port: 4175,
    reuseExistingServer: !process.env.CI,
  },
});
