import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.pw.ts',
  use: {
    baseURL: 'http://localhost:4173',
  },
  webServer: {
    command: 'npx http-server dist -p 4173',
    port: 4173,
    reuseExistingServer: false,
  },
});
