import { defineConfig, devices } from '@playwright/test';

// E2E config. Assumes the dev server is already running on :3001 with
// NEXT_PUBLIC_HETQML_API_URL pointing at a hetqml-service instance.
// Run via:
//     npm run e2e          # headless
//     npm run e2e:ui       # Playwright UI

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,                  // tests share state via localStorage
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  timeout: 30_000,
  expect: { timeout: 5_000 },

  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:3001',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
    },
    // Add firefox / webkit as desired:
    // { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    // { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],
});
