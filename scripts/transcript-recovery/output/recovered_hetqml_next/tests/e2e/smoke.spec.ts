// Smoke tests for the partner-facing flow: Initialize → Experiment → Validate.
// Each test runs against a live dev server + hetqml-service backend.
//
// Prerequisites:
//   - dev server on :3001 (env NEXT_PUBLIC_HETQML_API_URL=http://localhost:8002)
//   - service on :8002 with the demo tenant key configured
//   - artifacts/ contains a real classical chain
//
// Run with:
//   PLAYWRIGHT_API_KEY=<plaintext> npm run e2e

import { test, expect, type Page } from '@playwright/test';

const API_KEY = process.env.PLAYWRIGHT_API_KEY ?? 'TzUrgb6C0prfM-YADAOZcBAn8RFbqmGiY7P5S7t3OiM';

async function setKey(page: Page) {
  await page.goto('/');
  await page.evaluate(
    (k) => window.localStorage.setItem('hetqml_api_key', k),
    API_KEY,
  );
}

test.describe('unconfigured state', () => {
  test('sidebar shows API not configured before key is set', async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => window.localStorage.removeItem('hetqml_api_key'));
    await page.goto('/initialize');
    await expect(page.locator('.sidebar .status-bar')).toContainText(
      'API not configured',
      { timeout: 5_000 },
    );
  });
});

test.describe('Initialize page', () => {
  test.beforeEach(async ({ page }) => {
    await setKey(page);
    await page.goto('/initialize');
  });

  test('onboarding checklist runs and reaches all-ok', async ({ page }) => {
    // Wait for checks to complete (predict step is the slowest).
    await expect(page.locator('text=Service connected')).toBeVisible({ timeout: 15_000 });
  });

  test('sidebar badge shows live model_id', async ({ page }) => {
    const sidebar = page.locator('.sidebar .status-bar');
    await expect(sidebar).toContainText(/MDL-/, { timeout: 10_000 });
    await expect(sidebar).toContainText('logistic_regression');
  });

  test('investigation parameters are interactive', async ({ page }) => {
    await expect(page.locator('select').first()).toBeVisible();
  });
});

test.describe('Settings page', () => {
  test('Test connection succeeds with valid key', async ({ page }) => {
    await setKey(page);
    await page.goto('/settings');
    await page.click('text=Test connection');
    await expect(page.locator('text=connection ok')).toBeVisible({ timeout: 10_000 });
  });

  test('Clear key removes API key', async ({ page }) => {
    await setKey(page);
    await page.goto('/settings');
    await page.click('text=Clear key');
    const key = await page.evaluate(() => window.localStorage.getItem('hetqml_api_key'));
    expect(key).toBeNull();
  });
});

test.describe('Experiment page', () => {
  test.beforeEach(async ({ page }) => {
    await setKey(page);
    await page.goto('/experiment');
  });

  test('metric strip shows live PR-AUC', async ({ page }) => {
    await expect(page.locator('text=ACTIVE MODEL')).toBeVisible();
    await expect(page.locator('text=LATEST PR-AUC')).toBeVisible();
    // Wait for the value to land (not "..." or "—").
    await expect(page.locator('text=/^0\\.\\d{3}/')).toBeVisible({ timeout: 10_000 });
  });

  test('leaderboard shows live + inactive split', async ({ page }) => {
    await expect(page.locator('text=/LIVE · \\d+/')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('text=/IN CATALOG · NOT ACTIVE · \\d+/')).toBeVisible();
  });

  test('detailed metrics panel renders evaluation_id', async ({ page }) => {
    await expect(page.locator('text=evaluation_id:')).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Validate page', () => {
  test.beforeEach(async ({ page }) => {
    await setKey(page);
    await page.goto('/validate');
  });

  test('skeptic view renders concerns', async ({ page }) => {
    await expect(page.locator('text=What could weaken this candidate')).toBeVisible();
    // Either there's at least one concern, or the empty-state info concern fires.
    await expect(page.locator('text=/concern/i')).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Operations page', () => {
  test.beforeEach(async ({ page }) => {
    await setKey(page);
    await page.goto('/operations');
  });

  test('component health probes render', async ({ page }) => {
    await expect(page.locator('text=COMPONENT HEALTH')).toBeVisible();
    await expect(page.locator('text=entity_mappings')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('text=classical_model')).toBeVisible();
  });

  test('manifest panel shows active chain', async ({ page }) => {
    await expect(page.locator('text=ACTIVE MANIFEST')).toBeVisible();
    await expect(page.locator('text=model_id')).toBeVisible({ timeout: 10_000 });
  });
});
