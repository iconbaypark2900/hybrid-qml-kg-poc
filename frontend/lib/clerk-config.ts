export function getClerkPublishableKey(): string | undefined {
  const key = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY?.trim();
  if (!key) {
    return undefined;
  }

  const hasExpectedPrefix = key.startsWith("pk_test_") || key.startsWith("pk_live_");
  const looksLikePlaceholder = /dummy|placeholder|changeme|example/i.test(key);
  if (!hasExpectedPrefix || key.length < 24 || looksLikePlaceholder) {
    return undefined;
  }

  return key;
}
