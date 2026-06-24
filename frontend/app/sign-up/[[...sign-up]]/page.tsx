import { getClerkPublishableKey } from "@/lib/clerk-config";

export default async function SignUpPage() {
  const publishableKey = getClerkPublishableKey();
  if (!publishableKey) {
    return (
      <main className="min-h-screen bg-background p-8 text-on-surface">
        <h1 className="font-headline text-3xl font-semibold">Sign up disabled</h1>
        <p className="mt-3 max-w-xl text-on-surface-variant">
          Clerk environment variables are not configured in this development environment.
        </p>
      </main>
    );
  }

  const { SignUp } = await import("@clerk/nextjs");
  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-8">
      <SignUp />
    </main>
  );
}
