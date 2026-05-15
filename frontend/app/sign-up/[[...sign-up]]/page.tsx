import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) {
    return (
      <main className="min-h-screen bg-background p-8 text-on-surface">
        <h1 className="font-headline text-3xl font-semibold">Sign up disabled</h1>
        <p className="mt-3 max-w-xl text-on-surface-variant">
          Clerk environment variables are not configured in this development environment.
        </p>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-8">
      <SignUp />
    </main>
  );
}
