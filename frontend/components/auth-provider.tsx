import type { ReactNode } from "react";
import { getClerkPublishableKey } from "@/lib/clerk-config";

export async function AuthProvider({ children }: { children: ReactNode }) {
  const publishableKey = getClerkPublishableKey();
  if (!publishableKey) {
    return <>{children}</>;
  }

  const { ClerkProvider } = await import("@clerk/nextjs");
  return <ClerkProvider publishableKey={publishableKey}>{children}</ClerkProvider>;
}
