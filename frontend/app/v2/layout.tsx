import { Suspense } from "react";
import { V2Shell } from "@/components/v2/v2-shell";

export default function V2Layout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<V2ShellFallback />}>
      <V2Shell>{children}</V2Shell>
    </Suspense>
  );
}

function V2ShellFallback() {
  return (
    <div className="min-h-screen bg-background p-6 text-on-surface">
      Loading v2 dashboard...
    </div>
  );
}

