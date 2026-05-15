"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/sidebar";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  if (pathname.startsWith("/v2")) {
    return <>{children}</>;
  }

  return (
    <div className="journey-shell">
      <aside className="journey-sidebar">
        <Sidebar />
      </aside>
      <main className="journey-main">
        <div className="journey-topbar mb-4">
          <p className="text-xs text-on-surface-variant">
            Guided discovery playground: choose a starting point, change the setup,
            validate the evidence, then inspect the visuals.
          </p>
          <div className="hidden gap-2 lg:flex">
            <Link className="journey-button" href="/">
              Start
            </Link>
            <Link className="journey-button" href="/experiments">
              Experiment
            </Link>
            <Link className="journey-button" href="/validation">
              Validation
            </Link>
            <Link className="journey-button primary" href="/visualization">
              Visual
            </Link>
          </div>
        </div>
        {children}
      </main>
    </div>
  );
}
