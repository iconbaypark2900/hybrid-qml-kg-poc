import { Sidebar } from "@/components/sidebar";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col bg-background md:flex-row">
      <aside className="shrink-0 border-outline/10 bg-surface-container-low md:w-60 md:border-r md:border-b-0 border-b p-4">
        <div className="mb-8">
          <p className="font-headline text-lg font-semibold tracking-tight text-primary">
            Hybrid QML-KG
          </p>
          <p className="text-xs text-on-surface-variant">
            Biomedical link prediction
          </p>
        </div>
        <Sidebar />
      </aside>
      <main className="min-h-0 flex-1 overflow-auto p-6 md:p-10">{children}</main>
    </div>
  );
}
