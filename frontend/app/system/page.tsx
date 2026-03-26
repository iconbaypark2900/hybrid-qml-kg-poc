import { SystemStatusPanel } from "@/components/system-status-panel";

export default function SystemPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          System status
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Live health and readiness from the FastAPI orchestrator.
        </p>
      </header>
      <SystemStatusPanel />
    </div>
  );
}
