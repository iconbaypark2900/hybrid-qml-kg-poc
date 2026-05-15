import { IBMQuantumSettings } from "@/components/ibm-quantum-settings";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          Settings
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-on-surface-variant">
          Configure tenant-scoped integrations for quantum execution. IBM tokens
          are stored server-side and are never returned to the browser.
        </p>
      </header>

      <IBMQuantumSettings />
    </div>
  );
}
