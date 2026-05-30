import type { Metadata } from 'next';
import { PageStub } from '@/components/shared/PageStub';

export const metadata: Metadata = { title: 'Settings · Hetionet QML' };

export default function Page() {
  return (
    <PageStub
      step="SYSTEM · SETTINGS"
      title="Personalize the dashboard and platform defaults"
      lede="Settings persist in browser localStorage and apply to every investigation, run, and export. Changes save automatically — no submit button. Hover any setting for the side-effect description."
      todos={[
        'Profile (reviewer name, role, organization, contact email)',
        'Appearance (theme, accent, density, reduce-motion)',
        'Pipeline Defaults (default run path, default metaedge, hard-negative ratio, strict posture)',
        'Quantum Preferences (default backend, shots/circuit, job timeout, ZNE on/off)',
        'IBM Quantum Connection — API token + CRN with validate button',
        'Notifications channels',
        'Privacy & Telemetry toggles + decision retention',
        'Other API Keys & Integrations',
        'Keyboard Shortcuts reference',
        'About & Diagnostics'
      ]}
      nextHref="/initialize"
      nextLabel="Back to Initialize"
    />
  );
}
