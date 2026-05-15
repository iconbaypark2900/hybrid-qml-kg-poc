import { IBMQuantumSettings } from "@/components/ibm-quantum-settings";
import { PageHero } from "@/components/v2/v2-shell";

export default function V2SettingsPage() {
  return (
    <div className="space-y-5">
      <PageHero eyebrow="Configuration" title="IBM Quantum settings">
        Save tenant-scoped IBM Quantum credentials, verify runtime access, and
        keep the v2 workflow connected to real hardware credentials without
        exposing tokens in the browser.
      </PageHero>

      <IBMQuantumSettings />
    </div>
  );
}
