import type { Metadata } from 'next';
import { PageStub } from '@/components/shared/PageStub';

export const metadata: Metadata = { title: 'Operations · Hetionet QML' };

export default function Page() {
  return (
    <PageStub
      step="SYSTEM · OPERATIONS"
      title="Platform health, run history, and resource posture"
      lede="The systems-side view: are the quantum backends healthy, are upstream data sources fresh, what jobs are running, what's been spent. The Initialize → Visualize pipeline trusts that everything here is green."
      todos={[
        'Metric strip (active jobs, queue depth, success rate, MTD spend)',
        'System Health grid (9 services with status/latency/uptime)',
        'IBM Workload panel — reads token + CRN from settings, parses crn:v1:bluemix:..., shows per-account usage',
        'Quantum Backends detail (ibm_torino, brisbane, kyoto with T1/T2/fidelity)',
        'Active Job Queue table',
        'Recent Job History (last 24h)',
        'Resource Utilization (CPU-hours, Quantum-seconds, GPU-hours, etc.)',
        'Cost & Budget breakdown',
        'Data Sources freshness panel',
        'Alerts & Incidents'
      ]}
      nextHref="/initialize"
      nextLabel="Back to Initialize"
    />
  );
}
