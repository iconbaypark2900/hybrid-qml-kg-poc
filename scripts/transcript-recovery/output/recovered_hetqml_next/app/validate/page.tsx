import type { Metadata } from 'next';
import { PageStub } from '@/components/shared/PageStub';

export const metadata: Metadata = { title: 'Validate · Hetionet QML' };

export default function Page() {
  return (
    <PageStub
      step="03 · VALIDATE"
      title="Decide whether to trust this candidate"
      lede="A high model score is not evidence. This page splits the score into five independently-sourced axes, lays out where each comes from, surfaces what could weaken the candidate, and logs your decision."
      todos={[
        'Reactive metric strip (candidate, model score, composite trust, decision)',
        'Trust Scorecard — D3 radar chart with 5 axes (Clinical, Mechanism, Model, Baseline, Artifact)',
        'Reliability Diagram — D3 calibration curve (predicted vs observed) + Brier/ECE/MCE/LogLoss summary',
        'Reviewer Decision panel (Keep / Review / Reject buttons → log to localStorage)',
        'Skeptic View — generated warnings from compound context + CV variance + integrity guards',
        'Decision History — localStorage-backed audit log with stats cards (keep/review/reject counts + avg trust)'
      ]}
      nextHref="/visualize"
      nextLabel="Visualize evidence"
    />
  );
}
