import type { Metadata } from 'next';
import { PageStub } from '@/components/shared/PageStub';

export const metadata: Metadata = { title: 'Experiment · Hetionet QML' };

export default function Page() {
  return (
    <PageStub
      step="02 · EXPERIMENT"
      title="What this investigation produced"
      lede="Hybrid pipeline ran the selected investigation. The page lays out where each piece of evidence comes from, which model performed best, why the top candidate surfaced, and whether the experiment passed scientific quality checks."
      todos={[
        'Reactive metric strip (starting point, candidate, evidence source, run path) — pulls from useInvestigationStore',
        'Source Check panel (4 source cards: API status, latest run, candidate source, job source)',
        'Model Leaderboard — uses lib/scoring.ts → getActiveAlgorithms() + scoreAlgorithm()',
        'Benchmark Suite (6 tabs: Classification, Ranking, Calibration, Efficiency, Quantum HW, CV strategy)',
        'Detailed Metrics panel (PR-AUC + ROC + F1 + Brier + MCC + 5-fold CV bars)',
        'Candidate Spotlight (per-pair score + 3 reasons + ranking table)',
        'Scientific Quality Controls — derives from useGuardsStore'
      ]}
      nextHref="/validate"
      nextLabel="Send to Validate"
    />
  );
}
