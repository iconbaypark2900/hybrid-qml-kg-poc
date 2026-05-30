'use client';

import Link from 'next/link';
import { InvestigationParameters } from './InvestigationParameters';
import { CandidateContext } from './CandidateContext';
import { RunPath } from './RunPath';
import { LiveKgPreview } from './LiveKgPreview';
import { EvidencePosture } from './EvidencePosture';
import { Session } from './Session';

export function InitializePage() {
  return (
    <>
      <div className="page-hero">
        <div>
          <div className="step">01 · INITIALIZE</div>
          <h1 className="h1">Define the investigation</h1>
          <p className="lede">
            Set what you&apos;re investigating and how it should run. The choices on this page govern
            every downstream evidence claim — the page exists to make those choices explicit before
            any number appears.
          </p>
        </div>
        <span className="pill">● ready</span>
      </div>

      <div className="grid-7-5">
        <InvestigationParameters />
        <CandidateContext />
      </div>

      <div className="grid-7-5">
        <RunPath />
        <LiveKgPreview />
      </div>

      <div className="grid-7-5">
        <EvidencePosture />
        <Session />
      </div>

      <div className="footer-actions">
        <div style={{ display: 'flex', gap: 8 }}>
          <Link href="/visualize" className="btn">⌥ Visualize evidence</Link>
          <Link href="/operations" className="btn">⌥ Check operations</Link>
        </div>
        <Link href="/experiment" className="btn-primary">Open Experiment →</Link>
      </div>
    </>
  );
}
