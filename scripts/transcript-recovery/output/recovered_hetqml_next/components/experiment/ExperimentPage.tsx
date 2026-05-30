'use client';

import Link from 'next/link';
import { useManifest } from '@/lib/use-manifest';
import { MetricStrip } from './MetricStrip';
import { ModelLeaderboard } from './ModelLeaderboard';
import { DetailedMetrics } from './DetailedMetrics';

export function ExperimentPage() {
  const { state: manifestState } = useManifest();
  const activeModelId =
    manifestState.phase === 'ready' && manifestState.classicalChain
      ? manifestState.classicalChain.model_id
      : null;

  return (
    <>
      <div className="page-hero">
        <div>
          <div className="step">02 · EXPERIMENT</div>
          <h1 className="h1">What this investigation produced</h1>
          <p className="lede">
            The active model produced these numbers against the live evaluation log.
            Every claim below is wired to <code>/status</code> and
            <code> /evaluations</code> — empty states mean the service has no data
            to back the claim, not that the dashboard is broken.
          </p>
        </div>
        <span className="pill">{manifestState.phase === 'ready' ? '● live' : '○ unconfigured'}</span>
      </div>

      <MetricStrip />

      <div className="grid-7-5">
        <ModelLeaderboard activeModelId={activeModelId} />
        <DetailedMetrics activeModelId={activeModelId} />
      </div>

      <div className="footer-actions">
        <div style={{ display: 'flex', gap: 8 }}>
          <Link href="/initialize" className="btn">⌥ Back to Initialize</Link>
          <Link href="/operations" className="btn">⌥ Check operations</Link>
        </div>
        <Link href="/validate" className="btn-primary">Send to Validate →</Link>
      </div>
    </>
  );
}
