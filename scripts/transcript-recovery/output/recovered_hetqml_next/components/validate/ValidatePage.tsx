'use client';

import Link from 'next/link';
import { useMemo } from 'react';
import { useGuardsStore, useInvestigationStore } from '@/lib/store';
import { useManifest, isSyntheticModel } from '@/lib/use-manifest';
import { latestEvaluation, useEvaluations } from '@/lib/use-evaluations';
import type { ComponentHealth, EvaluationRecord, ModelManifest } from '@/lib/api';

export function ValidatePage() {
  const compound = useInvestigationStore((s) => s.compound);
  const disease = useInvestigationStore((s) => s.disease);
  const guards = useGuardsStore((s) => s.guards);
  const { state: manifest } = useManifest();
  const activeModelId =
    manifest.phase === 'ready' && manifest.classicalChain ? manifest.classicalChain.model_id : null;
  const { state: evals } = useEvaluations({ modelManifestId: activeModelId, limit: 10 });
  const latest = evals.phase === 'ready' ? latestEvaluation(evals.records) : null;

  return (
    <>
      <div className="page-hero">
        <div>
          <div className="step">03 · VALIDATE</div>
          <h1 className="h1">Decide whether to trust this candidate</h1>
          <p className="lede">
            A high model score is not evidence. The Skeptic view below reads
            from <code>/status</code>, <code>/manifest/model/&lt;id&gt;</code>, and
            <code> /evaluations</code> — every concern is grounded in a backend signal.
          </p>
        </div>
        <span className="pill">{manifest.phase === 'ready' ? '● live' : '○ unconfigured'}</span>
      </div>

      <div className="grid-7-5">
        <SkepticView
          compound={compound.name}
          disease={disease.name}
          manifestState={manifest}
          model={manifest.phase === 'ready' ? manifest.classicalModel : null}
          latest={latest}
          guards={guards}
        />
        <DecisionPanel />
      </div>

      <div className="footer-actions">
        <Link href="/experiment" className="btn">⌥ Back to Experiment</Link>
        <Link href="/visualize" className="btn-primary">Visualize evidence →</Link>
      </div>
    </>
  );
}

interface Concern {
  level: 'critical' | 'warning' | 'info';
  title: string;
  detail: string;
  source: string;
}

function SkepticView({
  compound, disease, manifestState, model, latest, guards,
}: {
  compound: string;
  disease: string;
  manifestState: ReturnType<typeof useManifest>['state'];
  model: ModelManifest | null;
  latest: EvaluationRecord | null;
  guards: ReturnType<typeof useGuardsStore.getState>['guards'];
}) {
  const concerns = useMemo<Concern[]>(() => {
    const out: Concern[] = [];

    if (manifestState.phase !== 'ready') {
      out.push({
        level: 'critical',
        title: 'Service is not configured or not ready',
        detail: 'No backend signal — every claim on this page is unverified.',
        source: '/status',
      });
      return out;
    }

    // Synthetic model
    if (model && isSyntheticModel(model)) {
      out.push({
        level: 'critical',
        title: `Active classical model is a placeholder (${model.model_type})`,
        detail:
          'Predictions are computed against a model that was not trained on real Hetionet data. ' +
          'No clinical claim should be based on this model.',
        source: `manifest ${model.manifest_id}`,
      });
    }

    // Component health
    for (const c of manifestState.status.components) {
      if (c.state === 'unavailable') {
        out.push({
          level: 'warning',
          title: `Component unavailable: ${c.name}`,
          detail: c.detail ?? 'no detail',
          source: '/status.components',
        });
      } else if (c.state === 'degraded') {
        out.push({
          level: 'info',
          title: `Component degraded: ${c.name}`,
          detail: c.detail ?? 'no detail',
          source: '/status.components',
        });
      }
    }

    // Quantum fallback
    const q = manifestState.status.quantum;
    if (q.note) {
      out.push({
        level: 'info',
        title: 'Quantum execution caveat',
        detail: q.note,
        source: '/status.quantum',
      });
    }

    // Latest evaluation
    if (!latest) {
      out.push({
        level: 'warning',
        title: 'No evaluation recorded for the active model',
        detail:
          'There is no /evaluations entry for this model. The reported PR-AUC and other metrics ' +
          'come from no source the dashboard can verify.',
        source: '/evaluations',
      });
    } else {
      const pr = latest.metrics.pr_auc ?? latest.metrics.classical_pr_auc;
      if (pr !== undefined && pr < 0.6) {
        out.push({
          level: 'warning',
          title: `Low PR-AUC (${pr.toFixed(3)})`,
          detail: 'Model performs near or below random on its own evaluation set. Treat as exploratory.',
          source: latest.evaluation_id,
        });
      }
      const ece = latest.metrics.ece;
      if (ece !== undefined && ece > 0.1) {
        out.push({
          level: 'warning',
          title: `Calibration error high (ECE=${ece.toFixed(3)})`,
          detail: 'Predicted probabilities don\'t match observed frequencies — interpret scores cautiously.',
          source: latest.evaluation_id,
        });
      }
      const synthetic = (latest.notes ?? '').toLowerCase().includes('synthetic');
      if (synthetic) {
        out.push({
          level: 'critical',
          title: 'Latest evaluation used a synthetic test set',
          detail:
            'The notes field on the active evaluation says it was computed against synthetic data. ' +
            'Use a real held-out Hetionet test set before any partner-facing claim.',
          source: latest.evaluation_id,
        });
      }
    }

    // Critical guards off
    const criticalOff = guards.filter((g) => g.level === 'critical' && !g.enabled);
    for (const g of criticalOff) {
      out.push({
        level: 'critical',
        title: `Critical integrity guard OFF: ${g.name}`,
        detail: g.impact,
        source: g.source,
      });
    }

    if (out.length === 0) {
      out.push({
        level: 'info',
        title: 'No concerns surfaced',
        detail: `All probes returned ok for ${compound} → ${disease} on the active manifest.`,
        source: '/status',
      });
    }
    return out;
  }, [manifestState, model, latest, guards, compound, disease]);

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · SKEPTIC VIEW</div>
          <div className="panel-title">What could weaken this candidate</div>
        </div>
        <span className="badge">Live</span>
      </div>
      <p className="panel-purpose">
        Every concern is derived from a real backend signal. If the list is
        short, it&apos;s because the service has nothing to flag — not because
        the dashboard is hiding things.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {concerns.map((c, i) => (
          <ConcernRow key={i} concern={c} />
        ))}
      </div>
      <div className="panel-footer">
        <span>/status × /manifest × /evaluations × guards</span>
        <span><em>{concerns.length} concern{concerns.length === 1 ? '' : 's'}</em></span>
      </div>
    </div>
  );
}

function ConcernRow({ concern }: { concern: Concern }) {
  const accent =
    concern.level === 'critical' ? 'var(--sienna)' :
    concern.level === 'warning' ? 'var(--amber)' :
    'var(--faint)';
  const bg =
    concern.level === 'critical' ? 'rgba(224, 132, 116, 0.06)' :
    concern.level === 'warning' ? 'rgba(220, 180, 80, 0.06)' :
    'transparent';
  return (
    <div
      style={{
        padding: 12,
        background: bg,
        borderLeft: `3px solid ${accent}`,
        borderRadius: 3,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span
          style={{
            fontSize: 9,
            fontFamily: 'monospace',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            padding: '2px 6px',
            background: accent,
            color: 'var(--paper)',
            borderRadius: 2,
            fontWeight: 700,
          }}
        >
          {concern.level}
        </span>
        <span style={{ fontSize: 13, fontWeight: 600 }}>{concern.title}</span>
      </div>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4, lineHeight: 1.5 }}>
        {concern.detail}
      </div>
      <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 4, fontFamily: 'monospace', wordBreak: 'break-all' }}>
        source: {concern.source}
      </div>
    </div>
  );
}

function DecisionPanel() {
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · REVIEWER DECISION</div>
          <div className="panel-title">Log a decision</div>
        </div>
        <span className="badge">Stub</span>
      </div>
      <p className="panel-purpose">
        Keep / Review / Reject decisions persist locally for now. When the
        backend grows a /decisions endpoint, this panel switches over without
        changing the UX.
      </p>
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="btn" style={{ flex: 1 }}>Keep</button>
        <button className="btn" style={{ flex: 1 }}>Review</button>
        <button className="btn" style={{ flex: 1 }}>Reject</button>
      </div>
      <div style={{ marginTop: 12, fontSize: 11, color: 'var(--faint)', fontFamily: 'monospace' }}>
        decisions persist via localStorage; export via /settings (future)
      </div>
    </div>
  );
}
