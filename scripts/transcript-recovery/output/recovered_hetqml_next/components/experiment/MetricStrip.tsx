'use client';

import { useInvestigationStore } from '@/lib/store';
import { useManifest } from '@/lib/use-manifest';
import { latestEvaluation, useEvaluations } from '@/lib/use-evaluations';

export function MetricStrip() {
  const compound = useInvestigationStore((s) => s.compound);
  const disease = useInvestigationStore((s) => s.disease);
  const runPath = useInvestigationStore((s) => s.runPath);
  const { state: manifest } = useManifest();

  const activeModelId =
    manifest.phase === 'ready' && manifest.classicalChain
      ? manifest.classicalChain.model_id
      : null;

  const { state: evals } = useEvaluations({
    modelManifestId: activeModelId,
    limit: 1,
  });

  const latest = evals.phase === 'ready' ? latestEvaluation(evals.records) : null;
  const prAuc = latest?.metrics.pr_auc ?? latest?.metrics.classical_pr_auc;

  return (
    <div
      className="panel"
      style={{
        marginBottom: 16,
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 12,
        padding: 14,
      }}
    >
      <Stat
        label="STARTING POINT"
        primary={compound.name}
        secondary={`${compound.id} · ${compound.cat}`}
      />
      <Stat
        label="CANDIDATE TARGET"
        primary={disease.name}
        secondary={`${disease.id} · ${disease.cat}`}
      />
      <Stat
        label="ACTIVE MODEL"
        primary={
          manifest.phase === 'ready'
            ? manifest.classicalModel?.model_type ?? 'no model'
            : '...'
        }
        secondary={
          activeModelId
            ? activeModelId
            : manifest.phase === 'unconfigured'
              ? 'configure API in /settings'
              : 'awaiting backend'
        }
      />
      <Stat
        label={`LATEST PR-AUC · ${runPath}`}
        primary={
          evals.phase === 'loading'
            ? '...'
            : evals.phase === 'unconfigured'
              ? '—'
              : evals.phase === 'error'
                ? 'error'
                : prAuc !== undefined
                  ? prAuc.toFixed(3)
                  : 'no eval'
        }
        secondary={
          latest
            ? `${new Date(latest.created_at * 1000).toISOString().slice(0, 10)} · cv=${latest.cv_folds ?? '?'}`
            : evals.phase === 'ready'
              ? 'no evaluation recorded for this model'
              : evals.phase === 'error'
                ? evals.error.message.slice(0, 32)
                : ''
        }
      />
    </div>
  );
}

function Stat({
  label, primary, secondary,
}: { label: string; primary: string; secondary: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div
        style={{
          fontFamily: 'monospace',
          fontSize: 9,
          letterSpacing: '0.5px',
          color: 'var(--faint)',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: 'Georgia, serif',
          fontSize: 18,
          fontWeight: 600,
          color: 'var(--ink)',
          lineHeight: 1.2,
        }}
      >
        {primary}
      </div>
      <div
        style={{
          fontFamily: 'monospace',
          fontSize: 10,
          color: 'var(--muted)',
          wordBreak: 'break-all',
        }}
      >
        {secondary}
      </div>
    </div>
  );
}
