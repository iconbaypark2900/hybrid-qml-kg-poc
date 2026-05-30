'use client';

import { latestEvaluation, useEvaluations } from '@/lib/use-evaluations';

const METRIC_ORDER = [
  ['pr_auc', 'PR-AUC'],
  ['roc_auc', 'ROC-AUC'],
  ['f1', 'F1'],
  ['accuracy', 'Accuracy'],
  ['precision', 'Precision'],
  ['recall', 'Recall'],
  ['brier', 'Brier'],
  ['mcc', 'MCC'],
  ['ece', 'ECE'],
] as const;

export function DetailedMetrics({ activeModelId }: { activeModelId: string | null }) {
  const { state: evals } = useEvaluations({ modelManifestId: activeModelId, limit: 10 });

  if (evals.phase === 'unconfigured') {
    return (
      <Panel title="Detailed metrics">
        <Empty
          headline="API not configured"
          detail="Set an API key in /settings to fetch /evaluations."
        />
      </Panel>
    );
  }
  if (evals.phase === 'loading') {
    return <Panel title="Detailed metrics"><Empty headline="loading…" detail="" /></Panel>;
  }
  if (evals.phase === 'error') {
    return (
      <Panel title="Detailed metrics">
        <Empty
          headline="couldn't load evaluations"
          detail={`${evals.error.code}: ${evals.error.message.slice(0, 80)}`}
          tone="sienna"
        />
      </Panel>
    );
  }

  const latest = latestEvaluation(evals.records);
  if (!latest) {
    return (
      <Panel title="Detailed metrics">
        <Empty
          headline="no evaluation recorded for this model"
          detail={
            activeModelId
              ? `Run benchmarking against ${activeModelId} and POST results to /evaluations to populate this panel.`
              : 'No active manifest chain — see /operations.'
          }
        />
      </Panel>
    );
  }

  const present: Array<[string, string, number]> = [];
  for (const [key, label] of METRIC_ORDER) {
    // Try prefixed (classical_pr_auc, quantum_pr_auc) first, then bare.
    const candidates = [`classical_${key}`, `quantum_${key}`, key];
    for (const c of candidates) {
      if (latest.metrics[c] !== undefined) {
        present.push([c, `${label}${c.startsWith('classical_') ? ' (cls)' : c.startsWith('quantum_') ? ' (q)' : ''}`, latest.metrics[c]]);
      }
    }
  }

  return (
    <Panel title="Detailed metrics">
      <div style={{ marginBottom: 10, fontSize: 11, color: 'var(--muted)' }}>
        evaluation_id: <code style={{ fontFamily: 'monospace', fontSize: 10 }}>{latest.evaluation_id}</code>
        <br />
        recorded: {new Date(latest.created_at * 1000).toISOString().replace('T', ' ').slice(0, 19)} UTC
        {latest.cv_folds ? ` · ${latest.cv_folds}-fold CV` : ''}
        {latest.notes ? <><br />notes: <em>{latest.notes}</em></> : null}
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 8,
          border: '1px solid var(--border-soft)',
          borderRadius: 4,
          overflow: 'hidden',
        }}
      >
        {present.length === 0 ? (
          <div style={{ padding: 12, fontSize: 12, color: 'var(--faint)', gridColumn: '1 / -1' }}>
            evaluation has no recognized metric keys
          </div>
        ) : (
          present.map(([k, label, v]) => (
            <div
              key={k}
              style={{
                padding: '8px 12px',
                borderRight: '1px solid var(--border-soft)',
                borderBottom: '1px solid var(--border-soft)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'baseline',
              }}
            >
              <span style={{ fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace', textTransform: 'uppercase' }}>
                {label}
              </span>
              <span style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>
                {v.toFixed(4)}
              </span>
            </div>
          ))
        )}
      </div>
      {evals.records.length > 1 ? (
        <div style={{ marginTop: 10, fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace' }}>
          {evals.records.length} historical evaluations for this model
        </div>
      ) : null}
    </Panel>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · DETAILED METRICS</div>
          <div className="panel-title">{title}</div>
        </div>
        <span className="badge">Live</span>
      </div>
      {children}
    </div>
  );
}

function Empty({
  headline, detail, tone = 'muted',
}: { headline: string; detail: string; tone?: 'muted' | 'sienna' }) {
  const color = tone === 'sienna' ? 'var(--sienna)' : 'var(--muted)';
  return (
    <div
      style={{
        padding: 24,
        textAlign: 'center',
        border: '1px dashed var(--border-soft)',
        borderRadius: 4,
        color,
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 600 }}>{headline}</div>
      {detail ? (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--faint)', lineHeight: 1.5 }}>
          {detail}
        </div>
      ) : null}
    </div>
  );
}
