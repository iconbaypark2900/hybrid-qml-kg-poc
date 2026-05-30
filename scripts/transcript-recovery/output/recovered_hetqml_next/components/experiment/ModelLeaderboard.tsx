'use client';

import { useEvaluations, latestEvaluation } from '@/lib/use-evaluations';
import { useLiveAlgorithms } from '@/lib/use-live-algorithms';
import type { Algorithm } from '@/data/types';

export function ModelLeaderboard({ activeModelId }: { activeModelId: string | null }) {
  const { algorithms, liveIds, loading: algosLoading } = useLiveAlgorithms();
  const { state: evals } = useEvaluations({ modelManifestId: activeModelId, limit: 50 });

  const live = algorithms.filter((a) => a.status === 'live');
  const dev = algorithms.filter((a) => a.status !== 'live');

  // Latest evaluation drives the metric column for `live` rows. The same
  // evaluation can carry both classical_* and quantum_* metric prefixes
  // (legacy CSV migration), so we show whichever maps to the algo group.
  const latest = evals.phase === 'ready' ? latestEvaluation(evals.records) : null;

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · MODEL LEADERBOARD</div>
          <div className="panel-title">What ran, what didn&apos;t</div>
        </div>
        <span className="badge">Reference</span>
      </div>
      <p className="panel-purpose">
        Algorithms with <strong>live</strong> status are wired to the active
        manifest chain and produced real predictions. Everything below the
        divider is in the catalog but no manifest references it — those rows
        are honest about being inactive.
      </p>

      <Section
        title="Live"
        rows={live}
        emptyMessage={
          algosLoading
            ? 'reading /status...'
            : 'no models active — populate /artifacts/runs and run /admin/reload'
        }
        evalRecord={latest}
        liveIds={liveIds}
      />
      <div style={{ height: 18 }} />
      <Section
        title="In catalog · not active"
        rows={dev}
        muted
        emptyMessage=""
        evalRecord={null}
        liveIds={liveIds}
      />

      <div className="panel-footer">
        <span>data/algorithms.ts × /status × /evaluations</span>
        <span><em>{live.length} live · {dev.length} inactive</em></span>
      </div>
    </div>
  );
}

function Section({
  title, rows, muted, emptyMessage, evalRecord, liveIds,
}: {
  title: string;
  rows: Algorithm[];
  muted?: boolean;
  emptyMessage: string;
  evalRecord: import('@/lib/api').EvaluationRecord | null;
  liveIds: Set<string>;
}) {
  return (
    <div>
      <div
        style={{
          fontSize: 10,
          fontFamily: 'monospace',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          color: muted ? 'var(--faint)' : 'var(--green)',
          marginBottom: 6,
        }}
      >
        {title} · {rows.length}
      </div>
      {rows.length === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--muted)', padding: '8px 0' }}>
          {emptyMessage}
        </div>
      ) : (
        <div
          style={{
            border: '1px solid var(--border-soft)',
            borderRadius: 4,
            overflow: 'hidden',
          }}
        >
          {rows.map((a, i) => (
            <Row
              key={a.id}
              algo={a}
              first={i === 0}
              muted={muted}
              metric={pickMetric(a, evalRecord, liveIds)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function Row({
  algo, first, muted, metric,
}: {
  algo: Algorithm;
  first: boolean;
  muted?: boolean;
  metric: { label: string; value: string } | null;
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 90px 70px 80px',
        gap: 12,
        padding: '8px 12px',
        borderTop: first ? 'none' : '1px solid var(--border-soft)',
        alignItems: 'baseline',
        fontSize: 12,
        opacity: muted ? 0.7 : 1,
      }}
    >
      <div>
        <span style={{ fontWeight: 600, color: 'var(--ink)' }}>{algo.name}</span>
        <span style={{ color: 'var(--faint)', fontSize: 10, marginLeft: 6 }}>
          {algo.group}
        </span>
      </div>
      <div style={{ fontFamily: 'monospace', color: 'var(--muted)', fontSize: 10 }}>
        {algo.path}
      </div>
      <div style={{ fontFamily: 'monospace', color: 'var(--faint)', fontSize: 10, textAlign: 'right' }}>
        {algo.runtime}
      </div>
      <div style={{ textAlign: 'right' }}>
        {metric ? (
          <div style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--ink)' }}>
            <span style={{ fontSize: 9, color: 'var(--faint)' }}>{metric.label} </span>
            <strong>{metric.value}</strong>
          </div>
        ) : (
          <span style={{ fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace' }}>
            {muted ? '—' : 'no eval'}
          </span>
        )}
      </div>
    </div>
  );
}

function pickMetric(
  algo: Algorithm,
  evalRecord: import('@/lib/api').EvaluationRecord | null,
  liveIds: Set<string>,
): { label: string; value: string } | null {
  if (!evalRecord) return null;
  if (!liveIds.has(algo.id)) return null;
  const m = evalRecord.metrics;

  // The legacy migration uses classical_* / quantum_* prefixes; new
  // evaluations may use the bare keys. Prefer prefixed when available.
  const prefix =
    algo.path === 'classical' ? 'classical_' :
    algo.path === 'quantum' || algo.path === 'hybrid' ? 'quantum_' :
    '';
  const prAuc = m[prefix + 'pr_auc'] ?? m['pr_auc'];
  if (prAuc !== undefined) return { label: 'PR-AUC', value: prAuc.toFixed(3) };
  const f1 = m[prefix + 'f1'] ?? m['f1'];
  if (f1 !== undefined) return { label: 'F1', value: f1.toFixed(3) };
  const acc = m[prefix + 'accuracy'] ?? m['accuracy'];
  if (acc !== undefined) return { label: 'ACC', value: acc.toFixed(3) };
  return null;
}
