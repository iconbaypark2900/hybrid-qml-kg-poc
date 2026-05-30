'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useManifest } from '@/lib/use-manifest';
import { ApiClient, ApiError, createApiClientFromEnv, type ComponentHealth } from '@/lib/api';

interface UsageData {
  loading: boolean;
  error: string | null;
  data: {
    quantum_seconds_used: number;
    quantum_jobs_run: number;
    estimated_cost_usd: number;
    period_start: number;
  } | null;
}

export function OperationsPage() {
  const { state, refresh } = useManifest({ pollMs: 15_000 });
  const [usage, setUsage] = useState<UsageData>({ loading: true, error: null, data: null });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const key = window.localStorage.getItem('hetqml_api_key');
    if (!key) {
      setUsage({ loading: false, error: 'API key not set', data: null });
      return;
    }
    const client: ApiClient = createApiClientFromEnv();
    let cancelled = false;
    async function load() {
      try {
        const u = await (client as any).request({ method: 'GET', path: '/usage' });
        if (!cancelled) setUsage({ loading: false, error: null, data: u });
      } catch (e) {
        if (cancelled) return;
        setUsage({
          loading: false,
          error: e instanceof ApiError ? `${e.code}: ${e.message}` : (e instanceof Error ? e.message : 'unknown'),
          data: null,
        });
      }
    }
    void load();
    const id = window.setInterval(load, 30_000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, []);

  return (
    <>
      <div className="page-hero">
        <div>
          <div className="step">SYSTEM · OPERATIONS</div>
          <h1 className="h1">Live status, manifest provenance, and usage</h1>
          <p className="lede">
            Every value below is read from <code>/status</code>, <code>/manifest/active</code>, or
            <code> /usage</code> on a 15-30s poll. Mirror this page on a public
            <code> status.&lt;your-domain&gt;</code> for partner-facing transparency.
          </p>
        </div>
        <button className="btn" onClick={refresh}>force refresh</button>
      </div>

      <OverallBanner state={state} />

      <div className="grid-7-5">
        <ComponentsPanel state={state} />
        <ManifestPanel state={state} />
      </div>

      <div className="grid-7-5">
        <UsagePanel usage={usage} />
        <QuantumPanel state={state} />
      </div>

      <div className="footer-actions">
        <Link href="/initialize" className="btn">⌥ Back to Initialize</Link>
        <Link href="/settings" className="btn">⌥ Settings</Link>
      </div>
    </>
  );
}

function OverallBanner({ state }: { state: ReturnType<typeof useManifest>['state'] }) {
  if (state.phase === 'unconfigured') {
    return (
      <Banner color="amber">
        <strong>API not configured</strong> · {state.reason}.{' '}
        <Link href="/settings">Open Settings</Link> to add an API key.
      </Banner>
    );
  }
  if (state.phase === 'loading') return <Banner color="amber">loading /status…</Banner>;
  if (state.phase === 'error') {
    return <Banner color="sienna"><strong>error</strong> · {state.error.code}: {state.error.message}</Banner>;
  }
  const overall = state.status.overall;
  const color = overall === 'ok' ? 'green' : overall === 'unavailable' ? 'sienna' : 'amber';
  return (
    <Banner color={color}>
      <strong>service overall: {overall.toUpperCase()}</strong> ·
      version {state.status.service_version} · git {state.status.git_sha} ·
      config {state.status.config_hash}
    </Banner>
  );
}

function ComponentsPanel({ state }: { state: ReturnType<typeof useManifest>['state'] }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · COMPONENT HEALTH</div>
          <div className="panel-title">Live probes</div>
        </div>
        <span className="badge">/status</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0, border: '1px solid var(--border-soft)', borderRadius: 4, overflow: 'hidden' }}>
        {state.phase === 'ready'
          ? state.status.components.map((c, i) => <ComponentRow key={c.name} comp={c} first={i === 0} />)
          : <div style={{ padding: 16, fontSize: 12, color: 'var(--muted)' }}>no data</div>
        }
      </div>
    </div>
  );
}

function ComponentRow({ comp, first }: { comp: ComponentHealth; first: boolean }) {
  const color =
    comp.state === 'ok' ? 'var(--green)' :
    comp.state === 'unavailable' ? 'var(--sienna)' :
    'var(--amber)';
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '24px 1fr 100px',
        gap: 10,
        padding: '10px 14px',
        borderTop: first ? 'none' : '1px solid var(--border-soft)',
        alignItems: 'center',
      }}
    >
      <span
        style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block' }}
      />
      <div>
        <div style={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 600 }}>{comp.name}</div>
        {comp.detail && (
          <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace' }}>{comp.detail}</div>
        )}
      </div>
      <div
        style={{
          textAlign: 'right',
          fontFamily: 'monospace',
          fontSize: 11,
          color,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}
      >
        {comp.state}
      </div>
    </div>
  );
}

function ManifestPanel({ state }: { state: ReturnType<typeof useManifest>['state'] }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · ACTIVE MANIFEST</div>
          <div className="panel-title">What&apos;s serving</div>
        </div>
        <span className="badge">/manifest</span>
      </div>
      {state.phase !== 'ready' || !state.classicalChain ? (
        <div style={{ fontSize: 12, color: 'var(--muted)', padding: '12px 0' }}>
          no active manifest chain
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <ChainBlock label="Classical chain" chain={state.classicalChain} model={state.classicalModel} />
          {state.status.active_quantum_manifest_chain && (
            <ChainBlock
              label="Quantum chain"
              chain={state.status.active_quantum_manifest_chain}
              model={null}
            />
          )}
        </div>
      )}
    </div>
  );
}

function ChainBlock({
  label, chain, model,
}: {
  label: string;
  chain: NonNullable<ReturnType<typeof useManifest>['state'] extends infer S ? S extends { classicalChain: any } ? S['classicalChain'] : never : never>;
  model: ReturnType<typeof useManifest>['state'] extends infer S ? S extends { classicalModel: any } ? S['classicalModel'] : null : null;
}) {
  return (
    <div style={{ border: '1px solid var(--border-soft)', borderRadius: 4, padding: 12 }}>
      <div style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--faint)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6 }}>
        {label}
      </div>
      <KeyVal k="model_id" v={chain.model_id} />
      <KeyVal k="feature_pipeline_id" v={chain.feature_pipeline_id} />
      <KeyVal k="embedding_id" v={chain.embedding_id} />
      {model && <KeyVal k="model_type" v={model.model_type} />}
    </div>
  );
}

function KeyVal({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '160px 1fr', gap: 8, fontSize: 11, lineHeight: 1.6 }}>
      <span style={{ fontFamily: 'monospace', color: 'var(--faint)' }}>{k}</span>
      <span style={{ fontFamily: 'monospace', color: 'var(--ink)', wordBreak: 'break-all' }}>{v}</span>
    </div>
  );
}

function UsagePanel({ usage }: { usage: UsageData }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · TENANT USAGE</div>
          <div className="panel-title">This billing period</div>
        </div>
        <span className="badge">/usage</span>
      </div>
      {usage.loading ? (
        <div style={{ fontSize: 12, color: 'var(--muted)' }}>loading…</div>
      ) : usage.error ? (
        <div style={{ fontSize: 11, color: 'var(--sienna)', fontFamily: 'monospace' }}>{usage.error}</div>
      ) : usage.data ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          <UsageStat label="QUANTUM SECONDS" value={usage.data.quantum_seconds_used.toFixed(1)} />
          <UsageStat label="JOBS RUN" value={String(usage.data.quantum_jobs_run)} />
          <UsageStat label="EST COST USD" value={`$${usage.data.estimated_cost_usd.toFixed(4)}`} />
        </div>
      ) : null}
      {usage.data && (
        <div style={{ marginTop: 8, fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace' }}>
          period started: {new Date(usage.data.period_start * 1000).toISOString().slice(0, 10)}
        </div>
      )}
    </div>
  );
}

function UsageStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: 9, fontFamily: 'monospace', color: 'var(--faint)', letterSpacing: '0.5px' }}>{label}</div>
      <div style={{ fontFamily: 'Georgia, serif', fontSize: 22, fontWeight: 600, color: 'var(--ink)', lineHeight: 1.1 }}>{value}</div>
    </div>
  );
}

function QuantumPanel({ state }: { state: ReturnType<typeof useManifest>['state'] }) {
  if (state.phase !== 'ready') return null;
  const q = state.status.quantum;
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · QUANTUM STATUS</div>
          <div className="panel-title">{q.default_mode}</div>
        </div>
        <span className="badge">/status.quantum</span>
      </div>
      <div style={{ fontSize: 11, lineHeight: 1.7 }}>
        <KeyVal k="default_mode" v={q.default_mode} />
        <KeyVal k="available_modes" v={q.available_modes.join(', ')} />
        <KeyVal k="ibm_runtime_reachable" v={q.ibm_runtime_reachable ? 'yes' : 'no'} />
        {q.note && (
          <div style={{ marginTop: 8, padding: 8, background: 'var(--card)', borderLeft: '2px solid var(--amber)', fontSize: 10, color: 'var(--muted)' }}>
            {q.note}
          </div>
        )}
      </div>
    </div>
  );
}

function Banner({
  color, children,
}: { color: 'green' | 'amber' | 'sienna'; children: React.ReactNode }) {
  const accent = color === 'green' ? 'var(--green)' : color === 'amber' ? 'var(--amber)' : 'var(--sienna)';
  const bg =
    color === 'green' ? 'rgba(120, 200, 130, 0.06)' :
    color === 'amber' ? 'rgba(220, 180, 80, 0.06)' :
    'rgba(224, 132, 116, 0.06)';
  return (
    <div
      style={{
        marginBottom: 16,
        padding: '12px 16px',
        background: bg,
        borderLeft: `3px solid ${accent}`,
        borderRadius: 3,
        fontSize: 13,
        color: 'var(--ink)',
      }}
    >
      {children}
    </div>
  );
}
