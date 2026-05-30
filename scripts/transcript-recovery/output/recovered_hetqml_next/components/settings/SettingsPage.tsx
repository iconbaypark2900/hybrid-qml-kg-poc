'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { ApiClient, ApiError } from '@/lib/api';

const API_KEY_STORAGE = 'hetqml_api_key';

type ConnectionState =
  | { phase: 'idle' }
  | { phase: 'testing' }
  | { phase: 'ok'; statusOverall: string; modelId: string | null; quantumModeId: string | null }
  | { phase: 'error'; message: string; code?: string };

export function SettingsPage() {
  const [apiKey, setApiKey] = useState('');
  const [savedKey, setSavedKey] = useState('');
  const [conn, setConn] = useState<ConnectionState>({ phase: 'idle' });
  const baseUrl = (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_HETQML_API_URL)
    || 'http://localhost:8000';

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const k = window.localStorage.getItem(API_KEY_STORAGE) ?? '';
    setApiKey(k);
    setSavedKey(k);
  }, []);

  function save() {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(API_KEY_STORAGE, apiKey);
    setSavedKey(apiKey);
    setConn({ phase: 'idle' });
  }

  function clear() {
    if (typeof window === 'undefined') return;
    window.localStorage.removeItem(API_KEY_STORAGE);
    setApiKey('');
    setSavedKey('');
    setConn({ phase: 'idle' });
  }

  async function testConnection() {
    setConn({ phase: 'testing' });
    const client = new ApiClient({ baseUrl, apiKey: savedKey });
    try {
      // /healthz first (no auth needed). If this fails the API isn't reachable.
      await client.getHealthz();
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'unknown';
      setConn({ phase: 'error', message: `cannot reach ${baseUrl}: ${msg}` });
      return;
    }
    if (!savedKey) {
      setConn({ phase: 'error', message: 'no API key saved — paste one above and click Save first' });
      return;
    }
    try {
      const status = await client.getStatus();
      const quantumComp = status.components.find((c) => c.name === 'quantum_model');
      setConn({
        phase: 'ok',
        statusOverall: status.overall,
        modelId: status.active_manifest_chain?.model_id ?? null,
        quantumModeId: quantumComp?.detail ?? null,
      });
    } catch (e) {
      if (e instanceof ApiError) {
        setConn({ phase: 'error', message: e.message, code: e.code });
      } else {
        setConn({ phase: 'error', message: e instanceof Error ? e.message : 'unknown' });
      }
    }
  }

  const dirty = apiKey !== savedKey;

  return (
    <>
      <div className="page-hero">
        <div>
          <div className="step">SYSTEM · SETTINGS</div>
          <h1 className="h1">Configure access to the service</h1>
          <p className="lede">
            The dashboard talks to a hetqml-service backend over HTTPS. Paste your
            tenant API key below; it&apos;s stored in localStorage on this device only.
            The Test button hits <code>/healthz</code> and <code>/status</code> to
            verify the key works.
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-head">
          <div>
            <div className="eyebrow">SECTION · API ACCESS</div>
            <div className="panel-title">Service connection</div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
          <Field
            label="API base URL"
            value={baseUrl}
            readOnly
            help="Set via NEXT_PUBLIC_HETQML_API_URL at build time."
          />
          <ApiKeyField
            value={apiKey}
            onChange={setApiKey}
            saved={savedKey !== ''}
          />

          <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
            <button
              type="button"
              className="btn-primary"
              onClick={save}
              disabled={!dirty}
              style={{ opacity: dirty ? 1 : 0.5, cursor: dirty ? 'pointer' : 'default' }}
            >
              {dirty ? 'Save key' : 'Saved'}
            </button>
            <button
              type="button"
              className="btn"
              onClick={testConnection}
              disabled={conn.phase === 'testing'}
            >
              {conn.phase === 'testing' ? 'Testing…' : 'Test connection'}
            </button>
            {savedKey && (
              <button type="button" className="btn" onClick={clear}>
                Clear key
              </button>
            )}
          </div>

          <ConnectionResult state={conn} />
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-head">
          <div>
            <div className="eyebrow">SECTION · ABOUT</div>
            <div className="panel-title">Generating an API key</div>
          </div>
        </div>
        <div style={{ marginTop: 12, fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>
          <p>
            On the service host, an operator runs:
          </p>
          <pre
            style={{
              background: 'var(--card)',
              border: '1px solid var(--border-soft)',
              padding: '12px 16px',
              fontFamily: 'monospace',
              fontSize: 11,
              borderRadius: 4,
              overflowX: 'auto',
            }}
          >
{`python -m service.tenants generate-key`}
          </pre>
          <p style={{ marginTop: 12 }}>
            The plaintext key is printed once; the operator pastes the sha256
            into <code>secrets/tenants.yaml</code>. Never commit the plaintext.
          </p>
        </div>
      </div>

      <div className="footer-actions">
        <Link href="/initialize" className="btn">⌥ Back to Initialize</Link>
      </div>
    </>
  );
}

function Field({
  label, value, readOnly, help,
}: { label: string; value: string; readOnly?: boolean; help?: string }) {
  return (
    <div>
      <label
        style={{
          display: 'block',
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          color: 'var(--muted)',
          marginBottom: 4,
          fontFamily: 'monospace',
        }}
      >
        {label}
      </label>
      <input
        type="text"
        value={value}
        readOnly={readOnly}
        style={{
          width: '100%',
          padding: '8px 10px',
          background: readOnly ? 'var(--paper-alt)' : 'var(--paper)',
          border: '1px solid var(--border-soft)',
          color: 'var(--ink)',
          fontFamily: 'monospace',
          fontSize: 12,
          borderRadius: 3,
          outline: 'none',
        }}
      />
      {help && (
        <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 4 }}>{help}</div>
      )}
    </div>
  );
}

function ApiKeyField({
  value, onChange, saved,
}: { value: string; onChange: (v: string) => void; saved: boolean }) {
  const [reveal, setReveal] = useState(false);
  return (
    <div>
      <label
        style={{
          display: 'flex',
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          color: 'var(--muted)',
          marginBottom: 4,
          fontFamily: 'monospace',
          gap: 8,
          alignItems: 'baseline',
        }}
      >
        <span>API key (Bearer token)</span>
        {saved && <span style={{ color: 'var(--green)' }}>● stored locally</span>}
      </label>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          type={reveal ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="paste your key from the operator"
          style={{
            flex: 1,
            padding: '8px 10px',
            background: 'var(--paper)',
            border: '1px solid var(--border-soft)',
            color: 'var(--ink)',
            fontFamily: 'monospace',
            fontSize: 12,
            borderRadius: 3,
            outline: 'none',
          }}
        />
        <button
          type="button"
          className="btn"
          onClick={() => setReveal((r) => !r)}
          style={{ flexShrink: 0 }}
        >
          {reveal ? 'Hide' : 'Show'}
        </button>
      </div>
      <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 4 }}>
        Stored only in this browser&apos;s localStorage. Clear above to remove.
      </div>
    </div>
  );
}

function ConnectionResult({ state }: { state: ConnectionState }) {
  if (state.phase === 'idle') return null;
  if (state.phase === 'testing') {
    return <Banner color="amber">Testing /healthz and /status…</Banner>;
  }
  if (state.phase === 'ok') {
    return (
      <Banner color="green">
        <div style={{ fontWeight: 600 }}>connection ok</div>
        <div style={{ fontSize: 11, marginTop: 4, fontFamily: 'monospace' }}>
          /status overall: <strong>{state.statusOverall}</strong>
          {state.modelId && (
            <>
              {' · '}active model: <strong>{state.modelId}</strong>
            </>
          )}
        </div>
      </Banner>
    );
  }
  return (
    <Banner color="sienna">
      <div style={{ fontWeight: 600 }}>connection failed</div>
      <div style={{ fontSize: 11, marginTop: 4, fontFamily: 'monospace' }}>
        {state.code && <span>{state.code}: </span>}
        {state.message}
      </div>
    </Banner>
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
        marginTop: 4,
        padding: '10px 12px',
        background: bg,
        borderLeft: `3px solid ${accent}`,
        borderRadius: 3,
        color: 'var(--ink)',
        fontSize: 12,
      }}
    >
      {children}
    </div>
  );
}
