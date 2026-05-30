'use client';

// 4-step onboarding checklist that auto-advances based on real backend
// state. Lives on /initialize so first-time users see it immediately.
//
//   1. API key configured (localStorage)
//   2. /healthz reachable
//   3. /status reports overall ok or degraded (not unavailable)
//   4. First /predict succeeds

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { ApiClient, ApiError, createApiClientFromEnv } from '@/lib/api';

type StepKey = 'key' | 'healthz' | 'status' | 'predict';
type StepState = 'pending' | 'ok' | 'fail';

interface Step {
  key: StepKey;
  title: string;
  detail: string;
  state: StepState;
  errorDetail?: string;
}

const STORAGE_DISMISSED = 'hetqml.onboarding_dismissed';

export function OnboardingChecklist() {
  const [steps, setSteps] = useState<Step[]>(() => initialSteps());
  const [dismissed, setDismissed] = useState(false);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (window.localStorage.getItem(STORAGE_DISMISSED) === '1') {
      setDismissed(true);
      return;
    }
    void runChecks(setSteps, setRunning);
  }, []);

  if (dismissed) return null;

  const allOk = steps.every((s) => s.state === 'ok');

  return (
    <div
      className="panel"
      style={{
        marginBottom: 16,
        background: allOk ? 'rgba(120, 200, 130, 0.04)' : 'var(--card)',
        borderLeft: `3px solid ${allOk ? 'var(--green)' : 'var(--amber)'}`,
      }}
    >
      <div className="panel-head">
        <div>
          <div className="eyebrow">SETUP · {allOk ? 'READY' : 'ONBOARDING'}</div>
          <div className="panel-title">
            {allOk ? 'Service connected. You can run an investigation.' : 'Connect to the service'}
          </div>
        </div>
        <button
          className="btn"
          onClick={() => {
            window.localStorage.setItem(STORAGE_DISMISSED, '1');
            setDismissed(true);
          }}
          style={{ fontSize: 11 }}
        >
          {allOk ? 'dismiss' : 'skip for now'}
        </button>
      </div>
      <p className="panel-purpose">
        Each step below is wired to the live backend — no mocks. If a check
        fails, click the link beside it to fix and the panel re-runs automatically.
      </p>
      <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {steps.map((s, i) => (
          <StepRow key={s.key} step={s} index={i + 1} />
        ))}
      </ol>
      <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
        <button
          className="btn"
          onClick={() => void runChecks(setSteps, setRunning)}
          disabled={running}
        >
          {running ? 'checking…' : 'recheck'}
        </button>
        {allOk && (
          <span style={{ fontSize: 11, color: 'var(--green)', alignSelf: 'center' }}>
            ● all checks pass — start by picking a compound and disease below
          </span>
        )}
      </div>
    </div>
  );
}

function StepRow({ step, index }: { step: Step; index: number }) {
  const dot =
    step.state === 'ok' ? 'var(--green)' :
    step.state === 'fail' ? 'var(--sienna)' :
    'var(--faint)';
  const label =
    step.state === 'ok' ? '✓' :
    step.state === 'fail' ? '!' :
    String(index);
  return (
    <li
      style={{
        display: 'grid',
        gridTemplateColumns: '24px 1fr auto',
        gap: 10,
        alignItems: 'baseline',
      }}
    >
      <span
        style={{
          width: 22, height: 22,
          borderRadius: '50%',
          background: dot,
          color: 'var(--paper)',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          fontWeight: 700,
          fontFamily: 'monospace',
        }}
      >
        {label}
      </span>
      <div>
        <div style={{ fontWeight: 600, fontSize: 13 }}>{step.title}</div>
        <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.5 }}>
          {step.detail}
          {step.errorDetail ? (
            <span style={{ display: 'block', color: 'var(--sienna)', fontFamily: 'monospace', fontSize: 10, marginTop: 2 }}>
              {step.errorDetail}
            </span>
          ) : null}
        </div>
      </div>
      <StepAction step={step} />
    </li>
  );
}

function StepAction({ step }: { step: Step }) {
  if (step.state === 'ok') return null;
  if (step.key === 'key') {
    return <Link href="/settings" className="btn" style={{ fontSize: 11 }}>open Settings</Link>;
  }
  if (step.key === 'healthz' || step.key === 'status') {
    return <Link href="/operations" className="btn" style={{ fontSize: 11 }}>view ops</Link>;
  }
  if (step.key === 'predict') {
    return <span style={{ fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace' }}>auto-runs</span>;
  }
  return null;
}

function initialSteps(): Step[] {
  return [
    { key: 'key', title: 'API key configured', detail: 'Bearer token saved in this browser.', state: 'pending' },
    { key: 'healthz', title: 'Service reachable', detail: 'GET /healthz returns 200.', state: 'pending' },
    { key: 'status', title: 'Service ready', detail: '/status reports overall ok or degraded (not unavailable).', state: 'pending' },
    { key: 'predict', title: 'First prediction', detail: 'A test /predict against a known drug-disease pair returns a probability.', state: 'pending' },
  ];
}

async function runChecks(
  setSteps: (s: Step[]) => void,
  setRunning: (b: boolean) => void,
): Promise<void> {
  if (typeof window === 'undefined') return;
  setRunning(true);
  const steps = initialSteps();

  // Step 1: API key
  const key = window.localStorage.getItem('hetqml_api_key') ?? '';
  steps[0].state = key ? 'ok' : 'fail';
  if (!key) {
    steps[0].errorDetail = 'Open Settings and paste your tenant API key.';
    setSteps([...steps]);
    setRunning(false);
    return;
  }
  setSteps([...steps]);

  const client: ApiClient = createApiClientFromEnv();

  // Step 2: /healthz
  try {
    await client.getHealthz();
    steps[1].state = 'ok';
  } catch (e) {
    steps[1].state = 'fail';
    steps[1].errorDetail = e instanceof Error ? e.message : 'unknown error';
    setSteps([...steps]);
    setRunning(false);
    return;
  }
  setSteps([...steps]);

  // Step 3: /status
  let ok = false;
  let primaryModelId: string | null = null;
  let drugId: string | null = null;
  let diseaseId: string | null = null;
  try {
    const status = await client.getStatus();
    ok = status.overall !== 'unavailable';
    primaryModelId = status.active_manifest_chain?.model_id ?? null;
    if (status.overall === 'unavailable') {
      steps[2].state = 'fail';
      steps[2].errorDetail = 'service overall is unavailable; see /operations';
    } else {
      steps[2].state = 'ok';
      // Best-effort: pick a known sample drug-disease pair from the curated list.
      drugId = 'DB00178';
      diseaseId = 'DOID:10534';
    }
  } catch (e) {
    steps[2].state = 'fail';
    if (e instanceof ApiError) {
      steps[2].errorDetail = `${e.code}: ${e.message}`;
    } else {
      steps[2].errorDetail = e instanceof Error ? e.message : 'unknown error';
    }
  }
  setSteps([...steps]);
  if (!ok || !drugId || !diseaseId) {
    setRunning(false);
    return;
  }

  // Step 4: /predict
  try {
    const resp = await client.predict({ drug_id: drugId, disease_id: diseaseId, method: 'classical' });
    steps[3].state = 'ok';
    steps[3].detail = `Predicted ${resp.probability.toFixed(3)} via ${resp.method_used} on ${primaryModelId ?? 'active model'}.`;
  } catch (e) {
    steps[3].state = 'fail';
    steps[3].errorDetail = e instanceof ApiError ? `${e.code}: ${e.message}` : (e instanceof Error ? e.message : 'unknown');
  }
  setSteps([...steps]);
  setRunning(false);
}
