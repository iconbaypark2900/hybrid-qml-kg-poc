'use client';

// Renders the active model fingerprint in the sidebar status bar.
// Replaces the previously-hardcoded "ibm_torino · queue: 2" placeholder
// with a live read of /status + /manifest/model/<id>.
//
// Honesty rules:
//   - synthetic placeholder models get a vivid warning treatment
//   - models trained on synthetic data (vqc_mini_trained_synthetic_data) get a
//     softer "synthetic data" caveat
//   - the quantum component health dictates whether quantum is shown as
//     active, degraded, or unavailable (no claim of "live quantum" without
//     evidence)

import { isSyntheticModel, useManifest } from '@/lib/use-manifest';
import type { ComponentHealth } from '@/lib/api';

const SYNTHETIC_LABEL: Record<string, string> = {
  synthetic_placeholder: 'SYNTHETIC PLACEHOLDER',
  'legacy-unknown': 'LEGACY (pre-rebuild)',
  vqc_mini_trained_synthetic_data: 'SYNTHETIC TRAINING DATA',
};

export function ActiveModelBadge() {
  const { state, refresh } = useManifest();

  if (state.phase === 'unconfigured') {
    return (
      <div className="status-bar" style={{ borderTopColor: 'var(--border-soft)' }}>
        <span className="status-dot" style={{ background: 'var(--faint)' }} />
        <span className="mono" style={{ color: 'var(--faint)' }}>API not configured</span>
        <div style={{ color: 'var(--faint)', fontSize: 10, marginTop: 4 }}>
          {state.reason}
        </div>
      </div>
    );
  }

  if (state.phase === 'loading') {
    return (
      <div className="status-bar">
        <span className="status-dot" style={{ background: 'var(--amber)' }} />
        <span className="mono">connecting...</span>
      </div>
    );
  }

  if (state.phase === 'error') {
    return (
      <div className="status-bar" style={{ borderTopColor: 'var(--sienna)' }}>
        <span className="status-dot" style={{ background: 'var(--sienna)' }} />
        <span className="mono">api error</span>
        <div style={{ color: 'var(--sienna)', fontSize: 10, marginTop: 4 }}>
          {state.error.code}: {state.error.message.slice(0, 64)}
          {state.canRetry ? (
            <button
              onClick={refresh}
              style={{
                marginLeft: 6,
                background: 'transparent',
                border: '1px solid var(--sienna)',
                color: 'var(--sienna)',
                cursor: 'pointer',
                padding: '0 4px',
                fontSize: 10,
                fontFamily: 'monospace',
              }}
            >
              retry
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  const { status, classicalModel, classicalChain } = state;
  const overallColor = pickColor(status.overall);
  const quantumComp = status.components.find((c) => c.name === 'quantum_model');
  const synthetic = isSyntheticModel(classicalModel);
  const syntheticLabel = classicalModel
    ? SYNTHETIC_LABEL[classicalModel.model_type] ?? null
    : null;

  return (
    <div
      className="status-bar"
      style={{
        borderTopColor: synthetic ? 'var(--sienna)' : 'var(--border-soft)',
        background: synthetic ? 'var(--sienna-bg, rgba(224,132,116,0.06))' : undefined,
      }}
    >
      <span className="status-dot" style={{ background: overallColor }} />
      <span className="mono" title={`config_hash=${status.config_hash}`}>
        {classicalChain?.model_id ?? 'no manifest'}
      </span>
      <div style={{ color: 'var(--faint)', fontSize: 10, marginTop: 4 }}>
        {classicalModel ? (
          <>
            <span>{classicalModel.model_type}</span>
            <span style={{ marginLeft: 6, color: 'var(--faint)' }}>
              · sha {status.git_sha}
            </span>
          </>
        ) : (
          <span>no model manifest</span>
        )}
      </div>

      {synthetic && syntheticLabel ? (
        <div
          style={{
            marginTop: 6,
            padding: '4px 6px',
            background: 'var(--sienna)',
            color: 'var(--paper, white)',
            fontFamily: 'monospace',
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: '0.5px',
            borderRadius: 3,
            textAlign: 'center',
          }}
          title="This model is not trained on real Hetionet data. Predictions are placeholders."
        >
          {syntheticLabel}
        </div>
      ) : null}

      <QuantumLine comp={quantumComp} />
    </div>
  );
}

function QuantumLine({ comp }: { comp: ComponentHealth | undefined }) {
  if (!comp) return null;
  const color = pickColor(comp.state);
  return (
    <div
      style={{
        marginTop: 4,
        fontSize: 10,
        color,
        fontFamily: 'monospace',
        display: 'flex',
        gap: 6,
        alignItems: 'baseline',
      }}
      title={comp.detail ?? undefined}
    >
      <span>quantum:</span>
      <span style={{ fontWeight: 600 }}>{comp.state}</span>
    </div>
  );
}

function pickColor(state: string): string {
  switch (state) {
    case 'ok':
      return 'var(--green)';
    case 'loading':
      return 'var(--amber)';
    case 'degraded':
      return 'var(--amber)';
    case 'unavailable':
      return 'var(--sienna)';
    default:
      return 'var(--faint)';
  }
}
