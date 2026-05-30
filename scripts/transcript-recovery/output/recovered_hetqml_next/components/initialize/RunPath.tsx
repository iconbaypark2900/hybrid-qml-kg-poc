'use client';

import { useInvestigationStore } from '@/lib/store';
import { ALGORITHMS } from '@/data/algorithms';
import type { RunPath as RunPathType } from '@/data/types';
import { HelpHint } from '@/components/shared/HelpHint';

const PATHS: { key: RunPathType; icon: string; title: string; sub: string; stats: string; detail: string; help: string }[] = [
  {
    key: 'classical',
    icon: '▢',
    title: 'Classical',
    sub: 'Stacking · GBDT · KGE',
    stats: '~50s · CPU',
    detail: 'straightforward baseline run',
    help: '<strong>Classical preset</strong>: pure CPU. Runs stacking, Extra Trees, Logistic Regression, XGBoost, LightGBM, SVM, MLP, KG embeddings (RotatE), and DWPC heuristic. ~50s wall-clock for the full sweep.'
  },
  {
    key: 'hybrid',
    icon: '⌥',
    title: 'Hybrid',
    sub: 'QSVC · VQC + baselines',
    stats: '~2m · mixed',
    detail: 'straightforward parameter-efficient run',
    help: '<strong>Hybrid preset</strong>: quantum kernel evaluation on IBM hardware + classical SVM solver. Runs QSVC (Pauli/Z), VQC, projected/fidelity quantum kernels, plus all classical baselines. ~2 minutes total.'
  },
  {
    key: 'quantum',
    icon: '⊗',
    title: 'Quantum HW',
    sub: 'QAOA · VQE + baselines',
    stats: '~4m · 18k shots',
    detail: 'straightforward hardware-validated run',
    help: '<strong>Quantum HW preset</strong>: end-to-end on IBM Quantum hardware (Heron r2 / Eagle r3). Runs QAOA, VQE-classifier, plus all classical baselines. ~4 minutes per job.'
  }
];

export function RunPath() {
  const runPath = useInvestigationStore((s) => s.runPath);
  const setRunPath = useInvestigationStore((s) => s.setRunPath);

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · RUN PATH</div>
          <div className="panel-title">How the investigation runs</div>
        </div>
        <span className="badge">Selector</span>
      </div>
      <p className="panel-purpose">
        <strong>Pick a preset bundle below.</strong> Each card runs the canonical set of algorithms
        for its category. Classical baselines always run alongside quantum / hybrid for comparison.
      </p>

      <div className="run-path-grid">
        {PATHS.map((p) => {
          const liveInPath = ALGORITHMS.filter((a) => a.path === p.key && (a.status === 'live' || a.status === 'fallback')).length;
          const baselines = p.key !== 'classical' ? ALGORITHMS.filter((a) => a.path === 'classical' && (a.status === 'live' || a.status === 'fallback')).length : 0;
          const countLabel = baselines > 0 ? `${liveInPath} + ${baselines} baselines` : `${liveInPath} algorithms`;
          return (
            <div
              key={p.key}
              className={'run-path' + (runPath === p.key ? ' active' : '')}
              onClick={() => setRunPath(p.key)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') setRunPath(p.key);
              }}
            >
              <div className="run-path-icon">{p.icon}</div>
              <div className="run-path-title" data-help>
                {p.title}
                <HelpHint text={p.help} />
              </div>
              <div className="run-path-sub">{p.sub}</div>
              <div className="run-path-stats">
                <span className="run-path-algo-count">{countLabel}</span>
                <span>{p.stats}</span>
              </div>
              <div className="run-path-detail">{p.detail}</div>
            </div>
          );
        })}
      </div>

      <div className="panel-footer">
        <span>compute_router :: {runPath}</span>
        <span><em>cost shown is estimated</em></span>
      </div>
    </div>
  );
}
