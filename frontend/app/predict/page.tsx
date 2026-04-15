"use client";

import { PredictForm } from "@/components/predict-form";

export default function PredictPage() {
  return (
    <div className="space-y-8">

      {/* ── Header ───────────────────────────────────────────────────── */}
      <header className="space-y-2 border-b border-outline/10 pb-6">
        <p className="font-label text-xs font-semibold uppercase tracking-widest text-primary">
          Predict treatment
        </p>
        <h1 className="font-headline text-3xl font-semibold tracking-tight text-on-surface">
          Drug – disease link prediction
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-on-surface-variant">
          Enter a drug and a disease below. The model scores the pair using
          RotatE graph embeddings and returns a treatment-probability estimate.
          Use systematic drug names (e.g.{" "}
          <code className="rounded bg-surface-container px-1 py-0.5 text-xs text-primary">
            pindolol
          </code>
          ) or DrugBank IDs (e.g.{" "}
          <code className="rounded bg-surface-container px-1 py-0.5 text-xs text-primary">
            DB00960
          </code>
          ).
        </p>
      </header>

      {/* ── Form + scoring guide ─────────────────────────────────────── */}
      <div className="grid gap-8 lg:grid-cols-[1fr_320px]">

        {/* Left: form */}
        <section className="space-y-2">
          <PredictForm />
        </section>

        {/* Right: scoring guide */}
        <aside className="space-y-4 self-start rounded-xl border border-outline/10 bg-surface-container-lowest/60 p-5 text-sm">
          <h2 className="font-semibold text-on-surface">Score interpretation</h2>
          <ul className="space-y-3 text-on-surface-variant">
            <li className="flex gap-2">
              <span className="mt-0.5 shrink-0 text-tertiary">→</span>
              <span>
                <strong className="text-on-surface">≥ 70 %</strong> — strong
                graph-structural evidence. Worth reviewing in literature.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="mt-0.5 shrink-0 text-secondary">→</span>
              <span>
                <strong className="text-on-surface">40 – 70 %</strong> —
                moderate signal. Structurally plausible but not well-supported
                by known Hetionet edges.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="mt-0.5 shrink-0 text-on-surface-variant">→</span>
              <span>
                <strong className="text-on-surface">&lt; 40 %</strong> — weak
                or no known relationship in Hetionet.
              </span>
            </li>
          </ul>

          <div className="border-t border-outline/10 pt-3 text-xs text-on-surface-variant space-y-1">
            <p className="font-medium text-on-surface">Model notes</p>
            <p>
              Served model: Logistic Regression trained on 4 × 12-dim RotatE
              embeddings (48 features).
            </p>
            <p>
              Benchmark winner: ExtraTrees, PR-AUC 0.81 — uses graph and domain
              features not available at inference time.
            </p>
            <p className="text-error/80 pt-1">
              Research tool only — not clinical guidance.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
