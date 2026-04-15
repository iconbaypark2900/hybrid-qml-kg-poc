"use client";

import Link from "next/link";
import { PredictForm } from "@/components/predict-form";

export default function HomePage() {
  return (
    <div className="space-y-12">

      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <header className="space-y-3 border-b border-outline/10 pb-8">
        <p className="font-label text-xs font-semibold uppercase tracking-widest text-primary">
          Hybrid Quantum-Classical · Biomedical Link Prediction
        </p>
        <h1 className="font-headline text-3xl font-semibold tracking-tight text-on-surface">
          Does this compound treat this disease?
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-on-surface-variant">
          This system uses the{" "}
          <strong className="text-on-surface">Hetionet</strong> biomedical
          knowledge graph — 47,031 entities, 2.25 million relationships — to
          score drug-disease pairs. Embeddings are learned from graph structure,
          then passed through classical and quantum models to produce a
          treatment-probability score. It is a research tool, not clinical
          guidance.
        </p>

        {/* Quick-nav pills */}
        <div className="flex flex-wrap gap-3 pt-2">
          <Link
            href="/experiments"
            className="rounded-full border border-outline/20 bg-surface-container-lowest px-4 py-1.5 text-xs font-medium text-on-surface transition-colors hover:bg-surface-container"
          >
            View experiment results →
          </Link>
          <Link
            href="/simulation/parameters"
            className="rounded-full border border-outline/20 bg-surface-container-lowest px-4 py-1.5 text-xs font-medium text-on-surface transition-colors hover:bg-surface-container"
          >
            Run a new pipeline job →
          </Link>
          <Link
            href="/knowledge-graph"
            className="rounded-full border border-outline/20 bg-surface-container-lowest px-4 py-1.5 text-xs font-medium text-on-surface transition-colors hover:bg-surface-container"
          >
            Explore the knowledge graph →
          </Link>
        </div>
      </header>

      {/* ── Prediction form ──────────────────────────────────────────── */}
      <section className="grid gap-8 lg:grid-cols-[1fr_320px]">
        <div className="space-y-4">
          <div>
            <h2 className="font-headline text-xl font-semibold text-on-surface">
              Try a prediction
            </h2>
            <p className="mt-1 text-sm text-on-surface-variant">
              Enter a drug and a disease. The model returns a probability that
              the compound treats the condition, based on graph-embedding
              similarity. Hetionet uses systematic drug names — try{" "}
              <code className="rounded bg-surface-container px-1 py-0.5 text-xs text-primary">
                pindolol
              </code>{" "}
              or{" "}
              <code className="rounded bg-surface-container px-1 py-0.5 text-xs text-primary">
                acetylsalicylic acid
              </code>
              , not common brand names.
            </p>
          </div>
          <PredictForm />
        </div>

        {/* Sidebar: how to read the score */}
        <aside className="space-y-4 rounded-xl border border-outline/10 bg-surface-container-lowest/60 p-5 text-sm">
          <h3 className="font-semibold text-on-surface">
            How to read the score
          </h3>
          <ul className="space-y-3 text-on-surface-variant">
            <li className="flex gap-2">
              <span className="mt-0.5 text-tertiary">→</span>
              <span>
                <strong className="text-on-surface">≥ 70 %</strong> — strong
                graph-structural evidence. Worth reviewing in literature.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="mt-0.5 text-secondary">→</span>
              <span>
                <strong className="text-on-surface">40 – 70 %</strong> —
                moderate signal. Pair is structurally plausible but not
                well-supported by known edges.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="mt-0.5 text-on-surface-variant">→</span>
              <span>
                <strong className="text-on-surface">&lt; 40 %</strong> — weak
                or no known relationship in Hetionet.
              </span>
            </li>
          </ul>
          <p className="border-t border-outline/10 pt-3 text-xs text-on-surface-variant">
            The served model is a Logistic Regression trained on 4 × 12-dim
            RotatE embeddings (48 features). The benchmark winner (ExtraTrees,
            PR-AUC 0.81) uses graph and domain features not available at
            inference time.
          </p>
        </aside>
      </section>

      {/* ── How it works ─────────────────────────────────────────────── */}
      <section className="space-y-4 border-t border-outline/10 pt-8">
        <h2 className="font-headline text-xl font-semibold text-on-surface">
          How it works
        </h2>
        <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              step: "01",
              title: "Graph embedding",
              body: "RotatE trains on all 2.25 M Hetionet edges. Each entity gets a 128-dim vector capturing its position in the relational graph.",
            },
            {
              step: "02",
              title: "Pair features",
              body: "For each candidate pair (drug, disease) we compute concatenation, absolute difference, and element-wise product of their embeddings.",
            },
            {
              step: "03",
              title: "Classical & quantum models",
              body: "Classical: ExtraTrees, RF, LogReg. Quantum: QSVC with Pauli feature map on 16-qubit PCA-reduced features. Best ensemble: PR-AUC 0.7987.",
            },
            {
              step: "04",
              title: "Score & rank",
              body: "The API returns a link-probability score. Use Ranked candidates to surface the top-K compounds for a given disease.",
            },
          ].map(({ step, title, body }) => (
            <li
              key={step}
              className="rounded-xl border border-outline/10 bg-surface-container-lowest/60 p-4"
            >
              <p className="font-mono text-xs font-bold text-primary opacity-60">
                {step}
              </p>
              <p className="mt-1 font-semibold text-on-surface">{title}</p>
              <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">
                {body}
              </p>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
