"use client";

import { useEffect, useState } from "react";
import type { AnalysisSummaryResponse } from "@/lib/api";
import { fetchAnalysisSummary, getApiBaseUrl } from "@/lib/api";

export default function DrugDeliveryAnalysisPage() {
  const [data, setData] = useState<AnalysisSummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await fetchAnalysisSummary();
        if (!cancelled) {
          setData(s);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Request failed");
          setData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          Analysis: drug delivery
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Aggregated model performance and drug-disease coverage from the latest
          pipeline run.
        </p>
      </header>

      {loading ? (
        <p className="text-sm text-on-surface-variant" role="status">
          Sequencing&hellip;
        </p>
      ) : error ? (
        <div className="rounded-lg border border-error/40 bg-error-container/20 p-4">
          <p className="text-sm font-medium text-error">
            Could not load analysis
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">{error}</p>
          <p className="mt-3 text-xs text-on-surface-variant">
            Base URL:{" "}
            <code className="text-on-surface">{getApiBaseUrl()}</code>
          </p>
        </div>
      ) : data?.status === "no_results" ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          No pipeline results yet. Run the pipeline first.
        </div>
      ) : data ? (
        <div className="space-y-6">
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Best model" value={data.best_model ?? "—"} />
            <Stat
              label="Best PR-AUC"
              value={data.best_pr_auc != null ? data.best_pr_auc.toFixed(4) : "—"}
              highlight
            />
            <Stat label="Models evaluated" value={String(data.model_count)} />
            <Stat label="Relation" value={data.relation ?? "—"} />
          </dl>

          <dl className="grid gap-4 sm:grid-cols-3">
            <Stat label="Classical" value={String(data.classical_count)} />
            <Stat label="Quantum" value={String(data.quantum_count)} />
            <Stat label="Ensemble" value={String(data.ensemble_count)} />
          </dl>

          {data.ranking && data.ranking.length > 0 ? (
            <div>
              <h2 className="mb-3 font-headline text-lg font-semibold text-on-surface">
                Model ranking
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-surface-container-high text-left text-xs uppercase tracking-wide text-on-surface-variant">
                      <th className="px-3 py-2">#</th>
                      <th className="px-3 py-2">Model</th>
                      <th className="px-3 py-2">Type</th>
                      <th className="px-3 py-2 text-right">PR-AUC</th>
                      <th className="px-3 py-2 text-right">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.ranking.map((r, i) => (
                      <tr
                        key={String(r.name)}
                        className="border-b border-outline-variant/10 hover:bg-surface-container-lowest/50"
                      >
                        <td className="px-3 py-2 font-mono text-on-surface-variant">
                          {i + 1}
                        </td>
                        <td className="px-3 py-2 font-medium text-on-surface">
                          {String(r.name)}
                        </td>
                        <td className="px-3 py-2 text-on-surface-variant">
                          {String(r.type)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-tertiary">
                          {fmt(r.pr_auc as number)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-on-surface">
                          {fmt(r.accuracy as number)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Stat({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="rounded-lg bg-surface-container-lowest/80 px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </dt>
      <dd
        className={`mt-1 font-mono text-sm ${highlight ? "text-tertiary" : "text-on-surface"}`}
      >
        {value}
      </dd>
    </div>
  );
}

function fmt(v: number | undefined | null): string {
  if (v == null) return "—";
  return v.toFixed(4);
}
