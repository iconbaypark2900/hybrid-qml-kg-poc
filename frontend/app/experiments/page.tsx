"use client";

import { useEffect, useState } from "react";
import type { LatestRunResponse } from "@/lib/api";
import { fetchLatestRun, getApiBaseUrl } from "@/lib/api";

export default function ExperimentsPage() {
  const [data, setData] = useState<LatestRunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const run = await fetchLatestRun();
        if (!cancelled) {
          setData(run);
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
          Experiments
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Latest pipeline run and model rankings.
        </p>
      </header>

      {loading ? (
        <p className="text-sm text-on-surface-variant" role="status">
          Sequencing&hellip;
        </p>
      ) : error ? (
        <div className="rounded-lg border border-error/40 bg-error-container/20 p-4">
          <p className="text-sm font-medium text-error">
            Could not load latest run
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">{error}</p>
          <p className="mt-3 text-xs text-on-surface-variant">
            Base URL:{" "}
            <code className="text-on-surface">{getApiBaseUrl()}</code>
          </p>
        </div>
      ) : data?.status === "no_results" ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5">
          <p className="text-sm text-on-surface-variant">
            No pipeline results yet. Run the pipeline from the repo root:
          </p>
          <pre className="mt-2 rounded bg-surface-container-lowest px-3 py-2 text-xs text-on-surface">
            python scripts/run_optimized_pipeline.py --relation CtD
          </pre>
        </div>
      ) : (
        <RunSummary data={data!} />
      )}
    </div>
  );
}

function RunSummary({ data }: { data: LatestRunResponse }) {
  const ranking = extractRanking(data);

  return (
    <div className="space-y-6">
      {data.message ? (
        <p className="text-xs text-on-surface-variant">{data.message}</p>
      ) : null}

      <dl className="grid gap-4 sm:grid-cols-3">
        <MetricCard label="Status" value={data.status} />
        <MetricCard label="Results dir" value={data.results_dir} />
        <MetricCard
          label="Best model"
          value={ranking.length > 0 ? ranking[0].name : "—"}
        />
      </dl>

      {ranking.length > 0 ? (
        <div>
          <h2 className="mb-3 font-headline text-lg font-semibold text-on-surface">
            Ranking by test PR-AUC
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
                  <th className="px-3 py-2 text-right">Fit time (s)</th>
                </tr>
              </thead>
              <tbody>
                {ranking.map((row, i) => (
                  <tr
                    key={row.name}
                    className="border-b border-outline-variant/10 hover:bg-surface-container-lowest/50"
                  >
                    <td className="px-3 py-2 font-mono text-on-surface-variant">
                      {i + 1}
                    </td>
                    <td className="px-3 py-2 font-medium text-on-surface">
                      {row.name}
                    </td>
                    <td className="px-3 py-2 text-on-surface-variant">
                      {row.type}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-tertiary">
                      {fmt(row.pr_auc)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-on-surface">
                      {fmt(row.accuracy)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-on-surface-variant">
                      {fmt(row.fit_time)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {data.latest_csv ? (
        <LatestCsvSummary csv={data.latest_csv} />
      ) : null}
    </div>
  );
}

function LatestCsvSummary({ csv }: { csv: Record<string, unknown> }) {
  const entries = Object.entries(csv).filter(
    ([, v]) => v !== null && v !== undefined && v !== "",
  );
  if (entries.length === 0) return null;

  return (
    <div>
      <h2 className="mb-3 font-headline text-lg font-semibold text-on-surface">
        Latest run snapshot
      </h2>
      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {entries.map(([k, v]) => (
          <div
            key={k}
            className="rounded-lg bg-surface-container-lowest/80 px-4 py-3"
          >
            <dt className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
              {k}
            </dt>
            <dd className="mt-1 truncate font-mono text-sm text-on-surface">
              {String(v)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-surface-container-lowest/80 px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </dt>
      <dd className="mt-1 font-mono text-sm text-tertiary">{value}</dd>
    </div>
  );
}

interface RankRow {
  name: string;
  type: string;
  pr_auc: number;
  accuracy: number;
  fit_time: number;
}

function extractRanking(data: LatestRunResponse): RankRow[] {
  const json = data.latest_json;
  if (!json) return [];
  const ranking = json.ranking as RankRow[] | undefined;
  if (Array.isArray(ranking)) {
    return [...ranking].sort(
      (a, b) => (b.pr_auc ?? 0) - (a.pr_auc ?? 0),
    );
  }
  return [];
}

function fmt(v: number | undefined | null): string {
  if (v == null) return "—";
  return v.toFixed(4);
}
