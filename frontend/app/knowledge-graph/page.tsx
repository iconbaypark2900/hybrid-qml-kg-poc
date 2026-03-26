"use client";

import { useEffect, useState } from "react";
import type { KGStatsResponse } from "@/lib/api";
import { fetchKGStats, getApiBaseUrl } from "@/lib/api";

export default function KnowledgeGraphPage() {
  const [data, setData] = useState<KGStatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await fetchKGStats();
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
          Knowledge graph exploration
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Hetionet graph statistics and sample edges from the loaded embedder.
        </p>
      </header>

      {loading ? (
        <p className="text-sm text-on-surface-variant" role="status">
          Sequencing&hellip;
        </p>
      ) : error ? (
        <div className="rounded-lg border border-error/40 bg-error-container/20 p-4">
          <p className="text-sm font-medium text-error">
            Could not load KG stats
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">{error}</p>
          <p className="mt-3 text-xs text-on-surface-variant">
            Base URL:{" "}
            <code className="text-on-surface">{getApiBaseUrl()}</code>
          </p>
        </div>
      ) : data?.status === "unavailable" ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          Orchestrator or embedder not loaded. Start the API with trained
          models.
        </div>
      ) : data ? (
        <div className="space-y-6">
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Entities" value={data.entity_count.toLocaleString()} />
            <Stat label="Edges" value={data.edge_count.toLocaleString()} />
            <Stat
              label="Relation types"
              value={String(data.relation_types.length)}
            />
            <Stat
              label="Embedding dim"
              value={
                data.embedding_dim != null ? String(data.embedding_dim) : "—"
              }
            />
          </dl>

          {data.relation_types.length > 0 ? (
            <div>
              <h2 className="mb-2 font-headline text-lg font-semibold text-on-surface">
                Relation types
              </h2>
              <div className="flex flex-wrap gap-2">
                {data.relation_types.map((r) => (
                  <span
                    key={r}
                    className="rounded bg-surface-container-highest px-2 py-1 text-xs font-mono text-on-surface"
                  >
                    {r}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {data.sample_edges.length > 0 ? (
            <div>
              <h2 className="mb-2 font-headline text-lg font-semibold text-on-surface">
                Sample edges
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-surface-container-high text-left text-xs uppercase tracking-wide text-on-surface-variant">
                      <th className="px-3 py-2">Source</th>
                      <th className="px-3 py-2">Relation</th>
                      <th className="px-3 py-2">Target</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.sample_edges.map((e, i) => (
                      <tr
                        key={i}
                        className="border-b border-outline-variant/10 hover:bg-surface-container-lowest/50"
                      >
                        <td className="px-3 py-2 font-mono text-on-surface">
                          {e.source}
                        </td>
                        <td className="px-3 py-2 text-secondary">
                          {e.metaedge}
                        </td>
                        <td className="px-3 py-2 font-mono text-on-surface">
                          {e.target}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          {data.sample_entities.length > 0 ? (
            <div>
              <h2 className="mb-2 font-headline text-lg font-semibold text-on-surface">
                Sample entities (first 20)
              </h2>
              <div className="flex flex-wrap gap-2">
                {data.sample_entities.map((e) => (
                  <span
                    key={e}
                    className="rounded bg-surface-container-lowest px-2 py-1 text-xs font-mono text-tertiary"
                  >
                    {e}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-surface-container-lowest/80 px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </dt>
      <dd className="mt-1 font-mono text-sm text-tertiary">{value}</dd>
    </div>
  );
}
