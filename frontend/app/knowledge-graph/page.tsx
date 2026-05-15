"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type {
  KGSearchResult,
  KGStatsResponse,
  VizKGSubgraphResponse,
} from "@/lib/api";
import {
  fetchKGStats,
  fetchVizKGSubgraph,
  getApiBaseUrl,
  searchKGEntities,
} from "@/lib/api";
import { ApiRecoveryCard } from "@/components/api-recovery-card";
import { KGGraph } from "@/components/kg-graph";
import { ResearchNextActions } from "@/components/research-next-actions";
import { LoadingBlock } from "@/components/spinner";

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
          Knowledge graph
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Hetionet entity counts, full-text entity search, and interactive subgraph explorer.
          To see KG views alongside predictions, embeddings, and model metrics, use the{" "}
          <a href="/visualization?tab=kggraph" className="text-primary underline-offset-2 hover:underline">
            KG Graph tab in Charts &amp; exploration
          </a>.
        </p>
      </header>

      {loading ? (
        <LoadingBlock text="Loading graph stats…" />
      ) : error ? (
        <ApiRecoveryCard title="Could not load KG stats" error={error} />
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

          <SubgraphExplorer
            seedEntity={data.sample_entities[0] ?? null}
            relationTypes={data.relation_types}
          />

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
            <SampleEdgesTable
              edges={data.sample_edges}
              relationTypes={data.relation_types}
            />
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

      <ResearchNextActions context="knowledge-graph" />
    </div>
  );
}

function SampleEdgesTable({
  edges,
  relationTypes,
}: {
  edges: Array<Record<string, string>>;
  relationTypes: string[];
}) {
  const [text, setText] = useState("");
  const [rel, setRel] = useState("all");

  const rows = useMemo(() => {
    const q = text.trim().toLowerCase();
    return edges.filter((e) => {
      if (rel !== "all" && e.metaedge !== rel) return false;
      if (!q) return true;
      return (
        (e.source || "").toLowerCase().includes(q) ||
        (e.target || "").toLowerCase().includes(q) ||
        (e.metaedge || "").toLowerCase().includes(q)
      );
    });
  }, [edges, text, rel]);

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-end justify-between gap-3">
        <h2 className="font-headline text-lg font-semibold text-on-surface">
          Sample edges
        </h2>
        <div className="flex flex-wrap gap-2">
          <input
            type="search"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Filter source/target…"
            className="rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-sm text-on-surface placeholder:text-on-surface-variant/60 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          />
          <select
            value={rel}
            onChange={(e) => setRel(e.target.value)}
            className="rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          >
            <option value="all">All relations</option>
            {relationTypes.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
      </div>
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
            {rows.length === 0 ? (
              <tr>
                <td
                  colSpan={3}
                  className="px-3 py-6 text-center text-sm text-on-surface-variant"
                >
                  No edges match.
                </td>
              </tr>
            ) : (
              rows.map((e, i) => (
                <tr
                  key={i}
                  className="border-b border-outline-variant/10 hover:bg-surface-container-lowest/50"
                >
                  <td className="px-3 py-2 font-mono text-on-surface">
                    {e.source}
                  </td>
                  <td className="px-3 py-2 text-secondary">{e.metaedge}</td>
                  <td className="px-3 py-2 font-mono text-on-surface">
                    {e.target}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-on-surface-variant">
        Showing {rows.length} of {edges.length} edges
      </p>
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

interface SubgraphExplorerProps {
  seedEntity: string | null;
  relationTypes: string[];
}

function SubgraphExplorer({ seedEntity, relationTypes }: SubgraphExplorerProps) {
  const [query, setQuery] = useState("");
  const [entity, setEntity] = useState<string | null>(seedEntity);
  const [maxNodes, setMaxNodes] = useState(40);
  const [hops, setHops] = useState(1);
  const [relationFilter, setRelationFilter] = useState<string>("all");
  const [suggestions, setSuggestions] = useState<KGSearchResult[]>([]);
  const [suggestOpen, setSuggestOpen] = useState(false);
  const [sub, setSub] = useState<VizKGSubgraphResponse | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reqIdRef = useRef(0);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (query.trim().length < 2) {
      setSuggestions([]);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      const reqId = ++reqIdRef.current;
      try {
        const { results } = await searchKGEntities(query.trim(), 8);
        if (reqId === reqIdRef.current) setSuggestions(results);
      } catch {
        if (reqId === reqIdRef.current) setSuggestions([]);
      }
    }, 200);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  useEffect(() => {
    if (!entity) return;
    let cancelled = false;
    setGraphLoading(true);
    setGraphError(null);
    (async () => {
      try {
        const out = await fetchVizKGSubgraph(
          entity,
          maxNodes,
          hops,
          relationFilter !== "all" ? relationFilter : undefined,
        );
        if (!cancelled) setSub(out);
      } catch (e) {
        if (!cancelled) {
          setGraphError(e instanceof Error ? e.message : "Request failed");
          setSub(null);
        }
      } finally {
        if (!cancelled) setGraphLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [entity, maxNodes, hops, relationFilter]);

  function pick(item: KGSearchResult) {
    setQuery(item.name || item.id);
    setEntity(item.id);
    setSuggestOpen(false);
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h2 className="font-headline text-lg font-semibold text-on-surface">
          Interactive subgraph
        </h2>
        {entity ? (
          <span className="text-xs text-on-surface-variant">
            Centered on <span className="font-mono text-tertiary">{entity}</span>
          </span>
        ) : null}
      </div>

      <div className="grid gap-3 md:grid-cols-[2fr_1fr_1fr_1fr]">
        <div className="relative">
          <input
            type="search"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSuggestOpen(true);
            }}
            onFocus={() => setSuggestOpen(true)}
            onBlur={() => setTimeout(() => setSuggestOpen(false), 150)}
            placeholder="Search entity (e.g. Aspirin, DB00945)…"
            className="w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            autoComplete="off"
          />
          {suggestOpen && suggestions.length > 0 ? (
            <ul
              className="absolute left-0 right-0 z-20 mt-1 max-h-60 overflow-auto rounded-lg border border-outline/20 bg-surface-container-high shadow-lg"
              role="listbox"
            >
              {suggestions.map((s) => (
                <li
                  key={s.id}
                  role="option"
                  onMouseDown={(e) => {
                    e.preventDefault();
                    pick(s);
                  }}
                  className="cursor-pointer px-3 py-2 text-sm text-on-surface hover:bg-surface-container-highest"
                >
                  <div className="font-medium">{s.name || s.id}</div>
                  <div className="text-xs text-on-surface-variant">
                    <span className="font-mono">{s.id}</span>
                    <span className="ml-2 uppercase">{s.kind}</span>
                  </div>
                </li>
              ))}
            </ul>
          ) : null}
        </div>

        <select
          value={maxNodes}
          onChange={(e) => setMaxNodes(Number(e.target.value))}
          className="rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          aria-label="Max nodes"
        >
          {[20, 40, 60, 80, 100].map((n) => (
            <option key={n} value={n}>
              {n} nodes
            </option>
          ))}
        </select>

        <select
          value={hops}
          onChange={(e) => setHops(Number(e.target.value))}
          className="rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          aria-label="Hops"
        >
          <option value={1}>1 hop</option>
          <option value={2}>2 hops</option>
        </select>

        <select
          value={relationFilter}
          onChange={(e) => setRelationFilter(e.target.value)}
          className="rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          aria-label="Relation filter"
        >
          <option value="all">All relations</option>
          {relationTypes.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      {!entity ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          Search for an entity above to explore its neighborhood.
        </div>
      ) : graphLoading ? (
        <div className="flex h-[520px] items-center justify-center rounded-lg border border-outline-variant/15 bg-surface-container-lowest/60 text-sm text-on-surface-variant">
          Loading subgraph…
        </div>
      ) : graphError ? (
        <div className="rounded-lg border border-error/40 bg-error-container/20 p-4 text-sm text-error">
          {graphError}
        </div>
      ) : sub && sub.nodes.length > 0 ? (
        <KGGraph
          nodes={sub.nodes}
          links={sub.links}
          centerId={sub.center_entity ?? entity}
        />
      ) : (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          {sub?.message ?? "No neighbors found for this entity."}
        </div>
      )}
    </div>
  );
}
