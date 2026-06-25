"use client";

import { useEffect, useState } from "react";
import { Card, EvidenceCard, StatusBadge } from "@/components/v2/v2-shell";
import { fetchFeatureImportance, type FeatureImportanceResponse, type FeatureImportanceItem } from "@/lib/api";

type LoadState = "loading" | "ok" | "empty" | "error";

export function MoAExplanationPanel() {
  const [state, setState] = useState<LoadState>("loading");
  const [data, setData] = useState<FeatureImportanceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchFeatureImportance()
      .then((resp) => {
        if (!active) return;
        setData(resp);
        if (resp.status === "ok") setState("ok");
        else if (resp.status === "empty") setState("empty");
        else setState("error");
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Request failed");
        setState("error");
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <Card
      title="Mechanism-of-Action explanation"
      kicker={
        state === "loading"
          ? "Loading…"
          : state === "ok"
            ? `${data?.model_name ?? "best classical"} — PR-AUC ${data?.pr_auc?.toFixed(4) ?? "?"}`
            : state === "empty"
              ? "No run artifact"
              : "API error"
      }
      help="Top features by RandomForest importance from the latest pipeline run. MoA features (mechanism-of-action) are highlighted to show how much of the prediction is driven by mechanism evidence vs. embeddings/structure."
    >
      {state === "loading" ? (
        <p className="text-sm text-on-surface-variant">Fetching feature importances from the latest run…</p>
      ) : state === "error" ? (
        <p className="text-sm text-on-error">Failed to load feature importances: {error ?? data?.message}</p>
      ) : state === "empty" ? (
        <p className="text-sm text-on-surface-variant">
          {data?.message ?? "No feature importances are available yet."} Run the pipeline at least once with
          <code className="mx-1 rounded bg-surface-container-highest px-1 py-0.5 text-xs">--use_moa_features</code>
          to populate this panel.
        </p>
      ) : data ? (
        <FeatureImportanceContent data={data} />
      ) : null}
    </Card>
  );
}

function FeatureImportanceContent({ data }: { data: FeatureImportanceResponse }) {
  const top = data.top_features ?? [];
  const maxImportance = Math.max(...top.map((f) => f.importance), 0.0001);
  const moaCount = top.filter((f) => f.is_moa).length;
  const totalFeatures = data.feature_count ?? top.length;
  const hasNames = !!data.feature_names;
  const useMoa = (data.config?.use_moa_features as boolean | undefined) ?? false;
  const useGraph = (data.config?.use_graph_features as boolean | undefined) ?? false;
  const useDomain = (data.config?.use_domain_features as boolean | undefined) ?? false;

  return (
    <div className="space-y-4">
      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <EvidenceCard
          title="Best classical model"
          value={data.model_name ?? "—"}
          detail={
            data.pr_auc != null
              ? `PR-AUC ${data.pr_auc.toFixed(4)} on test set`
              : "PR-AUC not recorded"
          }
          tone="success"
        />
        <EvidenceCard
          title="MoA in top-20"
          value={`${moaCount}`}
          detail={
            hasNames
              ? `${moaCount}/20 top features start with moa_`
              : "Feature names not yet recorded by the latest run"
          }
          tone={moaCount > 0 ? "quantum" : "warning"}
        />
        <EvidenceCard
          title="Feature budget"
          value={`${totalFeatures}`}
          detail={
            hasNames
              ? `Graph=${useGraph ? "on" : "off"}, domain=${useDomain ? "on" : "off"}, MoA=${useMoa ? "on" : "off"}`
              : "Re-run pipeline to record feature_names"
          }
          tone="success"
        />
        <EvidenceCard
          title="Run signal"
          value={data.message ?? "ok"}
          detail={hasNames ? "Latest run shipped feature_names" : "Latest run predates the feature_names save"}
          tone={hasNames ? "success" : "warning"}
        />
      </section>

      <div>
        <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">
          Top {top.length} features by importance
        </p>
        <ul className="mt-3 space-y-2">
          {top.map((f) => (
            <FeatureBar key={`${f.index}-${f.name}`} feature={f} max={maxImportance} />
          ))}
        </ul>
      </div>

      {data.message && hasNames ? (
        <p className="text-xs text-on-surface-variant">{data.message}</p>
      ) : null}
    </div>
  );
}

function FeatureBar({ feature, max }: { feature: FeatureImportanceItem; max: number }) {
  const pct = Math.max(2, Math.round((feature.importance / max) * 100));
  return (
    <li className="flex items-center gap-3">
      <span className="w-6 shrink-0 text-right font-mono text-xs text-on-surface-variant">{feature.rank}</span>
      <span
        className={`min-w-0 flex-1 truncate rounded-md border px-2 py-1 font-mono text-xs ${
          feature.is_moa
            ? "border-quantum/40 bg-quantum/15 font-semibold text-quantum"
            : "border-outline-variant/30 bg-surface-container-high text-on-surface"
        }`}
        title={feature.name}
      >
        {feature.is_moa ? "MoA · " : ""}
        {feature.name}
      </span>
      <span className="w-32 shrink-0">
        <span
          className={`block h-2 rounded-full ${feature.is_moa ? "bg-quantum/70" : "bg-primary/70"}`}
          style={{ width: `${pct}%` }}
          aria-hidden
        />
      </span>
      <span className="w-16 shrink-0 text-right font-mono text-xs text-on-surface-variant">
        {feature.importance.toFixed(4)}
      </span>
    </li>
  );
}

export function MoAStatusChip() {
  const [state, setState] = useState<LoadState>("loading");
  const [data, setData] = useState<FeatureImportanceResponse | null>(null);

  useEffect(() => {
    let active = true;
    fetchFeatureImportance()
      .then((resp) => {
        if (!active) return;
        setData(resp);
        if (resp.status === "ok") setState("ok");
        else setState("empty");
      })
      .catch(() => {
        if (active) setState("error");
      });
    return () => {
      active = false;
    };
  }, []);

  if (state === "loading") return <StatusBadge tone="warning">MoA: loading</StatusBadge>;
  if (state === "error") return <StatusBadge tone="danger">MoA: error</StatusBadge>;
  if (state === "empty" || !data) return <StatusBadge tone="warning">MoA: no data</StatusBadge>;
  const moaCount = data.top_features.filter((f) => f.is_moa).length;
  const tone = moaCount > 0 ? "quantum" : "success";
  return <StatusBadge tone={tone}>MoA: {moaCount}/20 in top features</StatusBadge>;
}
