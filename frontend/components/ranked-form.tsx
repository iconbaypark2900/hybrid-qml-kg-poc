"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { RankedMechanismsResponse } from "@/lib/api";
import { rankedMechanisms } from "@/lib/api";

interface RankedFormProps {
  hypothesisId: string;
  onHypothesisIdChange: (hypothesisId: string) => void;
  diseaseFocus?: string | null;
}

export function RankedForm({
  hypothesisId,
  onHypothesisIdChange,
  diseaseFocus,
}: RankedFormProps) {
  const [diseaseId, setDiseaseId] = useState("");
  const [topK, setTopK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RankedMechanismsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!diseaseId && diseaseFocus) {
      setDiseaseId(diseaseFocus);
    }
  }, [diseaseFocus, diseaseId]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const out = await rankedMechanisms({
        hypothesis_id: hypothesisId,
        disease_id: diseaseId,
        top_k: topK,
      });
      setResult(out);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ranking failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={onSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
            Hypothesis ID
          </label>
          <input
            className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            value={hypothesisId}
            onChange={(e) => onHypothesisIdChange(e.target.value)}
            placeholder="Select a hypothesis from the list above"
            required
          />
        </div>
        <div>
          <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
            Disease (name or ID)
          </label>
          <input
            className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface placeholder:text-on-surface-variant/60 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            value={diseaseId}
            onChange={(e) => setDiseaseId(e.target.value)}
            placeholder="e.g. DOID_9352"
            required
          />
        </div>
        <div>
          <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
            Top K
          </label>
          <input
            type="number"
            min={1}
            max={200}
            className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="primary-gradient rounded-lg px-5 py-2.5 text-sm font-semibold text-on-primary shadow-glow disabled:opacity-50"
        >
          {loading ? "Ranking…" : "Rank candidates"}
        </button>
      </form>

      {error ? (
        <p className="text-sm text-error" role="alert">
          {error}
        </p>
      ) : null}

      {result?.ranked_candidates?.length ? (
        <ul className="space-y-2">
          {result.ranked_candidates.map((c) => (
            <li
              key={`${c.compound_id}-${c.score}`}
              className="rounded-lg border border-outline/10 bg-surface-container-lowest/50 px-3 py-2"
            >
              <p className="font-medium text-on-surface">{c.compound_name}</p>
              <p className="text-xs text-tertiary">
                score {c.score.toFixed(4)} · {c.compound_id}
              </p>
              <p className="mt-1 text-xs text-on-surface-variant">
                {c.mechanism_summary}
              </p>
              <div className="mt-2 flex flex-wrap gap-2 text-xs">
                <Link
                  href={`/predict?drug=${encodeURIComponent(c.compound_id)}&disease=${encodeURIComponent(diseaseId)}`}
                  className="rounded border border-outline/20 bg-surface-container-lowest px-2 py-1 text-on-surface hover:bg-surface-container"
                >
                  Predict pair
                </Link>
                <Link
                  href={`/knowledge-graph?entity=${encodeURIComponent(c.compound_id)}`}
                  className="rounded border border-outline/20 bg-surface-container-lowest px-2 py-1 text-on-surface hover:bg-surface-container"
                >
                  KG drill-in
                </Link>
                <Link
                  href={`/visualization?tab=predictions`}
                  className="rounded border border-outline/20 bg-surface-container-lowest px-2 py-1 text-on-surface hover:bg-surface-container"
                >
                  Open chart context
                </Link>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
