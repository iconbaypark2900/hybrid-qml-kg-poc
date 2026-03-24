"use client";

import { useState } from "react";
import type { PredictionResponse } from "@/lib/api";
import { predictLink } from "@/lib/api";

export function PredictForm() {
  const [drug, setDrug] = useState("");
  const [disease, setDisease] = useState("");
  const [method, setMethod] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const out = await predictLink({ drug, disease, method });
      setResult(out);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={onSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
            Drug / compound
          </label>
          <input
            className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface placeholder:text-on-surface-variant/60 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            value={drug}
            onChange={(e) => setDrug(e.target.value)}
            placeholder="e.g. Aspirin or DB00945"
            required
          />
        </div>
        <div>
          <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
            Disease
          </label>
          <input
            className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface placeholder:text-on-surface-variant/60 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            value={disease}
            onChange={(e) => setDisease(e.target.value)}
            placeholder="e.g. Diabetes or DOID_9352"
            required
          />
        </div>
        <div>
          <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
            Method
          </label>
          <select
            className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
          >
            <option value="auto">auto</option>
            <option value="classical">classical</option>
            <option value="quantum">quantum</option>
          </select>
        </div>
        <button
          type="submit"
          disabled={loading}
          className="primary-gradient rounded-lg px-5 py-2.5 text-sm font-semibold text-on-primary shadow-glow disabled:opacity-50"
        >
          {loading ? "Resolving probability…" : "Predict link"}
        </button>
      </form>

      {error ? (
        <p className="text-sm text-error" role="alert">
          {error}
        </p>
      ) : null}

      {result && result.status === "success" ? (
        <div className="rounded-lg border border-tertiary/30 bg-surface-container-lowest/50 p-4">
          <p className="text-xs uppercase text-on-surface-variant">Link probability</p>
          <p className="text-2xl font-semibold text-tertiary">
            {(result.link_probability * 100).toFixed(2)}%
          </p>
          <p className="mt-2 text-xs text-on-surface-variant">
            Model: {result.model_used} · {result.drug_id} → {result.disease_id}
          </p>
        </div>
      ) : null}
    </div>
  );
}
