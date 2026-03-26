"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { JobCreateRequest } from "@/lib/api";
import { createPipelineJob } from "@/lib/api";

const DEFAULTS: Required<JobCreateRequest> = {
  relation: "CtD",
  embedding_method: "ComplEx",
  embedding_dim: 64,
  embedding_epochs: 100,
  qml_dim: 12,
  qml_feature_map: "ZZ",
  qml_feature_map_reps: 2,
  fast_mode: true,
  skip_quantum: false,
  run_ensemble: false,
  ensemble_method: "weighted_average",
  tune_classical: false,
  results_dir: "results",
  quantum_config_path: "config/quantum_config.yaml",
};

export default function SimulationParametersPage() {
  const router = useRouter();
  const [params, setParams] = useState<Required<JobCreateRequest>>({ ...DEFAULTS });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function set<K extends keyof JobCreateRequest>(key: K, value: JobCreateRequest[K]) {
    setParams((prev) => ({ ...prev, [key]: value }));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await createPipelineJob(params);
      router.push("/simulation");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Job creation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          Simulation parameters
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Configure and launch a pipeline run.
        </p>
      </header>

      <form onSubmit={onSubmit} className="max-w-2xl space-y-5">
        <fieldset className="space-y-4 rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5">
          <legend className="text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
            Embeddings
          </legend>
          <Row label="Relation">
            <Select value={params.relation} onChange={(v) => set("relation", v)} options={["CtD", "DaG", "CbG"]} />
          </Row>
          <Row label="Method">
            <Select value={params.embedding_method} onChange={(v) => set("embedding_method", v)} options={["ComplEx", "RotatE", "TransE", "DistMult"]} />
          </Row>
          <Row label="Dimension">
            <NumberInput value={params.embedding_dim} onChange={(v) => set("embedding_dim", v)} min={8} max={512} />
          </Row>
          <Row label="Epochs">
            <NumberInput value={params.embedding_epochs} onChange={(v) => set("embedding_epochs", v)} min={1} max={1000} />
          </Row>
        </fieldset>

        <fieldset className="space-y-4 rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5">
          <legend className="text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
            Quantum
          </legend>
          <Row label="Qubits">
            <NumberInput value={params.qml_dim} onChange={(v) => set("qml_dim", v)} min={2} max={32} />
          </Row>
          <Row label="Feature map">
            <Select value={params.qml_feature_map} onChange={(v) => set("qml_feature_map", v)} options={["ZZ", "Z", "Pauli", "custom_link_prediction"]} />
          </Row>
          <Row label="Feature map reps">
            <NumberInput value={params.qml_feature_map_reps} onChange={(v) => set("qml_feature_map_reps", v)} min={1} max={8} />
          </Row>
          <Row label="Skip quantum">
            <Toggle checked={params.skip_quantum} onChange={(v) => set("skip_quantum", v)} />
          </Row>
        </fieldset>

        <fieldset className="space-y-4 rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5">
          <legend className="text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
            Training
          </legend>
          <Row label="Fast mode">
            <Toggle checked={params.fast_mode} onChange={(v) => set("fast_mode", v)} />
          </Row>
          <Row label="Tune classical">
            <Toggle checked={params.tune_classical} onChange={(v) => set("tune_classical", v)} />
          </Row>
          <Row label="Run ensemble">
            <Toggle checked={params.run_ensemble} onChange={(v) => set("run_ensemble", v)} />
          </Row>
          {params.run_ensemble ? (
            <Row label="Ensemble method">
              <Select value={params.ensemble_method} onChange={(v) => set("ensemble_method", v)} options={["weighted_average", "voting", "stacking"]} />
            </Row>
          ) : null}
        </fieldset>

        {error ? (
          <p className="text-sm text-error" role="alert">{error}</p>
        ) : null}

        <button
          type="submit"
          disabled={loading}
          className="primary-gradient rounded-lg px-5 py-2.5 text-sm font-semibold text-on-primary shadow-glow disabled:opacity-50"
        >
          {loading ? "Submitting…" : "Launch pipeline"}
        </button>
      </form>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <label className="text-sm text-on-surface-variant">{label}</label>
      {children}
    </div>
  );
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <select
      className="rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  );
}

function NumberInput({ value, onChange, min, max }: { value: number; onChange: (v: number) => void; min?: number; max?: number }) {
  return (
    <input
      type="number"
      className="w-24 rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      min={min}
      max={max}
    />
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${checked ? "bg-primary" : "bg-outline/30"}`}
    >
      <span
        className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-on-surface shadow-sm transition-transform ${checked ? "translate-x-5" : "translate-x-0"}`}
      />
    </button>
  );
}
