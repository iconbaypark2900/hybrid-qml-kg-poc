"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import type { Hypothesis, JobCreateRequest } from "@/lib/api";
import { createPipelineJob, fetchHypotheses } from "@/lib/api";
import { ResearchNextActions } from "@/components/research-next-actions";

interface RunFormState {
  relation: string;
  embedding_method: string;
  embedding_dim: number;
  embedding_epochs: number;
  qml_dim: number;
  qml_feature_map: string;
  qml_feature_map_reps: number;
  fast_mode: boolean;
  skip_quantum: boolean;
  run_ensemble: boolean;
  ensemble_method: string;
  tune_classical: boolean;
  results_dir: string;
  quantum_config_path: string;
  hypothesis_id: string;
  experiment_note: string;
  experiment_tags: string;
}

const DEFAULTS: RunFormState = {
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
  hypothesis_id: "",
  experiment_note: "",
  experiment_tags: "",
};

const PRESETS: Record<string, Partial<RunFormState>> = {
  "classical-baseline": {
    skip_quantum: true,
    tune_classical: true,
    run_ensemble: false,
    embedding_method: "RotatE",
    embedding_dim: 128,
    embedding_epochs: 150,
  },
  "hybrid-default": {
    skip_quantum: false,
    run_ensemble: true,
    ensemble_method: "weighted_average",
    tune_classical: true,
    qml_dim: 12,
    qml_feature_map: "ZZ",
    qml_feature_map_reps: 2,
  },
  "quantum-heavy": {
    skip_quantum: false,
    run_ensemble: true,
    tune_classical: false,
    qml_dim: 16,
    qml_feature_map: "Pauli",
    qml_feature_map_reps: 3,
    fast_mode: false,
  },
  "hardware-ready": {
    skip_quantum: false,
    run_ensemble: false,
    qml_dim: 8,
    qml_feature_map: "ZZ",
    qml_feature_map_reps: 1,
    quantum_config_path: "config/quantum_config.yaml",
  },
};

export default function SimulationParametersPage() {
  const router = useRouter();
  const [params, setParams] = useState<RunFormState>({ ...DEFAULTS });
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [preset, setPreset] = useState<string>("hybrid-default");
  const [loading, setLoading] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [error, setError] = useState<string | null>(null);

  function set<K extends keyof RunFormState>(key: K, value: RunFormState[K]) {
    setParams((prev) => ({ ...prev, [key]: value }));
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await fetchHypotheses().catch(() => []);
        if (!cancelled) {
          setHypotheses(list);
          setParams((prev) => ({
            ...prev,
            hypothesis_id: prev.hypothesis_id || list[0]?.id || "",
          }));
          const query = new URLSearchParams(window.location.search);
          const presetParam = query.get("preset");
          if (presetParam && PRESETS[presetParam]) {
            setPreset(presetParam);
            setParams((prev) => ({ ...prev, ...PRESETS[presetParam] }));
          } else {
            setParams((prev) => ({ ...prev, ...PRESETS["hybrid-default"] }));
          }
        }
      } finally {
        if (!cancelled) setBootstrapping(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedHypothesis = useMemo(
    () => hypotheses.find((h) => h.id === params.hypothesis_id) ?? null,
    [hypotheses, params.hypothesis_id],
  );

  function applyPreset(nextPreset: string) {
    setPreset(nextPreset);
    setParams((prev) => ({ ...prev, ...PRESETS[nextPreset] }));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload: JobCreateRequest = {
        relation: params.relation,
        embedding_method: params.embedding_method,
        embedding_dim: params.embedding_dim,
        embedding_epochs: params.embedding_epochs,
        qml_dim: params.qml_dim,
        qml_feature_map: params.qml_feature_map,
        qml_feature_map_reps: params.qml_feature_map_reps,
        fast_mode: params.fast_mode,
        skip_quantum: params.skip_quantum,
        run_ensemble: params.run_ensemble,
        ensemble_method: params.ensemble_method,
        tune_classical: params.tune_classical,
        results_dir: params.results_dir,
        quantum_config_path: params.quantum_config_path,
        hypothesis_id: params.hypothesis_id || undefined,
        experiment_note: params.experiment_note || undefined,
        experiment_tags: params.experiment_tags
          ? params.experiment_tags
              .split(",")
              .map((t) => t.trim())
              .filter(Boolean)
          : undefined,
      };
      await createPipelineJob(payload);
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
          New run
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Configure and launch a pipeline run linked to a hypothesis, then track results in Pipeline jobs.
        </p>
      </header>

      <section className="rounded-lg border border-outline-variant/20 bg-surface-container-lowest/60 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
          Run presets
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <PresetButton
            active={preset === "classical-baseline"}
            onClick={() => applyPreset("classical-baseline")}
            label="Classical baseline"
          />
          <PresetButton
            active={preset === "hybrid-default"}
            onClick={() => applyPreset("hybrid-default")}
            label="Hybrid default"
          />
          <PresetButton
            active={preset === "quantum-heavy"}
            onClick={() => applyPreset("quantum-heavy")}
            label="Quantum-heavy"
          />
          <PresetButton
            active={preset === "hardware-ready"}
            onClick={() => applyPreset("hardware-ready")}
            label="Hardware-ready"
          />
        </div>
      </section>

      {selectedHypothesis ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-4">
          <p className="text-xs uppercase tracking-wide text-on-surface-variant">
            Active hypothesis
          </p>
          <p className="mt-1 text-sm font-medium text-on-surface">
            {selectedHypothesis.id} · {selectedHypothesis.name}
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">
            {selectedHypothesis.description}
          </p>
        </div>
      ) : null}

      <form onSubmit={onSubmit} className="max-w-2xl space-y-5">
        <fieldset className="space-y-4 rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5">
          <legend className="text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
            Experiment context
          </legend>
          <Row label="Hypothesis">
            <select
              className="w-48 rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              value={params.hypothesis_id}
              onChange={(e) => set("hypothesis_id", e.target.value)}
              disabled={bootstrapping}
            >
              <option value="">None</option>
              {hypotheses.map((h) => (
                <option key={h.id} value={h.id}>
                  {h.id} · {h.name}
                </option>
              ))}
            </select>
          </Row>
          <div>
            <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
              Experiment note
            </label>
            <textarea
              rows={2}
              className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              value={params.experiment_note}
              onChange={(e) => set("experiment_note", e.target.value)}
              placeholder="What are you testing in this run?"
            />
          </div>
          <div>
            <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
              Tags (comma-separated)
            </label>
            <input
              className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              value={params.experiment_tags}
              onChange={(e) => set("experiment_tags", e.target.value)}
              placeholder="baseline, qsvc, follow-up"
            />
          </div>
        </fieldset>

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
          disabled={loading || bootstrapping}
          className="primary-gradient rounded-lg px-5 py-2.5 text-sm font-semibold text-on-primary shadow-glow disabled:opacity-50"
        >
          {loading ? "Submitting…" : "Launch pipeline"}
        </button>
      </form>

      <ResearchNextActions context="simulation" />
    </div>
  );
}

function PresetButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
        active
          ? "border-primary/40 bg-primary/15 text-primary"
          : "border-outline/20 bg-surface-container-lowest text-on-surface hover:bg-surface-container"
      }`}
    >
      {label}
    </button>
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
