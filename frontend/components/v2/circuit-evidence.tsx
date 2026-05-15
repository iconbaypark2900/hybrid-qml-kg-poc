"use client";

import type { V2CircuitEvidence } from "@/lib/v2-live-data";

export function CircuitEvidence({ evidence }: { evidence: V2CircuitEvidence }) {
  const params = evidence.params ?? {
    feature_map: "Pauli",
    n_qubits: 16,
    n_reps: 2,
    entanglement: "full",
    execution_mode: null,
    backend: null,
    shots: null,
    status: "fallback",
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Chip label="Feature map" value={params.feature_map} />
        <Chip label="Qubits" value={String(params.n_qubits)} />
        <Chip label="Reps" value={String(params.n_reps)} />
        <Chip label="Entanglement" value={params.entanglement} />
      </div>
      <CircuitRows
        nQubits={params.n_qubits}
        nReps={params.n_reps}
        featureMap={params.feature_map}
      />
      <p className="rounded-lg border border-outline-variant/25 bg-surface-container-high/70 p-3 text-xs leading-relaxed text-on-surface-variant">
        <span className="font-semibold text-on-surface">
          Circuit source: {evidence.source}.
        </span>{" "}
        {evidence.message ??
          "The panel labels the quantum branch used for the hybrid evidence view."}
      </p>
    </div>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-surface-container-lowest/80 px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </dt>
      <dd className="mt-1 font-mono text-sm text-primary">{value}</dd>
    </div>
  );
}

function CircuitRows({
  nQubits,
  nReps,
  featureMap,
}: {
  nQubits: number;
  nReps: number;
  featureMap: string;
}) {
  const qubits = Array.from({ length: Math.min(nQubits, 6) }, (_, index) => index);
  const reps = Array.from({ length: Math.min(nReps, 3) }, (_, index) => index);
  const gateLabel = /zz/i.test(featureMap) ? "ZZ" : "P";

  return (
    <div className="overflow-x-auto rounded-lg border border-secondary/25 bg-surface-container-lowest p-5">
      <div className="min-w-[720px] space-y-4">
        {qubits.map((qubit) => (
          <div key={qubit} className="grid grid-cols-[70px_1fr] items-center gap-4">
            <div className="font-mono text-sm font-bold text-secondary">|0&gt; q{qubit}</div>
            <div className="relative flex h-12 items-center">
              <div className="absolute left-0 right-0 h-0.5 bg-secondary/40" />
              <div className="relative z-10 grid w-full grid-cols-6 gap-8">
                {reps.flatMap((rep) => [
                  <Gate key={`${qubit}-${rep}-h`} label="H" />,
                  <Gate key={`${qubit}-${rep}-rz`} label="RZ" />,
                  <Gate key={`${qubit}-${rep}-map`} label={gateLabel} />,
                ])}
              </div>
            </div>
          </div>
        ))}
      </div>
      {nQubits > 6 ? (
        <p className="mt-3 text-xs text-on-surface-variant">
          Showing 6 of {nQubits} qubit wires for readability.
        </p>
      ) : null}
    </div>
  );
}

function Gate({ label }: { label: string }) {
  return (
    <span className="flex h-10 items-center justify-center rounded-lg border border-secondary/40 bg-secondary/15 font-mono text-sm font-bold text-secondary">
      {label}
    </span>
  );
}
