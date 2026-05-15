"use client";

import { useEffect, useState } from "react";
import type { VizCircuitResponse } from "@/lib/api";
import { fetchVizCircuitParams } from "@/lib/api";
import { LoadingBlock } from "@/components/spinner";

export function QuantumCircuitDiagram() {
  const [data, setData] = useState<VizCircuitResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetchVizCircuitParams();
        if (!cancelled) setData(r);
      } catch (e) {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Request failed");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <LoadingBlock text="Loading circuit parameters…" />;
  if (error)
    return (
      <div className="rounded-lg border border-error/40 bg-error-container/20 p-4 text-sm text-error">
        {error}
      </div>
    );
  if (!data || data.status !== "ok") return null;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Chip label="Feature map" value={data.feature_map} tone="primary" />
        <Chip label="Qubits" value={String(data.n_qubits)} tone="tertiary" />
        <Chip label="Reps" value={String(data.n_reps)} tone="secondary" />
        <Chip
          label="Entanglement"
          value={data.entanglement}
          tone="primary"
        />
        {data.execution_mode ? (
          <Chip label="Execution" value={data.execution_mode} tone="default" />
        ) : null}
        {data.backend ? (
          <Chip label="Backend" value={data.backend} tone="default" />
        ) : null}
        {data.shots != null ? (
          <Chip label="Shots" value={String(data.shots)} tone="default" />
        ) : null}
      </div>

      <CircuitSVG
        nQubits={data.n_qubits}
        nReps={data.n_reps}
        featureMap={data.feature_map}
        entanglement={data.entanglement}
      />

      <p className="text-xs text-on-surface-variant">
        The {data.feature_map} feature map encodes a {data.n_qubits}-dim
        vector into {data.n_qubits} qubits and applies {data.n_reps}{" "}
        repetition{data.n_reps === 1 ? "" : "s"} with{" "}
        <span className="font-mono text-on-surface">{data.entanglement}</span>{" "}
        entanglement. The kernel measures state overlaps between pairs of
        encoded data points.
      </p>
    </div>
  );
}

function Chip({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "primary" | "secondary" | "tertiary" | "default";
}) {
  const toneClass =
    tone === "primary"
      ? "text-primary"
      : tone === "secondary"
        ? "text-secondary"
        : tone === "tertiary"
          ? "text-tertiary"
          : "text-on-surface";
  return (
    <div className="rounded-lg bg-surface-container-lowest/80 px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </dt>
      <dd className={`mt-1 font-mono text-sm ${toneClass}`}>{value}</dd>
    </div>
  );
}

interface CircuitSVGProps {
  nQubits: number;
  nReps: number;
  featureMap: string;
  entanglement: string;
}

function CircuitSVG({
  nQubits,
  nReps,
  featureMap,
  entanglement,
}: CircuitSVGProps) {
  const qubitsToShow = Math.min(nQubits, 6);
  const truncated = nQubits > qubitsToShow;
  const repsToShow = Math.min(nReps, 3);
  const repsTruncated = nReps > repsToShow;

  const rowHeight = 44;
  const colWidth = 54;
  const leftPad = 50;
  const rightPad = 30;
  const topPad = 20;

  const isZZ = /zz/i.test(featureMap);
  const gatesPerRep = isZZ ? 3 : 2; // H, Rz (and ZZ entangler for ZZ)
  const entanglerCol = gatesPerRep - 1;
  const cols = repsToShow * gatesPerRep;
  const svgWidth = leftPad + cols * colWidth + rightPad;
  const svgHeight = topPad * 2 + qubitsToShow * rowHeight;

  const qubitIdxs = Array.from({ length: qubitsToShow }, (_, i) => i);
  const repIdxs = Array.from({ length: repsToShow }, (_, i) => i);

  function yOf(q: number) {
    return topPad + q * rowHeight + rowHeight / 2;
  }
  function xOf(rep: number, col: number) {
    return leftPad + (rep * gatesPerRep + col) * colWidth + colWidth / 2;
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-outline-variant/20 bg-surface-container-lowest/60 p-3">
      <svg
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        className="w-full"
        role="img"
        aria-label={`${featureMap} feature map circuit with ${nQubits} qubits and ${nReps} repetitions`}
      >
        {/* Rep background bands */}
        {repIdxs.map((r) => (
          <rect
            key={`rep-${r}`}
            x={leftPad + r * gatesPerRep * colWidth}
            y={topPad - 6}
            width={gatesPerRep * colWidth}
            height={svgHeight - topPad * 2 + 12}
            fill={r % 2 === 0 ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.05)"}
            rx={6}
          />
        ))}

        {/* Qubit wires + labels */}
        {qubitIdxs.map((q) => (
          <g key={`q-${q}`}>
            <text
              x={10}
              y={yOf(q) + 4}
              fontSize={12}
              fill="#cdd6e4"
              fontFamily="ui-monospace, monospace"
            >
              q{q}
            </text>
            <line
              x1={leftPad - 6}
              y1={yOf(q)}
              x2={svgWidth - rightPad + 6}
              y2={yOf(q)}
              stroke="#5f6b80"
              strokeWidth={1.2}
            />
          </g>
        ))}

        {/* Rep labels */}
        {repIdxs.map((r) => (
          <text
            key={`rep-label-${r}`}
            x={leftPad + (r * gatesPerRep + gatesPerRep / 2) * colWidth}
            y={12}
            fontSize={10}
            fill="#8792a8"
            textAnchor="middle"
          >
            rep {r + 1}
          </text>
        ))}

        {/* Gates */}
        {repIdxs.map((r) => (
          <g key={`gates-${r}`}>
            {qubitIdxs.map((q) => (
              <GateBox
                key={`h-${r}-${q}`}
                x={xOf(r, 0)}
                y={yOf(q)}
                label="H"
                fill="#4f6fff"
              />
            ))}
            {qubitIdxs.map((q) => (
              <GateBox
                key={`rz-${r}-${q}`}
                x={xOf(r, 1)}
                y={yOf(q)}
                label={isZZ ? "P(x)" : "Rz(x)"}
                fill="#3cddc7"
                width={42}
              />
            ))}
            {isZZ
              ? qubitIdxs.slice(0, -1).map((q) => (
                  <g key={`zz-${r}-${q}`}>
                    <line
                      x1={xOf(r, entanglerCol)}
                      y1={yOf(q)}
                      x2={xOf(r, entanglerCol)}
                      y2={yOf(q + 1)}
                      stroke="#ff9aa2"
                      strokeWidth={2}
                    />
                    <circle
                      cx={xOf(r, entanglerCol)}
                      cy={yOf(q)}
                      r={4}
                      fill="#ff9aa2"
                    />
                    <circle
                      cx={xOf(r, entanglerCol)}
                      cy={yOf(q + 1)}
                      r={4}
                      fill="#ff9aa2"
                    />
                    <text
                      x={xOf(r, entanglerCol) + 8}
                      y={(yOf(q) + yOf(q + 1)) / 2 + 3}
                      fontSize={9}
                      fill="#ff9aa2"
                      fontFamily="ui-monospace, monospace"
                    >
                      ZZ
                    </text>
                  </g>
                ))
              : null}
          </g>
        ))}

        {/* Right-side measurement */}
        {qubitIdxs.map((q) => (
          <GateBox
            key={`m-${q}`}
            x={svgWidth - rightPad - 12}
            y={yOf(q)}
            label="⟨⟩"
            fill="#8792a8"
            width={26}
          />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-on-surface-variant">
        <Legend color="#4f6fff" label="Hadamard" />
        <Legend color="#3cddc7" label={isZZ ? "Phase(xᵢ)" : "Rotation Rz(xᵢ)"} />
        {isZZ ? <Legend color="#ff9aa2" label={`${entanglement} entangler`} /> : null}
        <Legend color="#8792a8" label="Measurement" />
        {truncated ? (
          <span className="italic">showing 6 of {nQubits} qubits</span>
        ) : null}
        {repsTruncated ? (
          <span className="italic">showing 3 of {nReps} reps</span>
        ) : null}
      </div>
    </div>
  );
}

function GateBox({
  x,
  y,
  label,
  fill,
  width = 30,
}: {
  x: number;
  y: number;
  label: string;
  fill: string;
  width?: number;
}) {
  const h = 24;
  return (
    <g>
      <rect
        x={x - width / 2}
        y={y - h / 2}
        width={width}
        height={h}
        rx={4}
        fill={fill}
        fillOpacity={0.9}
        stroke="#1b1f2a"
        strokeWidth={1}
      />
      <text
        x={x}
        y={y + 3.5}
        fontSize={11}
        textAnchor="middle"
        fill="#0f1216"
        fontFamily="ui-monospace, monospace"
        fontWeight={600}
      >
        {label}
      </text>
    </g>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span
        className="inline-block h-2.5 w-2.5 rounded"
        style={{ backgroundColor: color }}
      />
      {label}
    </span>
  );
}
