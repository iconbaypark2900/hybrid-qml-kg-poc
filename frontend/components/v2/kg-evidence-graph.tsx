"use client";

import { KGGraph } from "@/components/kg-graph";
import type { V2Candidate } from "@/lib/v2-data";
import type { V2GraphEvidence } from "@/lib/v2-live-data";

interface KGEvidenceGraphProps {
  evidence: V2GraphEvidence;
  candidate: V2Candidate;
}

export function KGEvidenceGraph({ evidence, candidate }: KGEvidenceGraphProps) {
  if (evidence.nodes.length > 0) {
    return (
      <div className="space-y-3">
        <KGGraph
          nodes={evidence.nodes}
          links={evidence.links}
          centerId={evidence.centerEntity}
          height={320}
        />
        <EvidenceCaption source={evidence.source} message={evidence.message} />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="relative min-h-[260px] rounded-lg border border-outline-variant/25 bg-surface-container-lowest">
        <GraphNode label="Drug" x="18%" y="48%" tone="bg-tertiary" />
        <GraphNode label="Target" x="44%" y="25%" tone="bg-primary" />
        <GraphNode label="Path" x="61%" y="55%" tone="bg-[#f8c64f]" />
        <GraphNode label="Dx" x="82%" y="42%" tone="bg-error" />
        <GraphEdge x1="24%" y1="51%" x2="43%" y2="31%" />
        <GraphEdge x1="51%" y1="31%" x2="59%" y2="53%" />
        <GraphEdge x1="66%" y1="55%" x2="78%" y2="46%" />
      </div>
      <EvidenceCaption
        source={evidence.source}
        message={
          evidence.message ??
          `Fallback graph sketch for ${candidate.candidate} to ${candidate.disease}.`
        }
      />
    </div>
  );
}

function EvidenceCaption({
  source,
  message,
}: {
  source: string;
  message: string | null;
}) {
  return (
    <p className="rounded-lg border border-outline-variant/25 bg-surface-container-high/70 p-3 text-xs leading-relaxed text-on-surface-variant">
      <span className="font-semibold text-on-surface">Graph source: {source}.</span>{" "}
      {message}
    </p>
  );
}

function GraphNode({
  label,
  x,
  y,
  tone,
}: {
  label: string;
  x: string;
  y: string;
  tone: string;
}) {
  return (
    <span
      className={`absolute flex h-14 w-14 items-center justify-center rounded-full ${tone} text-xs font-bold text-background`}
      style={{ left: x, top: y }}
    >
      {label}
    </span>
  );
}

function GraphEdge({
  x1,
  y1,
  x2,
  y2,
}: {
  x1: string;
  y1: string;
  x2: string;
  y2: string;
}) {
  return (
    <span
      className="absolute h-0.5 origin-left bg-outline/60"
      style={{
        left: x1,
        top: y1,
        width: `calc(${x2} - ${x1})`,
        transform: "rotate(-24deg)",
      }}
    />
  );
}
