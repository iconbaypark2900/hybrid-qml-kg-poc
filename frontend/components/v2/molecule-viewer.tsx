"use client";

import { useEffect, useRef, useState } from "react";
import type { VizAtom, VizBond } from "@/lib/api";
import type { EvidenceSource } from "@/lib/v2-live-data";

interface MoleculeViewerProps {
  compoundName: string;
  atoms: VizAtom[];
  bonds: VizBond[];
  source: EvidenceSource;
  message: string | null;
}

type ViewerApi = {
  clear: () => void;
  addModel: (data: string, format: string) => void;
  setStyle: (selection: Record<string, never>, style: Record<string, unknown>) => void;
  zoomTo: () => void;
  render: () => void;
};

export function MoleculeViewer({
  compoundName,
  atoms,
  bonds,
  source,
  message,
}: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ViewerApi | null>(null);
  const [viewerError, setViewerError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function renderMolecule() {
      if (!containerRef.current || atoms.length === 0) return;

      try {
        const threeDmol = await import("3dmol");
        if (cancelled || !containerRef.current) return;

        const api = threeDmol as unknown as {
          createViewer: (
            element: HTMLDivElement,
            config: Record<string, unknown>,
          ) => ViewerApi;
        };
        const viewer =
          viewerRef.current ??
          api.createViewer(containerRef.current, {
            backgroundColor: "#080b18",
          });

        viewerRef.current = viewer;
        viewer.clear();
        viewer.addModel(toMolBlock(compoundName, atoms, bonds), "sdf");
        viewer.setStyle({}, { stick: { radius: 0.18 }, sphere: { scale: 0.28 } });
        viewer.zoomTo();
        viewer.render();
        setViewerError(null);
      } catch (error) {
        setViewerError(
          error instanceof Error ? error.message : "Unable to load 3D molecule viewer.",
        );
      }
    }

    void renderMolecule();

    return () => {
      cancelled = true;
    };
  }, [atoms, bonds, compoundName]);

  return (
    <div className="space-y-3">
      <div className="relative min-h-[360px] overflow-hidden rounded-lg border border-outline-variant/25 bg-surface-container-lowest">
        {atoms.length > 0 ? (
          <div
            ref={containerRef}
            className="absolute inset-0"
            aria-label={`${compoundName} 3D molecule viewer`}
          />
        ) : (
          <FallbackMolecule compoundName={compoundName} />
        )}
      </div>
      <div className="rounded-lg border border-outline-variant/25 bg-surface-container-high/70 p-3 text-xs leading-relaxed text-on-surface-variant">
        <span className="font-semibold text-on-surface">
          Molecule source: {source}.
        </span>{" "}
        {viewerError ?? message ?? "Live molecule coordinates rendered in 3Dmol."}
      </div>
    </div>
  );
}

function FallbackMolecule({ compoundName }: { compoundName: string }) {
  return (
    <>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_35%_30%,rgba(123,208,255,0.16),transparent_30%),radial-gradient(circle_at_70%_65%,rgba(60,221,199,0.12),transparent_28%)]" />
      <Bond x1="23%" y1="52%" x2="36%" y2="60%" />
      <Bond x1="36%" y1="60%" x2="49%" y2="48%" />
      <Bond x1="49%" y1="48%" x2="62%" y2="57%" />
      <Atom label={compoundName.slice(0, 1)} x="23%" y="52%" color="bg-primary" />
      <Atom label="C" x="36%" y="60%" color="bg-tertiary" />
      <Atom label="O" x="49%" y="48%" color="bg-error" />
      <Atom label="N" x="62%" y="57%" color="bg-secondary" />
    </>
  );
}

function Atom({
  label,
  x,
  y,
  color,
}: {
  label: string;
  x: string;
  y: string;
  color: string;
}) {
  return (
    <span
      className={`absolute flex h-12 w-12 items-center justify-center rounded-full ${color} font-mono text-sm font-bold text-background shadow-glow`}
      style={{ left: x, top: y }}
    >
      {label}
    </span>
  );
}

function Bond({
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
      className="absolute h-1 origin-left rounded-full bg-outline/60"
      style={{
        left: x1,
        top: y1,
        width: `calc(${x2} - ${x1})`,
        transform: "rotate(24deg)",
      }}
    />
  );
}

function toMolBlock(compoundName: string, atoms: VizAtom[], bonds: VizBond[]) {
  const header = [
    compoundName.slice(0, 80),
    "  Hybrid QML-KG",
    "",
    `${padCount(atoms.length)}${padCount(bonds.length)}  0  0  0  0            999 V2000`,
  ];
  const atomLines = atoms.map((atom) => {
    const element = atom.element.slice(0, 3).padEnd(3, " ");
    return `${padCoord(atom.x)}${padCoord(atom.y)}${padCoord(atom.z)} ${element} 0  0  0  0  0  0  0  0  0  0  0  0`;
  });
  const bondLines = bonds.map((bond) => {
    const source = padCount(bond.source + 1);
    const target = padCount(bond.target + 1);
    const order = padCount(Math.max(1, Math.min(3, bond.order)));
    return `${source}${target}${order}  0  0  0  0`;
  });

  return [...header, ...atomLines, ...bondLines, "M  END", "$$$$"].join("\n");
}

function padCount(value: number) {
  return String(value).padStart(3, " ");
}

function padCoord(value: number) {
  return value.toFixed(4).padStart(10, " ");
}
