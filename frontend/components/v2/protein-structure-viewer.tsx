"use client";

import { useEffect, useRef, useState } from "react";
import type { RepurposingProteinStructureEvidence } from "@/lib/api";
import { getStructureArtifactUrl } from "@/lib/api";

type ViewerApi = {
  clear: () => void;
  addModel: (data: string, format: string) => void;
  setStyle: (selection: Record<string, unknown>, style: Record<string, unknown>) => void;
  zoomTo: () => void;
  render: () => void;
};

export function ProteinStructureViewer({
  protein,
}: {
  protein: RepurposingProteinStructureEvidence | null;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ViewerApi | null>(null);
  const [status, setStatus] = useState<string>("Loading local structure artifact.");

  useEffect(() => {
    let cancelled = false;

    async function renderProtein() {
      if (!containerRef.current || !protein?.viewer.supports_3d || !protein.viewer.artifact_path) {
        setStatus("No local 3D structure artifact is available for this target yet.");
        return;
      }

      try {
        const [threeDmol, response] = await Promise.all([
          import("3dmol"),
          fetch(getStructureArtifactUrl(protein.viewer.artifact_path)),
        ]);
        if (!response.ok) throw new Error(`Structure artifact request failed: ${response.status}`);
        const pdb = await response.text();
        if (cancelled || !containerRef.current) return;

        const api = threeDmol as unknown as {
          createViewer: (element: HTMLDivElement, config: Record<string, unknown>) => ViewerApi;
        };
        const viewer =
          viewerRef.current ??
          api.createViewer(containerRef.current, {
            backgroundColor: "#080b18",
          });

        viewerRef.current = viewer;
        viewer.clear();
        viewer.addModel(pdb, protein.viewer.artifact_format ?? "pdb");
        viewer.setStyle({}, { cartoon: { color: "spectrum" } });
        viewer.zoomTo();
        viewer.render();
        setStatus("Local PDB artifact rendered with 3Dmol.");
      } catch (error) {
        setStatus(error instanceof Error ? error.message : "Unable to render local protein structure.");
      }
    }

    void renderProtein();

    return () => {
      cancelled = true;
    };
  }, [protein]);

  return (
    <div className="space-y-2">
      <div className="relative min-h-[280px] overflow-hidden rounded-lg border border-outline-variant/25 bg-surface-container-lowest">
        {protein?.viewer.supports_3d ? (
          <div
            ref={containerRef}
            className="absolute inset-0"
            aria-label={`${protein.display_name} local protein structure viewer`}
          />
        ) : (
          <div className="flex h-[280px] items-center justify-center p-6 text-center text-sm text-on-surface-variant">
            No local structure artifact available for this mapped target.
          </div>
        )}
      </div>
      <p className="text-xs leading-relaxed text-on-surface-variant">
        <span className="font-semibold text-on-surface">{protein?.display_name ?? "Protein structure"}:</span> {status}
      </p>
    </div>
  );
}
