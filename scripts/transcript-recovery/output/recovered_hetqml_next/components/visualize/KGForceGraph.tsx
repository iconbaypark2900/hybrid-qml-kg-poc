'use client';

// Minimal 3D force graph using 3d-force-graph + three. Synthesizes a small
// neighborhood graph from the curated catalog data so the page demonstrates
// the visualization shape end-to-end. When the backend exposes a
// /manifest/embedding/<id>/neighbors endpoint, swap this for a real fetch.

import { useEffect, useRef } from 'react';
import { HETIONET_COMPOUNDS } from '@/data/compounds';
import { HETIONET_DISEASES } from '@/data/diseases';
import { HETIONET_GENES } from '@/data/genes';
import type { ManifestState } from '@/lib/use-manifest';

interface NodeT {
  id: string;
  label: string;
  group: 'compound' | 'disease' | 'gene';
  highlighted?: boolean;
}
interface LinkT { source: string; target: string; }

export function KGForceGraph({
  highlightDrugId, highlightDiseaseId, manifestState,
}: {
  highlightDrugId: string;
  highlightDiseaseId: string;
  manifestState: ManifestState;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!ref.current) return;
    let cancelled = false;

    (async () => {
      const ForceGraph3D = (await import('3d-force-graph')).default;
      if (cancelled || !ref.current) return;

      const nodes: NodeT[] = [];
      const links: LinkT[] = [];

      const drug = HETIONET_COMPOUNDS.slice(0, 12);
      const disease = HETIONET_DISEASES.slice(0, 12);
      const gene = HETIONET_GENES.slice(0, 18);
      drug.forEach((c) => nodes.push({
        id: c.id, label: c.name, group: 'compound',
        highlighted: c.id === highlightDrugId,
      }));
      disease.forEach((d) => nodes.push({
        id: d.id, label: d.name, group: 'disease',
        highlighted: d.id === highlightDiseaseId,
      }));
      gene.forEach((g) => nodes.push({
        id: g.id, label: g.symbol, group: 'gene',
      }));

      // Synthesize edges: each gene links to ~2 drugs and ~2 diseases.
      gene.forEach((g, i) => {
        links.push({ source: g.id, target: drug[i % drug.length]!.id });
        links.push({ source: g.id, target: drug[(i + 3) % drug.length]!.id });
        links.push({ source: g.id, target: disease[i % disease.length]!.id });
        links.push({ source: g.id, target: disease[(i + 5) % disease.length]!.id });
      });
      // Direct compound-disease link if both are in the highlight selection.
      if (
        nodes.find((n) => n.id === highlightDrugId) &&
        nodes.find((n) => n.id === highlightDiseaseId)
      ) {
        links.push({ source: highlightDrugId, target: highlightDiseaseId });
      }

      const colorByGroup = (n: NodeT): string =>
        n.highlighted
          ? '#ffd166'
          : n.group === 'compound'
            ? '#06d6a0'
            : n.group === 'disease'
              ? '#ef476f'
              : '#118ab2';

      const graph = ForceGraph3D()(ref.current as HTMLElement)
        .width(ref.current.clientWidth)
        .height(440)
        .backgroundColor('#0a0a0c')
        .graphData({ nodes, links })
        .nodeLabel('label')
        .nodeColor((n: any) => colorByGroup(n as NodeT))
        .nodeVal((n: any) => ((n as NodeT).highlighted ? 18 : 6))
        .linkColor(() => 'rgba(255,255,255,0.18)')
        .linkOpacity(0.4)
        .linkDirectionalParticles(0)
        .cooldownTicks(80);

      graphRef.current = graph;
    })();

    return () => {
      cancelled = true;
      if (graphRef.current && typeof graphRef.current._destructor === 'function') {
        graphRef.current._destructor();
      }
      graphRef.current = null;
    };
  }, [highlightDrugId, highlightDiseaseId]);

  return (
    <>
      <div
        ref={ref}
        role="img"
        aria-label={
          `3D knowledge graph showing ${highlightDrugId} (highlighted) and ` +
          `${highlightDiseaseId} (highlighted) embedded with their neighbors`
        }
        style={{
          width: '100%', height: 440,
          background: '#0a0a0c',
          borderRadius: 4,
          overflow: 'hidden',
        }}
      />
      <div style={{ display: 'flex', gap: 16, marginTop: 8, fontSize: 11, color: 'var(--muted)' }}>
        <Legend color="#ffd166" label="highlighted" />
        <Legend color="#06d6a0" label="compound" />
        <Legend color="#ef476f" label="disease" />
        <Legend color="#118ab2" label="gene" />
      </div>
      {manifestState.phase !== 'ready' && (
        <div style={{ marginTop: 8, fontSize: 10, color: 'var(--faint)' }}>
          (using synthetic neighbors — connect the API for real embedding-space neighbors)
        </div>
      )}
    </>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'inline-flex', gap: 6, alignItems: 'center', fontFamily: 'monospace' }}>
      <span style={{ width: 10, height: 10, borderRadius: '50%', background: color }} />
      {label}
    </span>
  );
}
