'use client';

import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useInvestigationStore } from '@/lib/store';
import { useManifest } from '@/lib/use-manifest';
import { Skeleton } from '@/components/shared/Skeleton';

// 3D KG renderer is browser-only (uses three + 3d-force-graph) — load
// dynamically with ssr: false so the build doesn't try to render it on the server.
const KGForceGraph = dynamic(() => import('./KGForceGraph').then((m) => m.KGForceGraph), {
  ssr: false,
  loading: () => (
    <div style={{ height: 480, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Skeleton.Block height={400} style={{ width: '100%' }} />
    </div>
  ),
});

const MoleculeViewer = dynamic(() => import('./MoleculeViewer').then((m) => m.MoleculeViewer), {
  ssr: false,
  loading: () => <Skeleton.Block height={300} />,
});

export function VisualizePage() {
  const compound = useInvestigationStore((s) => s.compound);
  const disease = useInvestigationStore((s) => s.disease);
  const { state: manifest } = useManifest();

  return (
    <>
      <div className="page-hero">
        <div>
          <div className="step">04 · VISUALIZE</div>
          <h1 className="h1">Inspect the candidate visually</h1>
          <p className="lede">
            The 3D knowledge graph below shows the active model&apos;s entity
            embeddings reduced to 3D, with the chosen compound and disease
            highlighted. The molecule viewer to the right is a placeholder —
            wire it to a real PDB/PubChem feed when partner data sources are
            connected.
          </p>
        </div>
        <span className="pill">{manifest.phase === 'ready' ? '● live' : '○ unconfigured'}</span>
      </div>

      <div className="grid-7-5">
        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">TOOL · KG SUBGRAPH</div>
              <div className="panel-title">3D entity neighborhood</div>
            </div>
            <span className="badge">Three.js</span>
          </div>
          <p className="panel-purpose">
            Each node is an entity from the active embedding manifest. Edges
            indicate close cosine similarity in the embedding space — proxy for
            &ldquo;the model thinks these are related.&rdquo; Highlighted nodes
            are <strong>{compound.name}</strong> and <strong>{disease.name}</strong>.
          </p>
          <KGForceGraph
            highlightDrugId={compound.id}
            highlightDiseaseId={disease.id}
            manifestState={manifest}
          />
        </div>

        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">TOOL · STRUCTURE</div>
              <div className="panel-title">{compound.name}</div>
            </div>
            <span className="badge">Placeholder</span>
          </div>
          <p className="panel-purpose">
            DrugBank ID <code>{compound.id}</code> · {compound.cat}.
            A real molecule render needs a SMILES/PDB feed; this placeholder
            shows the data shape.
          </p>
          <MoleculeViewer compound={compound} />
        </div>
      </div>

      <div className="footer-actions">
        <Link href="/validate" className="btn">⌥ Back to Validate</Link>
        <Link href="/initialize" className="btn">⌥ Reset investigation</Link>
      </div>
    </>
  );
}
