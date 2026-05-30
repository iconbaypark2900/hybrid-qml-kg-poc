import type { Metadata } from 'next';
import { PageStub } from '@/components/shared/PageStub';

export const metadata: Metadata = { title: 'Visualize · Hetionet QML' };

export default function Page() {
  return (
    <PageStub
      step="04 · VISUALIZE"
      title="Inspect every layer of evidence"
      lede="Six panels — molecule, knowledge graph, embedding projection, quantum kernel circuit, evidence overlays, interpretation. Each cites a different source. Cross-panel reinforcement is what turns a model score into something defensible."
      todos={[
        '3D molecule viewer — 3Dmol.js (use dynamic import with ssr:false; PubChem 3D conformer fetch)',
        '3D Knowledge Graph — custom Three.js force-directed layout (port renderKG from HTML)',
        '3D UMAP scatter — Three.js with OrbitControls (port renderUMAP from HTML)',
        'Quantum kernel circuit — D3-rendered ZZFeatureMap',
        'Clinical Support Strip (4 cards)',
        'Evidence Strength Matrix (6 layers × 5 status columns)',
        'Path Diagram (Compound → Target → Pathway → Disease)',
        'Model Agreement (Classical / Hybrid / Quantum bars + agreement gauge)',
        'Provenance Timeline · Quality Overlay flags',
        'Evidence Overlays + Interpretation Panel'
      ]}
      nextHref="/validate"
      nextLabel="Return to Validate"
    />
  );
}
