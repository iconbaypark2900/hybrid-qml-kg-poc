'use client';

import { useInvestigationStore } from '@/lib/store';
import { HETIONET_DISEASES } from '@/data/diseases';
import { HETIONET_COMPOUNDS } from '@/data/compounds';
import { HETIONET_GENES } from '@/data/genes';
import { HETIONET_METAEDGES, HETIONET_TOTALS } from '@/data/metaedges';
import { HelpHint } from '@/components/shared/HelpHint';

export function InvestigationParameters() {
  const { compound, disease, gene, metaedge, setCompound, setDisease, setGene, setMetaedge } =
    useInvestigationStore();

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · INVESTIGATION PARAMETERS</div>
          <div className="panel-title">What you&apos;re investigating</div>
        </div>
        <span className="badge">Form</span>
      </div>
      <p className="panel-purpose">
        The disease, compound, anchor target, and Hetionet relation. These four selections define
        the prediction task and what evidence will count.
      </p>

      <div className="field-grid">
        <div>
          <label className="field-label" data-help>
            DISEASE
            <HelpHint text="<strong>DOID</strong> = Disease Ontology ID. Choosing a parent disease (e.g. Diabetes mellitus) is broader than a child (e.g. T2DM) — the broader the disease, the noisier the model signal." />
          </label>
          <select
            className="select"
            value={disease.id}
            onChange={(e) => {
              const d = HETIONET_DISEASES.find((x) => x.id === e.target.value);
              if (d) setDisease(d);
            }}