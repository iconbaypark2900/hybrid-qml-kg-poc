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
          >
            {HETIONET_DISEASES.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
          <div className="field-hint">Primary clinical context</div>
        </div>

        <div>
          <label className="field-label" data-help>
            COMPOUND
            <HelpHint text="The compound is identified by its <strong>DrugBank ID</strong> (e.g. <code>DB12015</code> · Inaxaplin). Filter by therapeutic class via the dropdown options." />
          </label>
          <select
            className="select"
            value={compound.id}
            onChange={(e) => {
              const c = HETIONET_COMPOUNDS.find((x) => x.id === e.target.value);
              if (c) setCompound(c);
            }}
          >
            {HETIONET_COMPOUNDS.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} · {c.cat}
              </option>
            ))}
          </select>
          <div className="field-hint">Drug under repurposing review</div>
        </div>

        <div>
          <label className="field-label" data-help>
            ANCHOR TARGET (GENE)
            <HelpHint text="The <strong>anchor target</strong> is the gene whose protein the candidate acts through. Picking an ancestry-relevant gene (e.g. APOL1 G1/G2 for AA-specific kidney disease) is what makes the analysis equity-aware." />
          </label>
          <select
            className="select"
            value={gene.id}
            onChange={(e) => {
              const g = HETIONET_GENES.find((x) => x.id === e.target.value);
              if (g) setGene(g);
            }}
          >
            {HETIONET_GENES.map((g) => (
              <option key={g.id} value={g.id}>
                {g.symbol} — {g.name.length > 40 ? g.name.slice(0, 38) + '…' : g.name}
              </option>
            ))}
          </select>
          <div className="field-hint">Mechanistic anchor for ancestry-aware filtering</div>
        </div>

        <div>
          <label className="field-label" data-help>
            HETIONET METAEDGE
            <HelpHint text="A <strong>Hetionet metaedge</strong> is a typed relationship — e.g. <code>CtD</code> = Compound–treats–Disease (755 edges in v1.0). Picking the metaedge tells the model which evidence layer to score against." />
          </label>
          <select
            className="select"
            value={metaedge.code}
            onChange={(e) => {
              const m = HETIONET_METAEDGES.find((x) => x.code === e.target.value);
              if (m) setMetaedge(m);
            }}
          >
            {HETIONET_METAEDGES.map((m) => (
              <option key={m.code} value={m.code}>
                {m.code} · {m.name} ({m.count.toLocaleString()})
              </option>
            ))}
          </select>
          <div className="field-hint">Edge type for prediction</div>
        </div>
      </div>

      <div className="dataset-stats">
        <div className="dataset-stat">
          <div className="dataset-stat-num">{HETIONET_TOTALS.diseases}</div>
          <div className="dataset-stat-label">DISEASES</div>
        </div>
        <div className="dataset-stat">
          <div className="dataset-stat-num">{HETIONET_TOTALS.compounds.toLocaleString()}</div>
          <div className="dataset-stat-label">COMPOUNDS</div>
        </div>
        <div className="dataset-stat">
          <div className="dataset-stat-num">{HETIONET_TOTALS.genes.toLocaleString()}</div>
          <div className="dataset-stat-label">GENES</div>
        </div>
        <div className="dataset-stat">
          <div className="dataset-stat-num">{HETIONET_TOTALS.metaedges}</div>
          <div className="dataset-stat-label">METAEDGES</div>
        </div>
      </div>

      <div className="panel-footer">
        <span>investigation_contract :: draft · hetionet-v1.0</span>
        <span><em>unsaved</em></span>
      </div>
    </div>
  );
}
