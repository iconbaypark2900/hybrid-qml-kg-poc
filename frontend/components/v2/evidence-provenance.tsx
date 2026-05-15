import type { CandidateEvidenceLink, EvidenceProvenance } from "@/lib/api";

export function EvidenceProvenanceSummary({
  title = "Source trail",
  provenance,
}: {
  title?: string;
  provenance: EvidenceProvenance[];
}) {
  if (provenance.length === 0) {
    return (
      <p className="rounded-lg border border-outline-variant/25 bg-surface-container-high/70 p-3 text-xs text-on-surface-variant">
        <span className="font-semibold text-on-surface">{title}:</span> not recorded.
      </p>
    );
  }

  const primary = provenance[0];
  return (
    <details className="rounded-lg border border-outline-variant/25 bg-surface-container-high/70 p-3 text-xs text-on-surface-variant">
      <summary className="cursor-pointer font-semibold text-on-surface">
        {title}: {primary.source_kind}
        {primary.artifact_name ? ` / ${primary.artifact_name}` : ""}
        {primary.run_timestamp ? ` / ${primary.run_timestamp}` : ""}
      </summary>
      <div className="mt-3 space-y-3">
        {provenance.map((item, index) => (
          <div key={`${item.endpoint}-${item.artifact_name ?? index}`} className="space-y-1">
            <p>
              <span className="text-on-surface">Endpoint:</span> {item.endpoint}
            </p>
            <p>
              <span className="text-on-surface">Kind:</span> {item.source_kind}
            </p>
            {item.relation ? (
              <p>
                <span className="text-on-surface">Relation:</span> {item.relation}
              </p>
            ) : null}
            {item.artifact_name ? (
              <p>
                <span className="text-on-surface">Artifact:</span> {item.artifact_name}
              </p>
            ) : null}
            {item.embedding_method || item.embedding_dim || item.seed != null ? (
              <p>
                <span className="text-on-surface">Config:</span>{" "}
                {[
                  item.embedding_method,
                  item.embedding_dim ? `${item.embedding_dim}D` : null,
                  item.seed != null ? `seed ${item.seed}` : null,
                ]
                  .filter(Boolean)
                  .join(", ")}
              </p>
            ) : null}
            {item.notes?.length ? (
              <ul className="list-disc pl-4">
                {item.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ))}
      </div>
    </details>
  );
}

export function CandidateEvidenceLinks({
  links,
}: {
  links: CandidateEvidenceLink[];
}) {
  if (links.length === 0) return null;

  return (
    <div className="space-y-3">
      {links.map((link) => (
        <div
          key={`${link.kind}-${link.label}`}
          className="rounded-lg border border-outline-variant/25 bg-surface-container-high/60 p-3"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="font-semibold text-on-surface">{link.label}</p>
            <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-1 text-xs font-semibold text-primary">
              {link.kind}
            </span>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">
            {link.source}
          </p>
          <div className="mt-2">
            <EvidenceProvenanceSummary
              title="Evidence source"
              provenance={link.provenance ? [link.provenance] : []}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
