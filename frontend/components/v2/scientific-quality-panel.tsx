import { EvidenceWarningBanner } from "@/components/v2/evidence-warning";
import type { QualityControlItem, ScientificQualityEvidence } from "@/lib/v2-quality-data";

export function ScientificQualityPanel({
  quality,
}: {
  quality: ScientificQualityEvidence;
}) {
  const items = [
    quality.multiSeed,
    quality.confidenceInterval,
    quality.variance,
    quality.calibrationCurve,
    quality.randomBaseline,
    quality.degreeBaseline,
    quality.uncertainty,
  ];

  return (
    <div className="space-y-4">
      <EvidenceWarningBanner warnings={quality.warnings} />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <QualityCard key={item.key} item={item} />
        ))}
      </div>
    </div>
  );
}

function QualityCard({ item }: { item: QualityControlItem }) {
  return (
    <div className="rounded-lg border border-outline-variant/25 bg-surface-container-high/60 p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="font-semibold text-on-surface">{item.label}</p>
        <span className={`rounded-full px-2 py-1 text-xs font-bold ${toneFor(item.status)}`}>
          {item.status.replace("_", " ")}
        </span>
      </div>
      {item.value ? (
        <p className="mt-3 font-mono text-sm text-primary">{item.value}</p>
      ) : null}
      <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">
        {item.detail}
      </p>
      {item.nextAction ? (
        <p className="mt-2 text-xs leading-relaxed text-[#f8c64f]">
          Next: {item.nextAction}
        </p>
      ) : null}
    </div>
  );
}

function toneFor(status: QualityControlItem["status"]) {
  if (status === "recorded") return "bg-tertiary/15 text-tertiary";
  if (status === "artifact_only") return "bg-primary/15 text-primary";
  if (status === "fallback") return "bg-secondary/15 text-secondary";
  return "bg-[#f8c64f]/15 text-[#f8c64f]";
}
