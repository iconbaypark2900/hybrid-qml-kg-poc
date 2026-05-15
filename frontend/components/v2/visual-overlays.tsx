import type { VisualEvidenceOverlay } from "@/lib/v2-quality-data";

export function VisualOverlays({ overlays }: { overlays: VisualEvidenceOverlay[] }) {
  if (overlays.length === 0) return null;

  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {overlays.map((overlay) => (
        <div
          key={`${overlay.label}-${overlay.value}`}
          className="rounded-lg border border-outline-variant/25 bg-surface-container-high/70 p-3"
        >
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-widest text-on-surface-variant">
              {overlay.label}
            </p>
            <span className={`rounded-full px-2 py-1 text-xs font-bold ${toneFor(overlay.status)}`}>
              {overlay.status.replace("_", " ")}
            </span>
          </div>
          <p className="mt-2 font-mono text-sm text-primary">{overlay.value}</p>
          <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">
            {overlay.explanation}
          </p>
        </div>
      ))}
    </div>
  );
}

function toneFor(status: VisualEvidenceOverlay["status"]) {
  if (status === "live") return "bg-tertiary/15 text-tertiary";
  if (status === "fallback") return "bg-secondary/15 text-secondary";
  return "bg-[#f8c64f]/15 text-[#f8c64f]";
}
