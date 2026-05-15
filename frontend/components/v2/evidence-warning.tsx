export function EvidenceWarningBanner({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;

  return (
    <div className="rounded-lg border border-[#f8c64f]/35 bg-[#f8c64f]/10 p-4">
      <p className="font-label text-xs font-bold uppercase tracking-widest text-[#f8c64f]">
        Evidence warnings
      </p>
      <ul className="mt-3 list-disc space-y-1 pl-4 text-xs leading-relaxed text-on-surface-variant">
        {warnings.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>
    </div>
  );
}
