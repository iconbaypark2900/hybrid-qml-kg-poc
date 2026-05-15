import { EvidenceProvenanceSummary } from "@/components/v2/evidence-provenance";
import { Card, StatusBadge } from "@/components/v2/v2-shell";
import type { EvidenceState, V2MethodRow } from "@/lib/v2-live-data";

export function ModelLeaderboard({
  methods,
}: {
  methods: EvidenceState<V2MethodRow[]>;
}) {
  return (
    <Card
      title="Model leaderboard"
      kicker={methods.source === "live" ? "Latest run artifact" : "Fallback benchmark"}
      help="Compares model families for this investigation. Live rows come from run artifacts; fallback rows use paper-aligned defaults."
    >
      <EvidenceProvenanceSummary
        title="Leaderboard source"
        provenance={methods.provenance}
      />
      {methods.message ? (
        <p className="mt-3 text-xs leading-relaxed text-on-surface-variant">
          {methods.message}
        </p>
      ) : null}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[680px] text-sm">
          <thead>
            <tr className="border-b border-outline-variant/30 text-left font-label text-xs uppercase tracking-widest text-on-surface-variant">
              <th className="py-3 pr-4">Method</th>
              <th className="py-3 pr-4">Type</th>
              <th className="py-3 pr-4">PR-AUC</th>
              <th className="py-3 pr-4">Meaning</th>
            </tr>
          </thead>
          <tbody>
            {methods.data.map((row) => (
              <tr
                key={row.method}
                className="border-b border-outline-variant/20 last:border-b-0"
              >
                <td className="py-3 pr-4 font-semibold text-on-surface">
                  {row.method}
                </td>
                <td className="py-3 pr-4">
                  <StatusBadge tone={toneForMethod(row.type)}>{row.type}</StatusBadge>
                </td>
                <td className="py-3 pr-4 font-mono text-primary">{row.prAuc}</td>
                <td className="py-3 pr-4 text-on-surface-variant">
                  {row.meaning}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function toneForMethod(type: V2MethodRow["type"]) {
  if (type === "Quantum") return "quantum";
  if (type === "Hybrid") return "warning";
  return "success";
}
