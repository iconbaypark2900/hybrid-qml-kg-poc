import { Card, Metric, PageHero, StatusBadge } from "@/components/v2/v2-shell";
import { fetchOpsErrors, fetchOpsHealth, getApiBaseUrl } from "@/lib/api";

export default async function V2SystemPage() {
  const [health, errors] = await Promise.all([
    fetchOpsHealth().catch((error) => ({
      status: "error",
      uptime_seconds: 0,
      request_counts: {},
      request_failures: {},
      recent_failure_count: 0,
      orchestrator_ready: false,
      classical_model_loaded: false,
      quantum_model_loaded: false,
      cors_allowed_origins: [],
      internal_auth_enabled: false,
      environment: process.env.NEXT_PUBLIC_ENVIRONMENT ?? "development",
      error: error instanceof Error ? error.message : "Unable to load ops health.",
    })),
    fetchOpsErrors(8).catch(() => []),
  ]);

  return (
    <div className="space-y-6">
      <PageHero
        eyebrow="Ops"
        title="Monitor local v2 reliability and deployment readiness."
      >
        Local operational visibility for development and staging checks. External
        observability systems can replace this later.
      </PageHero>

      <section className="grid gap-4 md:grid-cols-4">
        <Metric
          label="Environment"
          value={health.environment}
          detail="NEXT_PUBLIC_ENVIRONMENT"
          help="Deployment label used by the frontend and ops panel."
        />
        <Metric
          label="API base"
          value={getApiBaseUrl()}
          detail="Browser API route"
          help="Browser calls use the governed proxy unless explicitly disabled."
        />
        <Metric
          label="Ops health"
          value={health.status}
          detail={`${Math.round(health.uptime_seconds)}s uptime`}
          help="Backend operational readiness summary."
        />
        <Metric
          label="Failures"
          value={String(health.recent_failure_count)}
          detail="Recent backend failures"
          help="In-memory counter; resets when the API process restarts."
        />
      </section>

      <section className="grid gap-4 xl:grid-cols-[.8fr_1.2fr]">
        <Card
          title="Backend readiness"
          kicker="FastAPI"
          help="Readiness and governance mode reported by /ops/health."
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <ReadinessRow label="Orchestrator" ready={health.orchestrator_ready} />
            <ReadinessRow label="Classical model" ready={health.classical_model_loaded} />
            <ReadinessRow label="Quantum model" ready={health.quantum_model_loaded} />
            <ReadinessRow label="Internal auth" ready={health.internal_auth_enabled} />
          </div>
        </Card>

        <Card
          title="Recent failures"
          kicker="Local counters"
          help="Recent backend failures captured by the FastAPI observability middleware."
        >
          {errors.length === 0 ? (
            <p className="text-sm text-on-surface-variant">No recent backend failures recorded.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[680px] text-sm">
                <thead>
                  <tr className="border-b border-outline-variant/30 text-left font-label text-xs uppercase tracking-widest text-on-surface-variant">
                    <th className="py-3 pr-4">Request</th>
                    <th className="py-3 pr-4">Status</th>
                    <th className="py-3 pr-4">Duration</th>
                    <th className="py-3 pr-4">ID</th>
                  </tr>
                </thead>
                <tbody>
                  {errors.map((item) => (
                    <tr key={item.request_id} className="border-b border-outline-variant/20">
                      <td className="py-3 pr-4 font-mono text-on-surface">
                        {item.method} {item.path}
                      </td>
                      <td className="py-3 pr-4 text-error">{item.status_code}</td>
                      <td className="py-3 pr-4">{item.duration_ms}ms</td>
                      <td className="py-3 pr-4 font-mono text-xs text-on-surface-variant">
                        {item.request_id}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </section>
    </div>
  );
}

function ReadinessRow({ label, ready }: { label: string; ready: boolean }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-outline-variant/25 bg-surface-container-high/60 p-3">
      <span className="text-sm text-on-surface">{label}</span>
      <StatusBadge tone={ready ? "success" : "warning"}>
        {ready ? "ready" : "not ready"}
      </StatusBadge>
    </div>
  );
}
