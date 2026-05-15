"use client";

import { useEffect, useState } from "react";
import type { StatusResponse } from "@/lib/api";
import { fetchStatus } from "@/lib/api";
import { ApiRecoveryCard } from "@/components/api-recovery-card";

export function SystemStatusPanel() {
  const [data, setData] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await fetchStatus();
        if (!cancelled) {
          setData(s);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Request failed");
          setData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <p className="text-sm text-on-surface-variant" role="status">
        Loading status…
      </p>
    );
  }

  if (error) {
    return (
      <ApiRecoveryCard
        title="Could not reach API"
        error={error}
        details="Use System status to verify readiness, then continue in New run."
      />
    );
  }

  if (!data) return null;

  return (
    <dl className="grid gap-4 sm:grid-cols-2">
      <StatusItem label="API status" value={data.status} />
      <StatusItem
        label="Orchestrator"
        value={data.orchestrator_ready ? "Ready" : "Not ready"}
      />
      <StatusItem
        label="Classical model"
        value={data.classical_model_loaded ? "Loaded" : "Not loaded"}
      />
      <StatusItem
        label="Quantum model"
        value={data.quantum_model_loaded ? "Loaded" : "Not loaded"}
      />
      <StatusItem
        label="Entity count"
        value={data.entity_count.toLocaleString()}
      />
      <StatusItem
        label="Relations"
        value={data.supported_relations.join(", ")}
      />
    </dl>
  );
}

function StatusItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-surface-container-lowest/80 px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </dt>
      <dd className="mt-1 font-mono text-sm text-tertiary">{value}</dd>
    </div>
  );
}
