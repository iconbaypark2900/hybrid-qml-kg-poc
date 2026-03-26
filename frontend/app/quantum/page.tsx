"use client";

import { useEffect, useState } from "react";
import type { QuantumConfigResponse } from "@/lib/api";
import { fetchQuantumConfig, getApiBaseUrl } from "@/lib/api";

export default function QuantumPage() {
  const [data, setData] = useState<QuantumConfigResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await fetchQuantumConfig();
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

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          Quantum logic
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Quantum execution configuration and model status.
        </p>
      </header>

      {loading ? (
        <p className="text-sm text-on-surface-variant" role="status">
          Resolving probability&hellip;
        </p>
      ) : error ? (
        <div className="rounded-lg border border-error/40 bg-error-container/20 p-4">
          <p className="text-sm font-medium text-error">
            Could not load quantum config
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">{error}</p>
          <p className="mt-3 text-xs text-on-surface-variant">
            Base URL:{" "}
            <code className="text-on-surface">{getApiBaseUrl()}</code>
          </p>
        </div>
      ) : data ? (
        <div className="space-y-6">
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Execution mode" value={data.execution_mode ?? "—"} />
            <Stat label="Backend" value={data.backend ?? "—"} />
            <Stat
              label="Shots"
              value={data.shots != null ? String(data.shots) : "—"}
            />
            <Stat
              label="Quantum model"
              value={data.quantum_model_loaded ? "Loaded" : "Not loaded"}
              highlight={data.quantum_model_loaded}
            />
          </dl>

          {data.config ? (
            <div>
              <h2 className="mb-2 font-headline text-lg font-semibold text-on-surface">
                Raw configuration
              </h2>
              <pre className="overflow-x-auto rounded-lg bg-surface-container-lowest p-4 text-xs text-on-surface">
                {JSON.stringify(data.config, null, 2)}
              </pre>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Stat({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="rounded-lg bg-surface-container-lowest/80 px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </dt>
      <dd
        className={`mt-1 font-mono text-sm ${highlight ? "text-tertiary" : "text-on-surface"}`}
      >
        {value}
      </dd>
    </div>
  );
}
