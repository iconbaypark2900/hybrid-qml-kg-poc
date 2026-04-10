"use client";

import { useEffect, useState } from "react";
import type { QuantumConfigResponse } from "@/lib/api";
import {
  fetchQuantumConfig,
  getApiBaseUrl,
  verifyQuantumRuntime,
} from "@/lib/api";

export default function QuantumPage() {
  const [data, setData] = useState<QuantumConfigResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [apiToken, setApiToken] = useState("");
  const [instanceCrn, setInstanceCrn] = useState("");
  const [verifyLoading, setVerifyLoading] = useState(false);
  const [verifyResult, setVerifyResult] = useState<string | null>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);

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

  useEffect(() => {
    return () => {
      setApiToken("");
      setInstanceCrn("");
    };
  }, []);

  async function onVerify(e: React.FormEvent) {
    e.preventDefault();
    setVerifyError(null);
    setVerifyResult(null);
    const trimmed = apiToken.trim();
    if (!trimmed) {
      setVerifyError("Enter an API token.");
      return;
    }
    setVerifyLoading(true);
    try {
      const r = await verifyQuantumRuntime({
        api_token: trimmed,
        instance_crn: instanceCrn.trim() || undefined,
      });
      if (r.status === "ok") {
        setVerifyResult(
          `${r.message} Backends visible: ${r.backend_count} (${r.simulator_count} simulators). ` +
            `Hardware sample: ${r.hardware_backend_names.slice(0, 5).join(", ") || "—"}` +
            (r.instances_count != null ? `. Instances: ${r.instances_count}.` : ""),
        );
        setApiToken("");
        setInstanceCrn("");
      } else {
        setVerifyError(r.message || "Connection failed.");
      }
    } catch (err) {
      setVerifyError(
        err instanceof Error ? err.message : "Request failed. Use HTTPS in production.",
      );
    } finally {
      setVerifyLoading(false);
    }
  }

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

      <section
        className="rounded-lg border border-outline-variant/40 bg-surface-container-lowest/50 p-4"
        aria-labelledby="byok-heading"
      >
        <h2
          id="byok-heading"
          className="font-headline text-lg font-semibold text-on-surface"
        >
          Test your IBM Quantum account
        </h2>
        <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">
          Enter your API token from{" "}
          <a
            href="https://quantum.ibm.com/"
            className="text-primary underline underline-offset-2"
            target="_blank"
            rel="noreferrer"
          >
            IBM Quantum Platform
          </a>{" "}
          and optionally your instance CRN (from <strong>Instances</strong>). The
          server uses them only for this request — they are{" "}
          <strong>not stored</strong>. Use HTTPS when the API is deployed.
        </p>
        <form onSubmit={onVerify} className="mt-4 space-y-3">
          <div>
            <label
              htmlFor="ibm-api-token"
              className="text-xs font-medium text-on-surface-variant"
            >
              API token
            </label>
            <input
              id="ibm-api-token"
              name="api_token"
              type="password"
              autoComplete="off"
              value={apiToken}
              onChange={(e) => setApiToken(e.target.value)}
              className="mt-1 w-full rounded-md border border-outline-variant bg-surface px-3 py-2 font-mono text-sm text-on-surface"
              placeholder="44-character token"
            />
          </div>
          <div>
            <label
              htmlFor="ibm-instance-crn"
              className="text-xs font-medium text-on-surface-variant"
            >
              Instance CRN (optional)
            </label>
            <input
              id="ibm-instance-crn"
              name="instance_crn"
              type="text"
              autoComplete="off"
              value={instanceCrn}
              onChange={(e) => setInstanceCrn(e.target.value)}
              className="mt-1 w-full rounded-md border border-outline-variant bg-surface px-3 py-2 font-mono text-xs text-on-surface"
              placeholder="crn:v1:bluemix:public:quantum-computing:..."
            />
          </div>
          <button
            type="submit"
            disabled={verifyLoading}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-on-primary disabled:opacity-50"
          >
            {verifyLoading ? "Checking…" : "Verify connection"}
          </button>
        </form>
        {verifyError ? (
          <p className="mt-3 text-sm text-error" role="alert">
            {verifyError}
          </p>
        ) : null}
        {verifyResult ? (
          <p className="mt-3 text-sm text-tertiary" role="status">
            {verifyResult}
          </p>
        ) : null}
      </section>

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
                Server configuration (sanitized)
              </h2>
              <p className="mb-2 text-xs text-on-surface-variant">
                Tokens and raw instance CRNs from <code>.env</code> are redacted
                in this view.
              </p>
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
