"use client";

import { useEffect, useState } from "react";
import {
  fetchIBMQuantumConfig,
  saveIBMQuantumConfig,
  verifyIBMQuantumConfig,
  type IBMQuantumConfigResponse,
} from "@/lib/api";
import { ApiRecoveryCard } from "@/components/api-recovery-card";
import { Spinner } from "@/components/spinner";

const DEFAULT_CHANNEL = "ibm_quantum_platform";

export function IBMQuantumSettings() {
  const [tenantId, setTenantId] = useState("local-dev");
  const [token, setToken] = useState("");
  const [instance, setInstance] = useState("");
  const [channel, setChannel] = useState(DEFAULT_CHANNEL);
  const [metadata, setMetadata] = useState<IBMQuantumConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyAction, setBusyAction] = useState<"save" | "verify" | "stored" | null>(
    null,
  );
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchIBMQuantumConfig("local-dev");
        if (!cancelled) {
          setMetadata(data);
          setInstance(data.instance ?? "");
          setChannel(data.channel || DEFAULT_CHANNEL);
        }
      } catch (err) {
        if (!cancelled) {
          setMetadata(null);
          setError(err instanceof Error ? err.message : "Could not load IBM settings.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function refreshMetadata(nextTenantId = tenantId) {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchIBMQuantumConfig(nextTenantId.trim() || undefined);
      setMetadata(data);
      setInstance(data.instance ?? "");
      setChannel(data.channel || DEFAULT_CHANNEL);
    } catch (err) {
      setMetadata(null);
      setError(err instanceof Error ? err.message : "Could not load IBM settings.");
    } finally {
      setLoading(false);
    }
  }

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    setMessage(null);
    setError(null);
    const cleanedToken = token.trim();
    if (!cleanedToken) {
      setError("Enter an IBM Quantum token before saving.");
      return;
    }
    setBusyAction("save");
    try {
      const saved = await saveIBMQuantumConfig({
        tenantId,
        token: cleanedToken,
        instance,
        channel,
      });
      setMetadata(saved);
      setMessage(saved.message ?? "IBM Quantum credentials saved.");
      setToken("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed.");
    } finally {
      setBusyAction(null);
    }
  }

  async function onVerifyTyped() {
    setMessage(null);
    setError(null);
    const cleanedToken = token.trim();
    if (!cleanedToken) {
      setError("Enter an IBM Quantum token to verify typed credentials.");
      return;
    }
    setBusyAction("verify");
    try {
      const result = await verifyIBMQuantumConfig({
        tenantId,
        token: cleanedToken,
        instance,
        channel,
      });
      setMessage(formatVerifyResult(result.message, result.backend_count, false));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed.");
    } finally {
      setBusyAction(null);
    }
  }

  async function onVerifyStored() {
    setMessage(null);
    setError(null);
    setBusyAction("stored");
    try {
      const result = await verifyIBMQuantumConfig({ tenantId });
      setMessage(formatVerifyResult(result.message, result.backend_count, true));
      await refreshMetadata(tenantId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Stored credential verification failed.");
    } finally {
      setBusyAction(null);
    }
  }

  const hasStoredCredentials = Boolean(metadata?.configured);

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-primary/25 bg-primary/10 p-4">
        <h2 className="font-headline text-lg font-semibold text-on-surface">
          Key distinction
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-on-surface-variant">
          <code>NEXT_PUBLIC_API_KEY</code> or <code>X-API-Key</code> protects this
          app&apos;s API. The IBM Quantum API token is your platform credential from{" "}
          <a
            href="https://quantum.ibm.com/"
            className="text-primary underline underline-offset-2"
            target="_blank"
            rel="noreferrer"
          >
            quantum.ibm.com
          </a>
          . Do not paste that IBM token into chat; enter it only here or send it
          directly to your local API.
        </p>
      </section>

      <form
        onSubmit={onSave}
        className="space-y-4 rounded-lg border border-outline-variant/35 bg-surface-container-lowest/70 p-4"
      >
        <div>
          <h2 className="font-headline text-lg font-semibold text-on-surface">
            IBM Quantum Integration
          </h2>
          <p className="mt-1 text-sm text-on-surface-variant">
            Credentials are scoped to the tenant ID below and stored server-side in
            SQLite. Set <code>INTEGRATION_ENCRYPTION_KEY</code> in shared environments
            so tokens use Fernet encryption instead of local/dev base64.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="block text-xs font-medium text-on-surface-variant">
            Tenant ID
            <input
              value={tenantId}
              onChange={(event) => setTenantId(event.target.value)}
              className="mt-1 w-full rounded-md border border-outline-variant bg-surface px-3 py-2 font-mono text-sm text-on-surface"
              placeholder="tenant-or-session-id"
            />
          </label>
          <label className="block text-xs font-medium text-on-surface-variant">
            Channel
            <input
              value={channel}
              onChange={(event) => setChannel(event.target.value)}
              className="mt-1 w-full rounded-md border border-outline-variant bg-surface px-3 py-2 font-mono text-sm text-on-surface"
              placeholder={DEFAULT_CHANNEL}
            />
          </label>
        </div>

        <label className="block text-xs font-medium text-on-surface-variant">
          Token
          <input
            type="password"
            autoComplete="off"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            className="mt-1 w-full rounded-md border border-outline-variant bg-surface px-3 py-2 font-mono text-sm text-on-surface"
            placeholder="IBM Quantum API token"
          />
        </label>

        <label className="block text-xs font-medium text-on-surface-variant">
          Instance CRN (optional)
          <input
            value={instance}
            onChange={(event) => setInstance(event.target.value)}
            className="mt-1 w-full rounded-md border border-outline-variant bg-surface px-3 py-2 font-mono text-xs text-on-surface"
            placeholder="crn:v1:bluemix:public:quantum-computing:..."
          />
        </label>

        <div className="flex flex-wrap gap-2">
          <ActionButton
            busy={busyAction === "verify"}
            disabled={busyAction !== null}
            onClick={onVerifyTyped}
          >
            Verify typed credentials
          </ActionButton>
          <button
            type="submit"
            disabled={busyAction !== null}
            className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-on-primary disabled:opacity-50"
          >
            {busyAction === "save" ? <Spinner /> : null}
            Save credentials
          </button>
          <ActionButton
            busy={busyAction === "stored"}
            disabled={!hasStoredCredentials || busyAction !== null}
            onClick={onVerifyStored}
          >
            Verify stored credentials
          </ActionButton>
          <ActionButton
            busy={loading}
            disabled={busyAction !== null}
            onClick={() => refreshMetadata()}
          >
            Load tenant status
          </ActionButton>
        </div>
      </form>

      {message ? (
        <p className="rounded-lg border border-tertiary/25 bg-tertiary/10 p-3 text-sm text-tertiary">
          {message}
        </p>
      ) : null}
      {error ? <ApiRecoveryCard title="IBM Quantum settings error" error={error} /> : null}

      <section className="rounded-lg border border-outline-variant/35 bg-surface-container-lowest/70 p-4">
        <h2 className="font-headline text-lg font-semibold text-on-surface">
          Stored tenant metadata
        </h2>
        {loading ? (
          <p className="mt-3 inline-flex items-center gap-2 text-sm text-on-surface-variant">
            <Spinner />
            Loading IBM Quantum metadata...
          </p>
        ) : metadata?.configured ? (
          <dl className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <Stat label="Tenant" value={metadata.tenant_id} />
            <Stat label="Token" value={metadata.token_preview ?? "[REDACTED]"} />
            <Stat label="Storage" value={metadata.secret_storage ?? "unknown"} />
            <Stat label="Last verified" value={formatTime(metadata.last_verified_at)} />
            <Stat label="Instance" value={metadata.instance ?? "Not set"} wide />
            <Stat label="Updated" value={formatTime(metadata.updated_at)} />
          </dl>
        ) : (
          <p className="mt-3 text-sm text-on-surface-variant">
            {metadata?.message ?? "No IBM Quantum credentials are stored for this tenant."}
          </p>
        )}
      </section>
    </div>
  );
}

function ActionButton({
  busy,
  disabled,
  onClick,
  children,
}: {
  busy: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={busy || disabled}
      onClick={onClick}
      className="inline-flex items-center gap-2 rounded-md border border-outline/25 bg-surface-container-lowest px-4 py-2 text-sm font-medium text-on-surface hover:bg-surface-container disabled:opacity-50"
    >
      {busy ? <Spinner /> : null}
      {children}
    </button>
  );
}

function Stat({
  label,
  value,
  wide = false,
}: {
  label: string;
  value: string;
  wide?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border border-outline-variant/20 bg-surface-container-high/50 p-3 ${
        wide ? "md:col-span-2 xl:col-span-3" : ""
      }`}
    >
      <dt className="text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
        {label}
      </dt>
      <dd className="mt-1 break-words font-mono text-xs text-on-surface">{value}</dd>
    </div>
  );
}

function formatVerifyResult(message: string, backendCount: number, stored: boolean) {
  const source = stored ? "stored tenant credentials" : "typed credentials";
  return `${message} Verified ${source}; ${backendCount} backend(s) visible.`;
}

function formatTime(value: number | null) {
  if (!value) return "Never";
  return new Date(value * 1000).toLocaleString();
}
