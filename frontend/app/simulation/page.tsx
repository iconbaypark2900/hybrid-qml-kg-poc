"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import type { JobResponse } from "@/lib/api";
import { fetchJobs } from "@/lib/api";
import { ApiRecoveryCard } from "@/components/api-recovery-card";
import { ResearchNextActions } from "@/components/research-next-actions";
import { LoadingBlock } from "@/components/spinner";

export default function SimulationPage() {
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const list = await fetchJobs();
      setJobs(list);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load jobs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  return (
    <div className="space-y-6">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
            Pipeline jobs
          </h1>
          <p className="mt-1 text-sm text-on-surface-variant">
            Queue and monitor experiment runs; open results when jobs complete.
          </p>
        </div>
        <Link
          href="/simulation/parameters"
          className="primary-gradient rounded-lg px-5 py-2.5 text-sm font-semibold text-on-primary shadow-glow"
        >
          New run
        </Link>
      </header>

      {loading ? (
        <LoadingBlock text="Loading jobs…" />
      ) : error ? (
        <ApiRecoveryCard title="Could not load jobs" error={error} />
      ) : jobs.length === 0 ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          No jobs yet. Start one from{" "}
          <Link href="/simulation/parameters" className="text-primary underline">
            New run
          </Link>
          , then return here to track progress and open results.
        </div>
      ) : (
        <div className="space-y-4">
          {jobs.some((j) => j.status === "success") ? (
            <div className="rounded-lg border border-tertiary/30 bg-tertiary/10 p-3 text-xs text-on-surface">
              At least one run completed successfully. Open{" "}
              <Link href="/experiments" className="text-primary underline">
                Latest run &amp; models
              </Link>{" "}
              or{" "}
              <Link href="/visualization?tab=comparison" className="text-primary underline">
                Charts &amp; exploration
              </Link>
              .
            </div>
          ) : null}
          <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-surface-container-high text-left text-xs uppercase tracking-wide text-on-surface-variant">
                <th className="px-3 py-2">ID</th>
                <th className="px-3 py-2">Hypothesis</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Created</th>
                <th className="px-3 py-2">Duration</th>
                <th className="px-3 py-2">Exit</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr
                  key={j.id}
                  className="border-b border-outline-variant/10 hover:bg-surface-container-lowest/50"
                >
                  <td className="px-3 py-2 font-mono text-on-surface">
                    <Link
                      href={`/simulation?job=${j.id}`}
                      className="text-primary underline"
                    >
                      {j.id}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    {j.hypothesis_id ? (
                      <Link href="/hypotheses/new" className="text-primary underline">
                        {j.hypothesis_id}
                      </Link>
                    ) : (
                      <span className="text-on-surface-variant">—</span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <StatusBadge status={j.status} />
                  </td>
                  <td className="px-3 py-2 text-on-surface-variant">
                    {new Date(j.created_at * 1000).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 font-mono text-on-surface-variant">
                    {duration(j)}
                  </td>
                  <td className="px-3 py-2 font-mono text-on-surface-variant">
                    {j.exit_code ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        </div>
      )}

      <ResearchNextActions context="simulation" />
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    queued: "bg-secondary/20 text-secondary",
    running: "bg-primary/20 text-primary",
    success: "bg-tertiary/20 text-tertiary",
    failed: "bg-error/20 text-error",
  };
  return (
    <span
      className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ${colors[status] ?? "text-on-surface-variant"}`}
    >
      {status}
    </span>
  );
}

function duration(j: JobResponse): string {
  if (!j.started_at) return "—";
  const end = j.finished_at ?? Date.now() / 1000;
  const s = Math.round(end - j.started_at);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}
