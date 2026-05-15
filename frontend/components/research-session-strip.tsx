"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { JobResponse, LatestRunResponse } from "@/lib/api";
import { fetchJobs, fetchLatestRun } from "@/lib/api";

function hasResults(run: LatestRunResponse | null): boolean {
  if (!run) return false;
  return run.status !== "no_results" && run.status !== "empty";
}

function runningJob(jobs: JobResponse[]): JobResponse | null {
  return jobs.find((j) => j.status === "running" || j.status === "queued") ?? null;
}

export function ResearchSessionStrip() {
  const [run, setRun] = useState<LatestRunResponse | null>(null);
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [latest, allJobs] = await Promise.all([
          fetchLatestRun().catch(() => null),
          fetchJobs().catch(() => []),
        ]);
        if (!cancelled) {
          setRun(latest);
          setJobs(allJobs);
          setLoaded(true);
        }
      } catch {
        if (!cancelled) setLoaded(true);
      }
    }

    load();
    const id = setInterval(load, 10000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const activeJob = useMemo(() => runningJob(jobs), [jobs]);
  const ready = hasResults(run);

  return (
    <div className="mb-6 rounded-lg border border-outline-variant/20 bg-surface-container-lowest/70 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs text-on-surface-variant">
          {!loaded ? (
            "Syncing research session\u2026"
          ) : activeJob ? (
            <>
              Active job:{" "}
              <span className="font-mono text-on-surface">{activeJob.id}</span>{" "}
              ({activeJob.status})
            </>
          ) : ready ? (
            <>
              Latest run ready in{" "}
              <span className="font-mono text-on-surface">{run?.results_dir ?? "results/"}</span>
            </>
          ) : (
            "No run results yet — start a pipeline run to begin analysis."
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          <Link
            href="/experiments"
            className="inline-flex items-center gap-1 rounded border border-outline/20 bg-surface-container-lowest px-2.5 py-1 text-xs font-medium text-on-surface hover:bg-surface-container"
          >
            <span className="material-symbols-outlined text-sm" aria-hidden>
              monitoring
            </span>
            Experiment
          </Link>
          <Link
            href="/validation"
            className="inline-flex items-center gap-1 rounded border border-outline/20 bg-surface-container-lowest px-2.5 py-1 text-xs font-medium text-on-surface hover:bg-surface-container"
          >
            <span className="material-symbols-outlined text-sm" aria-hidden>
              fact_check
            </span>
            Validation
          </Link>
          <Link
            href="/visualization"
            className="inline-flex items-center gap-1 rounded border border-outline/20 bg-surface-container-lowest px-2.5 py-1 text-xs font-medium text-on-surface hover:bg-surface-container"
          >
            <span className="material-symbols-outlined text-sm" aria-hidden>
              view_in_ar
            </span>
            Visual
          </Link>
        </div>
      </div>
    </div>
  );
}
