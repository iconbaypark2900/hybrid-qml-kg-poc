"use client";

import Link from "next/link";
import { getApiBaseUrl } from "@/lib/api";

interface ApiRecoveryCardProps {
  title: string;
  error: string;
  details?: string;
}

export function ApiRecoveryCard({ title, error, details }: ApiRecoveryCardProps) {
  return (
    <div className="rounded-lg border border-error/40 bg-error-container/20 p-4">
      <p className="text-sm font-medium text-error">{title}</p>
      <p className="mt-1 text-xs text-on-surface-variant">{error}</p>
      <p className="mt-3 text-xs text-on-surface-variant">
        Base URL: <code className="text-on-surface">{getApiBaseUrl()}</code>
      </p>
      {details ? <p className="mt-1 text-xs text-on-surface-variant">{details}</p> : null}
      <div className="mt-3 flex flex-wrap gap-2">
        <Link
          href="/system"
          className="inline-flex items-center gap-1 rounded border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-xs font-medium text-on-surface hover:bg-surface-container"
        >
          <span className="material-symbols-outlined text-sm" aria-hidden>
            dns
          </span>
          System status
        </Link>
        <Link
          href="/simulation/parameters"
          className="inline-flex items-center gap-1 rounded border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-xs font-medium text-on-surface hover:bg-surface-container"
        >
          <span className="material-symbols-outlined text-sm" aria-hidden>
            tune
          </span>
          New run
        </Link>
        <Link
          href="/simulation"
          className="inline-flex items-center gap-1 rounded border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-xs font-medium text-on-surface hover:bg-surface-container"
        >
          <span className="material-symbols-outlined text-sm" aria-hidden>
            play_circle
          </span>
          Pipeline jobs
        </Link>
      </div>
    </div>
  );
}
