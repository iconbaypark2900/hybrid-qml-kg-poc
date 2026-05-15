"use client";

import { useEffect } from "react";
import { reportFrontendError } from "@/lib/client-error-reporting";

export default function V2Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    reportFrontendError({
      source: "v2-error-boundary",
      message: error.message,
      digest: error.digest,
      path: typeof window !== "undefined" ? window.location.pathname : undefined,
    });
  }, [error]);

  return (
    <div className="rounded-lg border border-error/40 bg-error-container/10 p-6 text-on-surface">
      <p className="font-label text-xs font-bold uppercase tracking-widest text-error">
        V2 route error
      </p>
      <h1 className="mt-3 font-headline text-2xl font-semibold">
        This research view could not load.
      </h1>
      <p className="mt-3 text-sm text-on-surface-variant">
        The local error reporter logged this failure to the browser console. Retry the route or check API health.
      </p>
      <button
        type="button"
        onClick={reset}
        className="mt-5 rounded-lg border border-primary/60 bg-primary/20 px-4 py-2 text-sm font-semibold"
      >
        Retry view
      </button>
    </div>
  );
}
