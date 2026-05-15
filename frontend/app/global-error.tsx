"use client";

import { useEffect } from "react";
import { reportFrontendError } from "@/lib/client-error-reporting";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    reportFrontendError({
      source: "global-error",
      message: error.message,
      digest: error.digest,
      path: typeof window !== "undefined" ? window.location.pathname : undefined,
    });
  }, [error]);

  return (
    <html lang="en" className="dark">
      <body className="bg-background p-8 text-on-surface">
        <main className="mx-auto max-w-2xl rounded-lg border border-error/40 bg-error-container/10 p-6">
          <p className="font-label text-xs font-bold uppercase tracking-widest text-error">
            Application error
          </p>
          <h1 className="mt-3 font-headline text-3xl font-semibold">
            The dashboard hit an unrecoverable error.
          </h1>
          <p className="mt-3 text-sm text-on-surface-variant">
            Request a retry, then check local browser and backend logs if the error repeats.
          </p>
          <button
            type="button"
            onClick={reset}
            className="mt-5 rounded-lg border border-primary/60 bg-primary/20 px-4 py-2 text-sm font-semibold"
          >
            Try again
          </button>
        </main>
      </body>
    </html>
  );
}
