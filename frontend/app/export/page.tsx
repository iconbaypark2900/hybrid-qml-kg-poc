"use client";

import { useEffect, useState } from "react";
import type { ExportFileInfo } from "@/lib/api";
import { fetchExports, exportDownloadUrl, getApiBaseUrl } from "@/lib/api";

export default function ExportPage() {
  const [files, setFiles] = useState<ExportFileInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetchExports();
        if (!cancelled) {
          setFiles(res.files);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Request failed");
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
          Export
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Download pipeline artifacts from the results directory.
        </p>
      </header>

      {loading ? (
        <p className="text-sm text-on-surface-variant" role="status">
          Sequencing&hellip;
        </p>
      ) : error ? (
        <div className="rounded-lg border border-error/40 bg-error-container/20 p-4">
          <p className="text-sm font-medium text-error">
            Could not list exports
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">{error}</p>
          <p className="mt-3 text-xs text-on-surface-variant">
            Base URL:{" "}
            <code className="text-on-surface">{getApiBaseUrl()}</code>
          </p>
        </div>
      ) : files.length === 0 ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          No exportable files found. Run the pipeline first.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-surface-container-high text-left text-xs uppercase tracking-wide text-on-surface-variant">
                <th className="px-3 py-2">File</th>
                <th className="px-3 py-2 text-right">Size</th>
                <th className="px-3 py-2">Modified</th>
                <th className="px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {files.map((f) => (
                <tr
                  key={f.name}
                  className="border-b border-outline-variant/10 hover:bg-surface-container-lowest/50"
                >
                  <td className="px-3 py-2 font-mono text-on-surface">
                    {f.name}
                  </td>
                  <td className="px-3 py-2 text-right text-on-surface-variant">
                    {humanSize(f.size_bytes)}
                  </td>
                  <td className="px-3 py-2 text-on-surface-variant">
                    {new Date(f.modified * 1000).toLocaleString()}
                  </td>
                  <td className="px-3 py-2">
                    <a
                      href={exportDownloadUrl(f.name)}
                      download
                      className="text-primary underline"
                    >
                      Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
