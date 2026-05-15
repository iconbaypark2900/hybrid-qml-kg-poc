"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { AnalysisSummaryResponse } from "@/lib/api";
import { fetchAnalysisSummary } from "@/lib/api";
import { ApiRecoveryCard } from "@/components/api-recovery-card";
import { LoadingBlock } from "@/components/spinner";
import { NoPipelineResultsCta } from "@/components/no-pipeline-results-cta";
import { ResearchNextActions } from "@/components/research-next-actions";

export default function NextStepsPage() {
  const [data, setData] = useState<AnalysisSummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await fetchAnalysisSummary();
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

  const suggestions = data ? deriveNextSteps(data) : [];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          Recommendations
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Automated next-step suggestions derived from the latest pipeline run — e.g. enable
          quantum models, improve embeddings, tune hyperparameters. Based on the same summary
          as{" "}
          <Link href="/analysis/drug-delivery" className="text-primary underline-offset-2 hover:underline">
            Drug delivery analysis
          </Link>
          {" "}— that page shows aggregated performance numbers from the same data.
        </p>
      </header>

      {loading ? (
        <LoadingBlock text="Loading analysis summary…" />
      ) : error ? (
        <ApiRecoveryCard title="Could not load analysis" error={error} />
      ) : data?.status === "no_results" ? (
        <NoPipelineResultsCta />
      ) : suggestions.length > 0 ? (
        <ul className="space-y-3">
          {suggestions.map((s, i) => (
            <li
              key={i}
              className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-4"
            >
              <p className="text-sm font-medium text-on-surface">{s.title}</p>
              <p className="mt-1 text-xs text-on-surface-variant">
                {s.detail}
              </p>
              <Link
                href={s.href}
                className="mt-3 inline-flex items-center gap-1 rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-1.5 text-xs font-medium text-on-surface hover:bg-surface-container"
              >
                <span className="material-symbols-outlined text-sm" aria-hidden>
                  arrow_forward
                </span>
                {s.cta}
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          No specific recommendations at this time.
        </div>
      )}

      <ResearchNextActions context="analysis" />
    </div>
  );
}

interface Suggestion {
  title: string;
  detail: string;
  href: string;
  cta: string;
}

function deriveNextSteps(data: AnalysisSummaryResponse): Suggestion[] {
  const out: Suggestion[] = [];

  if (data.quantum_count === 0) {
    out.push({
      title: "Enable quantum models",
      detail:
        "The latest run did not include quantum models. Launch a hybrid preset to compare QSVC/VQC against classical baselines.",
      href: "/simulation/parameters?preset=hybrid-default",
      cta: "Open hybrid preset",
    });
  }

  if (data.ensemble_count === 0) {
    out.push({
      title: "Try ensemble methods",
      detail:
        "Combine classical and quantum predictions in a single run and compare aggregate performance in the comparison charts.",
      href: "/simulation/parameters?preset=hybrid-default",
      cta: "Configure ensemble run",
    });
  }

  if (data.best_pr_auc != null && data.best_pr_auc < 0.7) {
    out.push({
      title: "Improve embedding quality",
      detail:
        "Best PR-AUC is below 0.70. Try a stronger embedding preset with larger dimensions and epochs.",
      href: "/simulation/parameters?preset=classical-baseline",
      cta: "Open embedding-focused preset",
    });
  }

  if (data.best_pr_auc != null && data.best_pr_auc >= 0.7 && data.best_pr_auc < 0.8) {
    out.push({
      title: "Fine-tune hyperparameters",
      detail:
        "PR-AUC is promising. Run a tuning-focused follow-up to improve model selection and confidence bands.",
      href: "/simulation/parameters?preset=hybrid-default",
      cta: "Launch tuning follow-up",
    });
  }

  if (data.best_pr_auc != null && data.best_pr_auc >= 0.8) {
    out.push({
      title: "Validate on additional relations",
      detail:
        "PR-AUC >= 0.80 is strong. Validate generalization by exploring additional relations in New run.",
      href: "/simulation/parameters",
      cta: "Open New run",
    });
  }

  if (data.model_count > 0 && data.classical_count > 0) {
    out.push({
      title: "Classical tuning",
      detail:
        "Run classical hyperparameter tuning and compare with current leaderboard before selecting the next hypothesis iteration.",
      href: "/simulation/parameters?preset=classical-baseline",
      cta: "Configure classical tuning",
    });
  }

  return out;
}
