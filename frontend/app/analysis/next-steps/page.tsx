"use client";

import { useEffect, useState } from "react";
import type { AnalysisSummaryResponse } from "@/lib/api";
import { fetchAnalysisSummary, getApiBaseUrl } from "@/lib/api";

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
          Analysis: next steps
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Automated recommendations based on the latest pipeline run.
        </p>
      </header>

      {loading ? (
        <p className="text-sm text-on-surface-variant" role="status">
          Sequencing&hellip;
        </p>
      ) : error ? (
        <div className="rounded-lg border border-error/40 bg-error-container/20 p-4">
          <p className="text-sm font-medium text-error">
            Could not load analysis
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">{error}</p>
          <p className="mt-3 text-xs text-on-surface-variant">
            Base URL:{" "}
            <code className="text-on-surface">{getApiBaseUrl()}</code>
          </p>
        </div>
      ) : data?.status === "no_results" ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          No pipeline results yet. Run the pipeline first.
        </div>
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
            </li>
          ))}
        </ul>
      ) : (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 text-sm text-on-surface-variant">
          No specific recommendations at this time.
        </div>
      )}
    </div>
  );
}

interface Suggestion {
  title: string;
  detail: string;
}

function deriveNextSteps(data: AnalysisSummaryResponse): Suggestion[] {
  const out: Suggestion[] = [];

  if (data.quantum_count === 0) {
    out.push({
      title: "Enable quantum models",
      detail:
        "The last run had no quantum models. Re-run without --skip_quantum to compare QSVC/VQC against classical baselines.",
    });
  }

  if (data.ensemble_count === 0) {
    out.push({
      title: "Try ensemble methods",
      detail:
        "Add --run_ensemble --ensemble_method stacking to combine classical and quantum predictions for potentially higher PR-AUC.",
    });
  }

  if (data.best_pr_auc != null && data.best_pr_auc < 0.7) {
    out.push({
      title: "Improve embedding quality",
      detail:
        "Best PR-AUC is below 0.70. Consider RotatE embeddings (--embedding_method RotatE --embedding_dim 128 --embedding_epochs 200) and full-graph context (--full_graph_embeddings).",
    });
  }

  if (data.best_pr_auc != null && data.best_pr_auc >= 0.7 && data.best_pr_auc < 0.8) {
    out.push({
      title: "Fine-tune hyperparameters",
      detail:
        "PR-AUC is promising. Run Optuna search (python scripts/optuna_pipeline_search.py --n_trials 50 --objective best) to squeeze out further gains.",
    });
  }

  if (data.best_pr_auc != null && data.best_pr_auc >= 0.8) {
    out.push({
      title: "Validate on additional relations",
      detail:
        "PR-AUC >= 0.80 is strong. Test generalization by running on other relations (e.g. --relation DaG).",
    });
  }

  if (data.model_count > 0 && data.classical_count > 0) {
    out.push({
      title: "Classical tuning",
      detail:
        "Add --tune_classical to run GridSearchCV on ExtraTrees, RandomForest, and LogisticRegression for potential gains with minimal overhead.",
    });
  }

  return out;
}
