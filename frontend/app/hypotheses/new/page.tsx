"use client";

import { RankedForm } from "@/components/ranked-form";
import { ResearchNextActions } from "@/components/research-next-actions";
import { ApiRecoveryCard } from "@/components/api-recovery-card";
import {
  createHypothesis,
  fetchHypotheses,
  fetchHypothesisExperiments,
  type ExperimentHistory,
  type Hypothesis,
  updateHypothesis,
} from "@/lib/api";
import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";

export default function NewHypothesisPage() {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [timeline, setTimeline] = useState<ExperimentHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");

  const selected = useMemo(
    () => hypotheses.find((h) => h.id === selectedId) ?? null,
    [hypotheses, selectedId],
  );

  const loadHypotheses = useCallback(async () => {
    try {
      const list = await fetchHypotheses();
      setHypotheses(list);
      setSelectedId((prev) => {
        if (prev && list.some((h) => h.id === prev)) return prev;
        return list[0]?.id ?? "";
      });
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load hypotheses");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHypotheses();
  }, [loadHypotheses]);

  useEffect(() => {
    if (!selectedId) {
      setTimeline([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const items = await fetchHypothesisExperiments(selectedId);
        if (!cancelled) setTimeline(items);
      } catch {
        if (!cancelled) setTimeline([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  async function onCreateHypothesis(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim() || !newDescription.trim()) return;
    setSaving(true);
    try {
      const created = await createHypothesis({
        name: newName.trim(),
        description: newDescription.trim(),
        status: "draft",
      });
      setNewName("");
      setNewDescription("");
      await loadHypotheses();
      setSelectedId(created.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create hypothesis");
    } finally {
      setSaving(false);
    }
  }

  async function onUpdateSelected(
    patch: Partial<Pick<Hypothesis, "name" | "description" | "disease_focus" | "notes" | "status">>,
  ) {
    if (!selected) return;
    setSaving(true);
    try {
      await updateHypothesis(selected.id, patch);
      await loadHypotheses();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update hypothesis");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          Ranked candidates
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Hypothesis-driven compound ranking: given a disease and a mechanism hypothesis
          (H-001, H-002, H-003), the model returns the top-K compounds ranked by score.
          This is different from{" "}
          <a href="/predict" className="text-primary underline-offset-2 hover:underline">
            single-pair prediction
          </a>
          {" "}— use this page when you want a ranked list across many candidates, not a
          one-off probability for a specific drug–disease pair.
        </p>
      </header>

      {error ? <ApiRecoveryCard title="Could not load hypothesis workspace" error={error} /> : null}

      <div className="rounded-lg border border-outline-variant/20 bg-surface-container-lowest/60 p-4 text-xs text-on-surface-variant">
        Tip: rank candidates first, then validate individual pairs in{" "}
        <Link href="/predict" className="text-primary underline-offset-2 hover:underline">
          Predict treatment
        </Link>{" "}
        and inspect context in{" "}
        <Link href="/knowledge-graph" className="text-primary underline-offset-2 hover:underline">
          Knowledge graph
        </Link>
        .
      </div>

      {loading ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-4 text-sm text-on-surface-variant">
          Loading hypotheses...
        </div>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[340px_1fr]">
          <section className="space-y-4 rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-4">
            <div>
              <h2 className="font-headline text-lg font-semibold text-on-surface">
                Hypothesis library
              </h2>
              <p className="mt-1 text-xs text-on-surface-variant">
                Saved hypotheses persist across sessions and track tested runs.
              </p>
            </div>
            <div className="space-y-2">
              {hypotheses.map((hypothesis) => (
                <button
                  key={hypothesis.id}
                  type="button"
                  onClick={() => setSelectedId(hypothesis.id)}
                  className={`w-full rounded-lg border p-3 text-left transition-colors ${
                    selectedId === hypothesis.id
                      ? "border-primary/40 bg-primary/10"
                      : "border-outline/15 bg-surface-container-lowest hover:bg-surface-container"
                  }`}
                >
                  <p className="text-sm font-medium text-on-surface">
                    {hypothesis.id} · {hypothesis.name}
                  </p>
                  <p className="mt-1 line-clamp-2 text-xs text-on-surface-variant">
                    {hypothesis.description}
                  </p>
                  <p className="mt-1 text-[11px] text-on-surface-variant">
                    status: {hypothesis.status}
                    {hypothesis.last_tested_run_id
                      ? ` · last tested: ${hypothesis.last_tested_run_id}`
                      : ""}
                  </p>
                </button>
              ))}
            </div>

            <form onSubmit={onCreateHypothesis} className="space-y-2 border-t border-outline/10 pt-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
                Create hypothesis
              </p>
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Name"
                className="w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              />
              <textarea
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                placeholder="Description"
                rows={3}
                className="w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              />
              <button
                type="submit"
                disabled={saving || !newName.trim() || !newDescription.trim()}
                className="rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-on-primary disabled:opacity-50"
              >
                {saving ? "Saving..." : "Save hypothesis"}
              </button>
            </form>
          </section>

          <section className="space-y-4">
            {selected ? (
              <>
                <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-4">
                  <h2 className="font-headline text-lg font-semibold text-on-surface">
                    Active hypothesis
                  </h2>
                  <p className="mt-1 text-xs text-on-surface-variant">
                    {selected.id} · metadata used by ranking and run linkage
                  </p>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    <Field
                      label="Name"
                      value={selected.name}
                      onSave={(value) => onUpdateSelected({ name: value })}
                      disabled={saving}
                    />
                    <Field
                      label="Status"
                      value={selected.status}
                      onSave={(value) =>
                        onUpdateSelected({
                          status: value as Hypothesis["status"],
                        })
                      }
                      options={["draft", "active", "tested"]}
                      disabled={saving}
                    />
                    <Field
                      label="Disease focus"
                      value={selected.disease_focus ?? ""}
                      onSave={(value) => onUpdateSelected({ disease_focus: value })}
                      disabled={saving}
                    />
                    <Field
                      label="Notes"
                      value={selected.notes ?? ""}
                      onSave={(value) => onUpdateSelected({ notes: value })}
                      disabled={saving}
                    />
                  </div>
                  <div className="mt-3">
                    <Field
                      label="Description"
                      value={selected.description}
                      onSave={(value) => onUpdateSelected({ description: value })}
                      multiline
                      disabled={saving}
                    />
                  </div>
                </div>

                <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-4">
                  <h3 className="font-headline text-lg font-semibold text-on-surface">
                    Rank candidates
                  </h3>
                  <p className="mt-1 text-xs text-on-surface-variant">
                    Uses selected hypothesis and disease to produce candidate rankings.
                  </p>
                  <div className="mt-4">
                    <RankedForm
                      hypothesisId={selected.id}
                      onHypothesisIdChange={setSelectedId}
                      diseaseFocus={selected.disease_focus}
                    />
                  </div>
                </div>

                <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-4">
                  <h3 className="font-headline text-lg font-semibold text-on-surface">
                    Hypothesis timeline
                  </h3>
                  {timeline.length === 0 ? (
                    <p className="mt-2 text-xs text-on-surface-variant">
                      No runs linked yet. Launch a run from New run and associate it with this hypothesis.
                    </p>
                  ) : (
                    <div className="mt-3 overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="bg-surface-container-high text-left text-xs uppercase tracking-wide text-on-surface-variant">
                            <th className="px-3 py-2">Job</th>
                            <th className="px-3 py-2">Status</th>
                            <th className="px-3 py-2">Created</th>
                            <th className="px-3 py-2">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {timeline.map((run) => (
                            <tr key={run.job_id} className="border-b border-outline-variant/10">
                              <td className="px-3 py-2 font-mono text-on-surface">{run.job_id}</td>
                              <td className="px-3 py-2 text-on-surface-variant">{run.status}</td>
                              <td className="px-3 py-2 text-on-surface-variant">
                                {new Date(run.created_at * 1000).toLocaleString()}
                              </td>
                              <td className="px-3 py-2">
                                <div className="flex flex-wrap gap-2 text-xs">
                                  <Link href="/simulation" className="text-primary underline">
                                    Open jobs
                                  </Link>
                                  <Link href="/visualization?tab=comparison" className="text-primary underline">
                                    Compare in charts
                                  </Link>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-4 text-sm text-on-surface-variant">
                Create a hypothesis to start ranking candidates.
              </div>
            )}
          </section>
        </div>
      )}

      <ResearchNextActions context="ranked" />
    </div>
  );
}

function Field({
  label,
  value,
  onSave,
  options,
  multiline = false,
  disabled = false,
}: {
  label: string;
  value: string;
  onSave: (value: string) => void;
  options?: string[];
  multiline?: boolean;
  disabled?: boolean;
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  return (
    <div className="space-y-1">
      <label className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </label>
      {options ? (
        <select
          value={draft}
          onChange={(e) => {
            const next = e.target.value;
            setDraft(next);
            onSave(next);
          }}
          disabled={disabled}
          className="w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-60"
        >
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : multiline ? (
        <textarea
          value={draft}
          rows={4}
          onBlur={() => onSave(draft)}
          onChange={(e) => setDraft(e.target.value)}
          disabled={disabled}
          className="w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-60"
        />
      ) : (
        <input
          value={draft}
          onBlur={() => onSave(draft)}
          onChange={(e) => setDraft(e.target.value)}
          disabled={disabled}
          className="w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-60"
        />
      )}
    </div>
  );
}
