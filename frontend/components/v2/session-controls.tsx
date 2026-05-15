"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  createResearchSession,
  exportEvidencePacketUrl,
  fetchResearchSessions,
  updateResearchSession,
  type ResearchSession,
} from "@/lib/api";
import {
  buildV2Params,
  parseV2Session,
  type V2Session,
} from "@/lib/v2-data";
import type { V2EvidenceSnapshot } from "@/lib/v2-live-data";

interface ReviewerState {
  name: string;
  email: string;
}

export function StartSessionPanel({
  session,
}: {
  session: V2Session;
}) {
  const router = useRouter();
  const [reviewer, setReviewer] = useState<ReviewerState>({ name: "", email: "" });
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function handleStart() {
    setSaving(true);
    setMessage(null);
    try {
      const created = await createResearchSession({
        title: `${session.selectedCandidate.candidate} to ${session.selectedCandidate.disease}`,
        reviewer_name: reviewer.name || null,
        reviewer_email: reviewer.email || null,
        selected_entity: asRecord(session.selectedEntity),
        selected_candidate: asRecord(session.selectedCandidate),
        run_mode: session.runMode,
        score_threshold: session.scoreThreshold,
        mechanism_weight: session.mechanismWeight,
        decision: session.selectedCandidate.decision,
        notes: "",
        evidence_state: { source: "start", message: "Session started before validation." },
        provenance: [
          {
            endpoint: "/v2/start",
            source_kind: "fallback",
            notes: ["Saved from v2 Start before evidence review."],
          },
        ],
      });
      const next = {
        ...session,
        sessionId: created.id,
      };
      router.push(`/v2/experiment${buildV2Params(next)}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to create session.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-3">
      <ReviewerFields reviewer={reviewer} setReviewer={setReviewer} />
      <button
        type="button"
        onClick={handleStart}
        disabled={saving}
        className="rounded-lg border border-primary/60 bg-primary/20 px-4 py-2 text-xs font-semibold text-on-surface hover:bg-primary/30 disabled:opacity-60"
      >
        {saving ? "Starting..." : "Start saved investigation"}
      </button>
      {message ? <p className="text-xs text-error">{message}</p> : null}
    </div>
  );
}

export function ResumeSessionsPanel() {
  const [sessions, setSessions] = useState<ResearchSession[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchResearchSessions()
      .then((items) => {
        if (!cancelled) setSessions(items.slice(0, 5));
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Load failed.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <p className="text-xs text-error">{error}</p>;
  if (sessions.length === 0) {
    return <p className="text-xs text-on-surface-variant">No saved v2 sessions yet.</p>;
  }

  return (
    <div className="space-y-2">
      {sessions.map((item) => (
        <Link
          key={item.id}
          href={`/v2/validation${paramsForResearchSession(item)}`}
          className="block rounded-lg border border-outline-variant/25 bg-surface-container-high/60 p-3 hover:border-primary/40"
        >
          <div className="flex items-center justify-between gap-3">
            <p className="font-semibold text-on-surface">{item.title}</p>
            <span className="font-mono text-xs text-primary">{item.id}</span>
          </div>
          <p className="mt-1 text-xs text-on-surface-variant">
            {item.decision} · {item.reviewer_name || "unassigned reviewer"}
          </p>
        </Link>
      ))}
    </div>
  );
}

export function ValidationSessionControls({
  session,
  evidenceSnapshot,
}: {
  session: V2Session;
  evidenceSnapshot: V2EvidenceSnapshot;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [reviewer, setReviewer] = useState<ReviewerState>({ name: "", email: "" });
  const [notes, setNotes] = useState(session.selectedCandidate.workingConclusion);
  const [decision, setDecision] = useState(session.selectedCandidate.decision);
  const [sessionId, setSessionId] = useState(session.sessionId ?? "");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const exportUrl = useMemo(
    () => (sessionId ? exportEvidencePacketUrl(sessionId) : null),
    [sessionId],
  );

  async function handleSave() {
    setSaving(true);
    setMessage(null);
    const payload = {
      title: `${session.selectedCandidate.candidate} to ${session.selectedCandidate.disease}`,
      reviewer_name: reviewer.name || null,
      reviewer_email: reviewer.email || null,
      selected_entity: asRecord(session.selectedEntity),
      selected_candidate: asRecord(session.selectedCandidate),
      run_mode: session.runMode,
      score_threshold: session.scoreThreshold,
      mechanism_weight: session.mechanismWeight,
      decision,
      notes,
      evidence_state: evidenceSnapshot as unknown as Record<string, unknown>,
      provenance: evidenceSnapshot.provenance as unknown as Array<Record<string, unknown>>,
    };

    try {
      const saved = sessionId
        ? await updateResearchSession(sessionId, payload)
        : await createResearchSession(payload);
      setSessionId(saved.id);
      setMessage(`Saved ${saved.id}`);
      const updatedParams = buildV2Params({ ...session, sessionId: saved.id });
      router.replace(`${pathname}${updatedParams}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save session.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4">
      <ReviewerFields reviewer={reviewer} setReviewer={setReviewer} />
      <label className="block text-xs font-semibold uppercase tracking-widest text-on-surface-variant">
        Decision
        <select
          value={decision}
          onChange={(event) => setDecision(event.target.value)}
          className="mt-2 w-full rounded-lg border border-outline-variant/50 bg-surface-container-high px-3 py-2 normal-case tracking-normal text-on-surface"
        >
          <option>Review</option>
          <option>Keep</option>
          <option>Reject</option>
          <option>Shortlist</option>
        </select>
      </label>
      <label className="block text-xs font-semibold uppercase tracking-widest text-on-surface-variant">
        Notes
        <textarea
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          rows={5}
          className="mt-2 w-full rounded-lg border border-outline-variant/50 bg-surface-container-high px-3 py-2 normal-case tracking-normal text-on-surface"
        />
      </label>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="rounded-lg border border-primary/60 bg-primary/20 px-4 py-2 text-xs font-semibold text-on-surface hover:bg-primary/30 disabled:opacity-60"
        >
          {saving ? "Saving..." : sessionId ? "Update session" : "Save session"}
        </button>
        {exportUrl ? (
          <a
            href={exportUrl}
            className="rounded-lg border border-outline-variant/60 bg-surface-container-high px-4 py-2 text-xs font-semibold text-on-surface hover:border-primary/40"
          >
            Export evidence packet
          </a>
        ) : null}
      </div>
      {sessionId ? (
        <p className="font-mono text-xs text-primary">Current session: {sessionId}</p>
      ) : null}
      {message ? <p className="text-xs text-on-surface-variant">{message}</p> : null}
    </div>
  );
}

function ReviewerFields({
  reviewer,
  setReviewer,
}: {
  reviewer: ReviewerState;
  setReviewer: (reviewer: ReviewerState) => void;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="block text-xs font-semibold uppercase tracking-widest text-on-surface-variant">
        Reviewer name
        <input
          value={reviewer.name}
          onChange={(event) => setReviewer({ ...reviewer, name: event.target.value })}
          className="mt-2 w-full rounded-lg border border-outline-variant/50 bg-surface-container-high px-3 py-2 normal-case tracking-normal text-on-surface"
        />
      </label>
      <label className="block text-xs font-semibold uppercase tracking-widest text-on-surface-variant">
        Reviewer email
        <input
          value={reviewer.email}
          onChange={(event) => setReviewer({ ...reviewer, email: event.target.value })}
          className="mt-2 w-full rounded-lg border border-outline-variant/50 bg-surface-container-high px-3 py-2 normal-case tracking-normal text-on-surface"
        />
      </label>
    </div>
  );
}

function paramsForResearchSession(item: ResearchSession) {
  const selectedEntity = item.selected_entity as { name?: string };
  const selectedCandidate = item.selected_candidate as { disease?: string };
  return buildV2Params(
    parseV2Session({
      session_id: item.id,
      entity: selectedEntity.name,
      runMode: item.run_mode,
      candidate: selectedCandidate.disease,
    }),
  );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}
