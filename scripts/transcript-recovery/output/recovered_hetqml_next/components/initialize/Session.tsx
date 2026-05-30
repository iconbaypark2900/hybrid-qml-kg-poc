'use client';

import { useEffect, useState } from 'react';
import { useInvestigationStore, useGuardsStore } from '@/lib/store';
import { loadJson, saveJson, STORAGE_KEYS } from '@/lib/persistence';

interface SavedSession {
  id: string;
  ts: number;
  reviewer: string;
  name: string;
  compoundId: string;
  compoundName: string;
  diseaseId: string;
  diseaseName: string;
  geneSymbol: string;
  metaedge: string;
  runPath: string;
  guardsCompromised: number;
}

const SESSION_MAX = 12;

function fmtRel(ts: number): string {
  const ms = Date.now() - ts;
  const s = Math.floor(ms / 1000);
  if (s < 30) return 'just now';
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export function Session() {
  const { compound, disease, gene, metaedge, runPath, setCompound, setDisease, setGene, setMetaedge, setRunPath } = useInvestigationStore();
  const guards = useGuardsStore((s) => s.guards);
  const [sessions, setSessions] = useState<SavedSession[]>([]);
  const [search, setSearch] = useState('');

  // Hydrate from localStorage on mount
  useEffect(() => {
    setSessions(loadJson<SavedSession[]>(STORAGE_KEYS.sessions, []));
  }, []);

  function captureSession(): SavedSession {
    const offCritical = guards.filter((g) => !g.enabled && g.level === 'critical').length;
    return {
      id: 'sess_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      ts: Date.now(),
      reviewer: 'J. Anderson',
      name: `${compound.name} → ${disease.name}`,
      compoundId: compound.id,
      compoundName: compound.name,
      diseaseId: disease.id,
      diseaseName: disease.name,
      geneSymbol: gene.symbol,
      metaedge: metaedge.code,
      runPath,
      guardsCompromised: offCritical
    };
  }

  function saveSnapshot() {
    const snap = captureSession();
    if (sessions[0] && sessions[0].compoundId === snap.compoundId && sessions[0].diseaseId === snap.diseaseId && sessions[0].runPath === snap.runPath) {
      return; // dedupe
    }
    const next = [snap, ...sessions].slice(0, SESSION_MAX);
    setSessions(next);
    saveJson(STORAGE_KEYS.sessions, next);
  }

  function deleteSession(id: string) {
    const next = sessions.filter((s) => s.id !== id);
    setSessions(next);
    saveJson(STORAGE_KEYS.sessions, next);
  }

  function resumeSession(s: SavedSession) {
    // Lazy-import data so we don't pull all entities into the store eagerly
    import('@/data/compounds').then(({ HETIONET_COMPOUNDS }) => {
      const c = HETIONET_COMPOUNDS.find((x) => x.id === s.compoundId);
      if (c) setCompound(c);
    });
    import('@/data/diseases').then(({ HETIONET_DISEASES }) => {
      const d = HETIONET_DISEASES.find((x) => x.id === s.diseaseId);
      if (d) setDisease(d);
    });
    import('@/data/genes').then(({ HETIONET_GENES }) => {
      const g = HETIONET_GENES.find((x) => x.symbol === s.geneSymbol);
      if (g) setGene(g);
    });
    import('@/data/metaedges').then(({ HETIONET_METAEDGES }) => {
      const m = HETIONET_METAEDGES.find((x) => x.code === s.metaedge);
      if (m) setMetaedge(m);
    });
    setRunPath(s.runPath as 'classical' | 'hybrid' | 'quantum');
  }

  function clearAll() {
    if (!confirm('Delete all saved sessions? This cannot be undone.')) return;
    setSessions([]);
    saveJson(STORAGE_KEYS.sessions, []);
  }

  const matchesCurrent = (s: SavedSession) =>
    s.compoundId === compound.id && s.diseaseId === disease.id && s.runPath === runPath;

  const visibleSessions = sessions.filter((s) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return s.name.toLowerCase().includes(q) || s.compoundName.toLowerCase().includes(q) || s.diseaseName.toLowerCase().includes(q);
  });

  const offCritical = guards.filter((g) => !g.enabled && g.level === 'critical').length;

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · SESSION</div>
          <div className="panel-title">Save or resume</div>
        </div>
        <span className="badge">Action</span>
      </div>
      <p className="panel-purpose">
        Investigations are reusable artifacts. Save a snapshot of the four parameters + run path
        + integrity posture, then restore it later or share with a collaborator. State persists in
        the browser via localStorage.
      </p>

      {/* Current state card */}
      <div style={{
        padding: '10px 12px',
        background: 'var(--card)',
        border: '1px solid var(--border-soft)',
        borderLeft: '3px solid var(--teal)',
        borderRadius: 4,
        marginBottom: 12,
        fontSize: 11
      }}>
        <div style={{ fontSize: 9, color: 'var(--gold)', textTransform: 'uppercase', letterSpacing: '0.5px', fontFamily: 'monospace', marginBottom: 4 }}>
          CURRENT INVESTIGATION (UNSAVED)
        </div>
        <div style={{ color: 'var(--muted)', lineHeight: 1.5 }}>
          <span style={{ color: 'var(--ink)', fontWeight: 500 }}>{compound.name}</span>{' '}
          <span style={{ color: 'var(--gold)', fontFamily: 'monospace', fontSize: 10 }}>{compound.id}</span>{' '}
          → <span style={{ color: 'var(--ink)', fontWeight: 500 }}>{disease.name}</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <SmallPill>{metaedge.code}</SmallPill>
          <SmallPill>anchor: {gene.symbol}</SmallPill>
          <SmallPill color="var(--teal)">{runPath}</SmallPill>
          {offCritical > 0 && <SmallPill color="var(--sienna)">{offCritical} critical off</SmallPill>}
        </div>
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
        <button className="btn-primary" onClick={saveSnapshot} style={{ flex: 1, justifyContent: 'center' }}>
          ▢ Save snapshot
        </button>
        <button className="btn" onClick={() => sessions[0] && resumeSession(sessions[0])} title="Resume most recent session">↻ Quick resume</button>
      </div>

      {/* Search + clear */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        <input
          type="text"
          className="input"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter saved sessions..."
          style={{ flex: 1, fontSize: 12 }}
        />
        <button className="btn" onClick={clearAll} style={{ color: 'var(--sienna)' }}>Clear all</button>
      </div>

      {/* List */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span className="section-label" style={{ margin: 0 }}>RECENT ({sessions.length})</span>
      </div>
      {visibleSessions.length === 0 ? (
        <p className="session-empty">
          {search ? `No sessions match "${search}".` : 'No saved sessions yet. Tap "Save snapshot" to start a history.'}
        </p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 280, overflowY: 'auto' }}>
          {visibleSessions.map((s) => {
            const isActive = matchesCurrent(s);
            return (
              <div
                key={s.id}
                style={{
                  padding: '10px 12px',
                  background: 'var(--card)',
                  border: `1px solid ${isActive ? 'var(--teal)' : 'var(--border-soft)'}`,
                  borderRadius: 4,
                  cursor: 'pointer',
                  ...(isActive ? { background: 'var(--teal-light)' } : {})
                }}
                onClick={() => resumeSession(s)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{s.name}</span>
                  <span style={{ fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace' }}>{fmtRel(s.ts)}</span>
                </div>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 6 }}>
                  <SmallPill color="var(--gold)">{s.compoundId}</SmallPill>
                  <SmallPill color="var(--sienna)">{s.diseaseId}</SmallPill>
                  <SmallPill color="var(--purple)">{s.geneSymbol}</SmallPill>
                  <SmallPill>{s.metaedge}</SmallPill>
                  <SmallPill color="var(--teal)">{s.runPath}</SmallPill>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace' }}>{s.reviewer}</span>
                  <button className="btn" onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }} style={{ fontSize: 10, padding: '2px 8px', color: 'var(--sienna)' }}>delete</button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="panel-footer">
        <span>localStorage://hetqml.sessions</span>
        <span><em>persisted locally</em></span>
      </div>
    </div>
  );
}

function SmallPill({ children, color }: { children: React.ReactNode; color?: string }) {
  return (
    <span style={{
      fontSize: 10,
      padding: '1px 6px',
      borderRadius: 3,
      background: 'var(--paper-alt)',
      color: color ?? 'var(--muted)',
      border: `1px solid ${color ? color : 'var(--border-soft)'}`,
      fontFamily: 'monospace'
    }}>
      {children}
    </span>
  );
}
