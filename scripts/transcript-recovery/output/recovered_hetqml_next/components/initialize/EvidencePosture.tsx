'use client';

import { useGuardsStore } from '@/lib/store';
import type { IntegrityGuard } from '@/data/types';
import { useMemo, useState } from 'react';

type FilterKey = 'all' | 'critical' | 'recommended' | 'optional' | 'off';

export function EvidencePosture() {
  const guards = useGuardsStore((s) => s.guards);
  const toggleGuard = useGuardsStore((s) => s.toggleGuard);
  const [filter, setFilter] = useState<FilterKey>('all');
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['Bias / equity', 'Quantum integrity']));

  const summary = useMemo(() => {
    const total = guards.length;
    const enabled = guards.filter((g) => g.enabled).length;
    const criticalTotal = guards.filter((g) => g.level === 'critical').length;
    const criticalOff = guards.filter((g) => g.level === 'critical' && !g.enabled).length;
    return { total, enabled, criticalTotal, criticalOff };
  }, [guards]);

  const groups = useMemo(() => {
    const out: Record<string, IntegrityGuard[]> = {};
    guards.forEach((g) => {
      if (!out[g.group]) out[g.group] = [];
      out[g.group]!.push(g);
    });
    return out;
  }, [guards]);

  const isVisible = (g: IntegrityGuard) => {
    if (filter === 'all') return true;
    if (filter === 'off') return !g.enabled;
    return g.level === filter;
  };

  const compromised = summary.criticalOff > 0;

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">TOOL · EVIDENCE POSTURE</div>
          <div className="panel-title">Integrity guards</div>
        </div>
        <span className="badge">Reference</span>
      </div>
      <p className="panel-purpose">
        Every switch that governs an evidence claim downstream. State is inspectable —
        running with any critical guard off produces a different kind of result and must
        be visible to anyone reading a report.
      </p>

      {/* Posture summary banner */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 10,
        padding: 12,
        background: 'var(--card)',
        border: `1px solid ${compromised ? 'var(--sienna)' : 'var(--border-soft)'}`,
        borderRadius: 4,
        marginBottom: 12,
        ...(compromised ? { background: 'var(--sienna-bg)' } : {})
      }}>
        <SummaryStat num={`${summary.enabled}/${summary.total}`} label="guards enabled" color="green" />
        <SummaryStat num={`${summary.criticalTotal - summary.criticalOff}/${summary.criticalTotal}`} label="critical on" color={summary.criticalOff === 0 ? 'green' : 'sienna'} />
        <SummaryStat num={String(summary.criticalOff)} label="critical off" color={summary.criticalOff > 0 ? 'sienna' : 'green'} />
        <SummaryStat num={compromised ? 'COMPROMISED' : 'PASSING'} label="posture status" color={compromised ? 'sienna' : 'green'} small />
      </div>

      {/* Filter pills */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 12 }}>
        {(['all', 'critical', 'recommended', 'optional', 'off'] as const).map((f) => (
          <span
            key={f}
            onClick={() => setFilter(f)}
            style={{
              fontSize: 10,
              padding: '3px 9px',
              borderRadius: 999,
              background: filter === f ? 'var(--teal)' : 'var(--card)',
              border: `1px solid ${filter === f ? 'var(--teal)' : 'var(--border-soft)'}`,
              color: filter === f ? 'var(--paper)' : 'var(--muted)',
              cursor: 'pointer',
              fontFamily: 'monospace',
              textTransform: 'lowercase'
            }}
          >
            {f === 'off' ? 'off only' : f}
          </span>
        ))}
      </div>

      {/* Groups */}
      {Object.entries(groups).map(([groupName, list]) => {
        const visibleList = list.filter(isVisible);
        if (visibleList.length === 0 && filter !== 'all') return null;
        const onCount = list.filter((g) => g.enabled).length;
        const offCriticalCount = list.filter((g) => !g.enabled && g.level === 'critical').length;
        const isExpanded = expanded.has(groupName) || (offCriticalCount > 0 || filter !== 'all');

        return (
          <div key={groupName} style={{ marginBottom: 6, border: '1px solid var(--border-soft)', borderRadius: 4, overflow: 'hidden' }}>
            <div
              onClick={() => {
                const next = new Set(expanded);
                if (next.has(groupName)) next.delete(groupName);
                else next.add(groupName);
                setExpanded(next);
              }}
              style={{
                padding: '8px 12px',
                background: 'var(--card)',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                userSelect: 'none'
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--ink)', display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ fontSize: 9, color: 'var(--faint)', display: 'inline-block', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>▶</span>
                {groupName}
              </div>
              <div style={{ display: 'flex', gap: 10, fontSize: 10, fontFamily: 'monospace', color: 'var(--faint)' }}>
                <span style={{ color: 'var(--green)' }}>{onCount} of {list.length} on</span>
                {offCriticalCount > 0 && <span style={{ color: 'var(--sienna)' }}>{offCriticalCount} critical off</span>}
              </div>
            </div>
            {isExpanded && (
              <div>
                {visibleList.map((g) => (
                  <GuardRow key={g.id} guard={g} onToggle={() => toggleGuard(g.id)} />
                ))}
              </div>
            )}
          </div>
        );
      })}

      <div className="panel-footer">
        <span>config/integrity.yaml · pipeline/preflight.py</span>
        <span><em>{compromised ? `${summary.criticalOff} critical guard${summary.criticalOff > 1 ? 's' : ''} off · audit blocked` : `${summary.enabled} of ${summary.total} enabled · all critical on`}</em></span>
      </div>
    </div>
  );
}

function SummaryStat({ num, label, color, small = false }: { num: string; label: string; color: 'green' | 'sienna'; small?: boolean }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontFamily: small ? '-apple-system, sans-serif' : 'Georgia, serif',
        fontSize: small ? 14 : 22,
        fontWeight: 600,
        color: `var(--${color})`,
        lineHeight: 1.1,
        paddingTop: small ? 6 : 0
      }}>
        {num}
      </div>
      <div style={{ fontSize: 10, color: 'var(--faint)', textTransform: 'uppercase', letterSpacing: '0.5px', marginTop: 2, fontFamily: 'monospace' }}>
        {label}
      </div>
    </div>
  );
}

function GuardRow({ guard, onToggle }: { guard: IntegrityGuard; onToggle: () => void }) {
  const isCriticalOff = !guard.enabled && guard.level === 'critical';
  const levelColor = guard.level === 'critical' ? 'var(--sienna)' : guard.level === 'recommended' ? 'var(--amber)' : 'var(--faint)';
  const levelBg = guard.level === 'critical' ? 'var(--sienna-bg)' : guard.level === 'recommended' ? 'var(--amber-bg)' : 'var(--paper-alt)';

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '32px 1fr 80px 70px',
      gap: 12,
      padding: 12,
      borderTop: '1px solid var(--border-soft)',
      alignItems: 'flex-start',
      fontSize: 12,
      background: isCriticalOff ? 'rgba(224, 132, 116, 0.04)' : 'transparent'
    }}>
      <div
        onClick={onToggle}
        style={{
          width: 28,
          height: 16,
          background: guard.enabled ? 'var(--green)' : (isCriticalOff ? 'var(--sienna)' : 'var(--border)'),
          borderRadius: 999,
          position: 'relative',
          cursor: 'pointer',
          flexShrink: 0
        }}
      >
        <div style={{
          position: 'absolute',
          top: 2,
          left: guard.enabled ? 14 : 2,
          width: 12, height: 12,
          background: guard.enabled ? 'var(--paper)' : (isCriticalOff ? 'var(--paper)' : 'var(--ink)'),
          borderRadius: '50%',
          transition: 'left 0.2s'
        }} />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 600, color: 'var(--ink)', fontSize: 12 }}>{guard.name}</span>
          <span style={{
            fontSize: 9,
            padding: '1px 6px',
            borderRadius: 3,
            fontFamily: 'monospace',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            background: levelBg,
            color: levelColor,
            border: `1px solid ${levelColor}`
          }}>
            {guard.level}
          </span>
        </div>
        <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.5 }}>{guard.desc}</div>
        {!guard.enabled && (
          <div style={{
            fontSize: 10,
            color: 'var(--sienna)',
            lineHeight: 1.45,
            padding: '6px 10px',
            background: 'var(--sienna-bg)',
            borderLeft: '2px solid var(--sienna)',
            borderRadius: 3,
            marginTop: 4
          }}>
            <strong>If off:</strong> {guard.impact}
          </div>
        )}
        <div style={{ fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace', wordBreak: 'break-all' }}>
          {guard.source}
        </div>
      </div>

      <div />

      <div style={{ fontSize: 11, fontFamily: 'monospace', textAlign: 'right', color: guard.enabled ? 'var(--green)' : 'var(--sienna)' }}>
        {guard.enabled ? '● ON' : '○ OFF'}
      </div>
    </div>
  );
}
