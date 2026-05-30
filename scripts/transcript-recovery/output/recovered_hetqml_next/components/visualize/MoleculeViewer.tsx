'use client';

// 3Dmol.js wrapper. The package isn't strictly typed; we treat its API
// as `any` and import lazily. Without a real PDB/SMILES feed we render a
// labeled placeholder card and document where the real data should
// plug in.

import { useEffect, useRef } from 'react';
import type { Compound } from '@/data/types';
import { COMPOUND_CONTEXT } from '@/data/compoundContext';

const PUBCHEM_BASE = 'https://pubchem.ncbi.nlm.nih.gov/compound';

export function MoleculeViewer({ compound }: { compound: Compound }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let viewer: any | null = null;
    let cancelled = false;
    (async () => {
      if (typeof window === 'undefined' || !ref.current) return;
      try {
        const lib: any = await import('3dmol');
        if (cancelled || !ref.current) return;
        viewer = lib.createViewer(ref.current, { backgroundColor: '#0a0a0c' });
        // No real coordinates wired yet — drop a labeled cube as a placeholder
        // so the surface shows that 3Dmol is loaded and rendering.
        viewer.setStyle({}, { sphere: { radius: 1.0, color: '0x06d6a0' } });
        viewer.zoomTo();
        viewer.render();
      } catch (e) {
        // 3Dmol isn't available or import failed — leave the placeholder div.
        // eslint-disable-next-line no-console
        console.warn('3Dmol load failed', e);
      }
    })();
    return () => {
      cancelled = true;
      if (viewer && typeof viewer.removeAllSurfaces === 'function') {
        try { viewer.removeAllSurfaces(); } catch { /* noop */ }
      }
    };
  }, [compound.id]);

  const ctx = COMPOUND_CONTEXT[compound.id];
  const cid = compound.pubchemCID;

  return (
    <>
      <div
        ref={ref}
        role="img"
        aria-label={`Molecular view of ${compound.name}`}
        style={{
          width: '100%', height: 220,
          background: '#0a0a0c',
          borderRadius: 4,
          position: 'relative',
        }}
      />
      <div style={{ marginTop: 12, fontSize: 11, color: 'var(--muted)', lineHeight: 1.6 }}>
        <div style={{ fontFamily: 'monospace', fontSize: 10, color: 'var(--faint)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          MECHANISM
        </div>
        <div>{compound.mech}</div>
        {ctx ? (
          <>
            <div style={{ marginTop: 8, fontFamily: 'monospace', fontSize: 10, color: 'var(--faint)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              APPROVAL
            </div>
            <div>{ctx.approval}</div>
          </>
        ) : null}
        {cid ? (
          <div style={{ marginTop: 8, fontSize: 11 }}>
            <a href={`${PUBCHEM_BASE}/${cid}`} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--teal)' }}>
              View on PubChem (CID {cid}) ↗
            </a>
          </div>
        ) : (
          <div style={{ marginTop: 8, fontSize: 10, color: 'var(--faint)' }}>
            No PubChem CID on file. Add one to data/compounds.ts to enable an
            external structure link.
          </div>
        )}
      </div>
    </>
  );
}
