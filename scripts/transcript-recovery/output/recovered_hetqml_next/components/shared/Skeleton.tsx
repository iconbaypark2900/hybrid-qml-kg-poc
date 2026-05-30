'use client';

import type { CSSProperties, ReactNode } from 'react';

// Loading skeleton primitives. Use these in place of bespoke "loading..."
// strings so loading states are consistent across pages.
//
//   <Skeleton.Bar width={120} />
//   <Skeleton.Block height={80} />
//   <Skeleton.Panel title="Active model">
//     <Skeleton.Bar width="60%" />
//     <Skeleton.Bar width="40%" />
//   </Skeleton.Panel>

const PULSE: CSSProperties = {
  background: 'linear-gradient(90deg, var(--card) 0%, var(--paper-alt) 50%, var(--card) 100%)',
  backgroundSize: '200% 100%',
  animation: 'hetqml-skeleton-pulse 1.6s ease-in-out infinite',
  borderRadius: 3,
};

function Bar({
  width = '100%', height = 12, style,
}: { width?: number | string; height?: number; style?: CSSProperties }) {
  return (
    <div
      aria-hidden="true"
      style={{
        ...PULSE,
        width: typeof width === 'number' ? `${width}px` : width,
        height,
        marginBottom: 8,
        ...style,
      }}
    />
  );
}

function Block({
  height = 48, style,
}: { height?: number; style?: CSSProperties }) {
  return (
    <div
      aria-hidden="true"
      style={{
        ...PULSE,
        height,
        marginBottom: 12,
        ...style,
      }}
    />
  );
}

function Panel({
  title, children,
}: { title?: string; children: ReactNode }) {
  return (
    <div className="panel" aria-busy="true">
      <div className="panel-head">
        <div>
          <div className="eyebrow" style={{ opacity: 0.6 }}>LOADING</div>
          <div className="panel-title">{title ?? <Bar width={140} height={16} />}</div>
        </div>
      </div>
      <div>{children}</div>
    </div>
  );
}

export const Skeleton = { Bar, Block, Panel };
