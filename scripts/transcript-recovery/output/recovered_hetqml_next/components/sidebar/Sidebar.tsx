'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { useUiStore } from '@/lib/store';
import { loadJson, saveJson, STORAGE_KEYS } from '@/lib/persistence';

const WORKFLOW_NAV = [
  { href: '/initialize', num: '01', title: 'Initialize',  sub: 'Define investigation' },
  { href: '/experiment', num: '02', title: 'Experiment',  sub: 'Produce evidence' },
  { href: '/validate',   num: '03', title: 'Validate',    sub: 'Trust the result' },
  { href: '/visualize',  num: '04', title: 'Visualize',   sub: 'Inspect visually' }
];

const SYSTEM_NAV = [
  { href: '/operations', icon: '▢', title: 'Operations' },
  { href: '/settings',   icon: '⚙', title: 'Settings' }
];

export function Sidebar() {
  const pathname = usePathname();
  const collapsed = useUiStore((s) => s.sidebarCollapsed);
  const setSidebarCollapsed = useUiStore((s) => s.setSidebarCollapsed);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);

  // On mount, load the persisted sidebar state and apply to <body>
  useEffect(() => {
    const saved = loadJson<boolean>(STORAGE_KEYS.sidebarCollapsed, false);
    setSidebarCollapsed(saved);
    setMounted(true);
    // Clean up the preload class from layout.tsx
    document.documentElement.classList.remove('preload-sidebar-collapsed');
  }, [setSidebarCollapsed]);

  // Sync body class whenever collapsed state changes; persist to localStorage
  useEffect(() => {
    if (!mounted) return;
    document.body.classList.toggle('sidebar-collapsed', collapsed);
    saveJson(STORAGE_KEYS.sidebarCollapsed, collapsed);
  }, [collapsed, mounted]);

  // Floating tooltip for collapsed-mode hover
  function showTip(e: React.MouseEvent<HTMLAnchorElement>, label: string) {
    if (!collapsed || !tooltipRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    tooltipRef.current.textContent = label;
    tooltipRef.current.style.left = `${rect.right + 12}px`;
    tooltipRef.current.style.top = `${rect.top + rect.height / 2 - 14}px`;
    tooltipRef.current.classList.add('show');
  }
  function hideTip() {
    tooltipRef.current?.classList.remove('show');
  }

  return (
    <>
      <aside className="sidebar">
        <button
          className="sidebar-toggle"
          onClick={toggleSidebar}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label="Toggle sidebar"
        >
          ‹
        </button>

        <div className="brand">
          <div className="brand-icon">◎</div>
          <div>
            <div className="brand-name">Hetionet · QML</div>
            <div className="brand-ver">v0.7.2</div>
          </div>
        </div>

        <div className="section-label">Workflow</div>
        {WORKFLOW_NAV.map((n) => (
          <Link
            key={n.href}
            href={n.href}
            className={'nav-item ' + (pathname === n.href ? 'active' : '')}
            onMouseEnter={(e) => showTip(e, `${n.title} · ${n.sub}`)}
            onMouseLeave={hideTip}
          >
            <span className="nav-num">{n.num}</span>
            <div className="nav-text">
              <div className="nav-title">{n.title}</div>
              <div className="nav-sub">{n.sub}</div>
            </div>
          </Link>
        ))}

        <div className="section-label">System</div>
        {SYSTEM_NAV.map((n) => (
          <Link
            key={n.href}
            href={n.href}
            className={'nav-item ' + (pathname === n.href ? 'active' : '')}
            onMouseEnter={(e) => showTip(e, n.title)}
            onMouseLeave={hideTip}
          >
            <span className="nav-icon">{n.icon}</span>
            <div className="nav-text">
              <div className="nav-title">{n.title}</div>
            </div>
          </Link>
        ))}

        <div className="section-label">Appearance</div>
        <div className="theme-row">
          <div className="theme-btn">☼ Light</div>
          <div className="theme-btn">☾ Dark</div>
          <div className="theme-btn active">▢ Auto</div>
        </div>

        <div className="status-bar">
          <span className="status-dot" />
          <span className="mono">ibm_torino</span>
          <div style={{ color: 'var(--faint)', fontSize: 10, marginTop: 4 }}>
            queue: 2 · last verified 14:32 UTC
          </div>
        </div>
      </aside>
      <div className="sidebar-floating-tooltip" ref={tooltipRef} />
    </>
  );
}
