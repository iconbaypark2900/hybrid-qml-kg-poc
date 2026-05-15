"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { navSections } from "@/lib/nav";

function isActive(pathname: string, href: string): boolean {
  if (pathname === href) return true;
  if (href !== "/" && pathname.startsWith(`${href}/`)) return true;
  return false;
}

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="journey-rail-inner">
      <div className="brand">
        <Link href="/" className="flex min-w-0 items-center gap-3">
          <span className="mark">
            QG
          </span>
          <span className="brand-copy min-w-0">
            <span className="block truncate font-headline text-base font-bold text-on-surface">
              Hybrid QML-KG
            </span>
            <span className="block truncate text-xs text-on-surface-variant">
              Four page UI mockup
            </span>
          </span>
        </Link>
      </div>

      <nav className="flex flex-col gap-6" aria-label="Primary">
        {navSections.map((section) => (
          <div key={section.heading}>
            <p className="nav-label mb-2 font-label text-xs font-semibold uppercase tracking-[0.16em] text-on-surface-variant">
              {section.heading}
            </p>
            <ul className="flex flex-col gap-1">
              {section.items.map((item) => {
                const active = isActive(pathname, item.href);
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      title={item.label}
                      data-short={shortFor(item.href)}
                      className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                        active ? "active" : ""
                      }`}
                    >
                      <span
                        className="material-symbols-outlined text-lg opacity-90"
                        aria-hidden
                      >
                        {iconFor(item.href)}
                      </span>
                      <span className="nav-text">{item.label}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      <p className="nav-note mt-8 border-t border-outline/15 pt-5 text-xs leading-relaxed text-on-surface-variant">
        Proposed order: Start, Experiment, Validation, Visual.
      </p>
    </div>
  );
}

function shortFor(href: string): string {
  const map: Record<string, string> = {
    "/": "S",
    "/experiments": "E",
    "/validation": "V",
    "/visualization": "3D",
    "/settings": "Cfg",
  };
  return map[href] ?? ">";
}

function iconFor(href: string): string {
  const map: Record<string, string> = {
    "/": "home",
    "/experiments": "experiment",
    "/validation": "fact_check",
    "/visualization": "view_in_ar",
    "/settings": "settings",
    "/predict": "biotech",
    // Legacy/deep links
    "/analysis/drug-delivery": "medication",
    "/hypotheses/new": "format_list_numbered",
    "/simulation": "play_circle",
    "/simulation/parameters": "tune",
    "/system": "dns",
    "/knowledge-graph": "hub",
    "/quantum": "blur_on",
    "/molecular-design": "science",
    "/export": "download",
    "/analysis/next-steps": "arrow_forward",
  };
  return map[href] ?? "chevron_right";
}
