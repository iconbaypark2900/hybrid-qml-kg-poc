"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useState } from "react";
import { CurrentInvestigationStrip } from "@/components/v2/v2-ui";
import { parseV2Session } from "@/lib/v2-data";
import { v2NavItems } from "@/lib/v2-nav";

export {
  ActionLink,
  ActionRail,
  Card,
  Chip,
  CurrentInvestigationStrip,
  EvidenceCard,
  HelpTooltip,
  Metric,
  MetricStrip,
  PageHero,
  RunSummaryCard,
  StatusBadge,
} from "@/components/v2/v2-ui";

export function V2Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [collapsed, setCollapsed] = useState(false);
  const query = searchParams.toString();
  const hrefFor = (href: string) => (query ? `${href}?${query}` : href);
  const session = parseV2Session(searchParams);

  return (
    <div className="fixed inset-0 z-[100] flex flex-col overflow-hidden bg-background text-on-surface md:flex-row">
      <header className="border-b border-outline-variant/30 bg-surface-container-low px-4 py-3 md:hidden">
        <div className="flex items-center justify-between gap-3">
          <Brand compact />
          <span className="rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
            Research cockpit
          </span>
        </div>
        <nav className="mt-3 flex gap-2 overflow-x-auto pb-1" aria-label="V2 mobile pages">
          {v2NavItems.map((page) => {
            const active =
              pathname === page.href || pathname.startsWith(`${page.href}/`);
            return (
              <Link
                key={page.href}
                href={hrefFor(page.href)}
                className={`shrink-0 rounded-lg border px-3 py-2 text-xs font-semibold transition-colors ${
                  active
                    ? "border-primary/50 bg-primary/20 text-primary"
                    : "border-outline-variant/50 bg-surface-container-high text-on-surface"
                }`}
              >
                {page.label}
              </Link>
            );
          })}
        </nav>
      </header>
      <aside
        className={`hidden shrink-0 border-r border-outline-variant/30 bg-surface-container-low transition-all duration-200 md:block ${
          collapsed ? "w-[76px]" : "w-[292px]"
        }`}
      >
        <div className="flex h-full flex-col p-4">
          <Brand collapsed={collapsed} />

          <button
            type="button"
            onClick={() => setCollapsed((value) => !value)}
            className="mt-7 flex h-10 items-center justify-center rounded-lg border border-outline-variant/40 bg-surface-container-high text-sm text-on-surface-variant hover:border-primary/50 hover:text-primary"
            aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
          >
            <span className="material-symbols-outlined text-lg" aria-hidden>
              {collapsed ? "keyboard_double_arrow_right" : "keyboard_double_arrow_left"}
            </span>
            {!collapsed ? <span className="ml-2">Collapse</span> : null}
          </button>

          <nav className="mt-8" aria-label="V2 pages">
            {!collapsed ? (
              <p className="mb-3 px-2 font-label text-xs font-semibold uppercase tracking-widest text-on-surface-variant">
                Research workflow
              </p>
            ) : null}
            <ul className="space-y-2">
              {v2NavItems.map((page) => {
                const active =
                  pathname === page.href || pathname.startsWith(`${page.href}/`);
                return (
                  <li key={page.href}>
                    <Link
                      href={hrefFor(page.href)}
                      className={`group flex min-h-11 items-center rounded-lg px-3 text-sm transition-colors ${
                        active
                          ? "bg-surface-container-highest text-primary"
                          : "text-on-surface-variant hover:bg-surface-container hover:text-on-surface"
                      } ${collapsed ? "justify-center" : "gap-3"}`}
                      title={page.label}
                    >
                      <span
                        className="material-symbols-outlined text-lg"
                        aria-hidden
                      >
                        {page.icon}
                      </span>
                      {collapsed ? (
                        <span className="text-[0.65rem] font-semibold">{page.short}</span>
                      ) : (
                        <span className="min-w-0">
                          <span className="block font-semibold">{page.label}</span>
                          <span className="block truncate text-[0.68rem] text-on-surface-variant group-hover:text-on-surface-variant">
                            {page.description}
                          </span>
                        </span>
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>

          {!collapsed ? (
            <p className="mt-auto border-t border-outline-variant/30 pt-5 text-xs leading-relaxed text-on-surface-variant">
              Initialize, experiment, validate, and visualize each hypothesis before
              treating it as evidence. Operations keeps jobs and IBM Runtime visible.
            </p>
          ) : null}
        </div>
      </aside>

      <main className="min-h-0 min-w-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[1680px] space-y-5 p-4 sm:p-5 lg:space-y-6 lg:p-7">
          <div className="hidden flex-wrap items-center justify-between gap-3 rounded-xl border border-outline-variant/25 bg-surface-container-low/70 px-4 py-3 md:flex">
            <p className="text-xs text-on-surface-variant">
              Research cockpit: define a question, inspect model evidence, validate trust,
              and keep operational state visible.
            </p>
            <div className="flex flex-wrap gap-2">
              {v2NavItems.map((page) => {
                const active =
                  pathname === page.href || pathname.startsWith(`${page.href}/`);
                return (
                  <Link
                    key={page.href}
                    href={hrefFor(page.href)}
                    className={`rounded-lg border px-3 py-2 text-xs font-semibold transition-colors ${
                      active
                        ? "border-primary/50 bg-primary/20 text-primary"
                        : "border-outline-variant/50 bg-surface-container-high text-on-surface hover:border-primary/40"
                    }`}
                  >
                    {page.label}
                  </Link>
                );
              })}
              <Link
                href="/v2/settings"
                className="rounded-lg border border-outline-variant/50 bg-surface-container-high px-3 py-2 text-xs font-semibold text-on-surface hover:border-primary/40"
              >
                Settings
              </Link>
            </div>
          </div>
          <CurrentInvestigationStrip session={session} />
          {children}
        </div>
      </main>
    </div>
  );
}

function Brand({
  collapsed = false,
  compact = false,
}: {
  collapsed?: boolean;
  compact?: boolean;
}) {
  return (
    <div className="flex min-w-0 items-center gap-3">
      <div
        className={`flex shrink-0 items-center justify-center rounded-xl border border-primary/40 bg-primary/15 font-headline font-bold text-primary ${
          compact ? "h-9 w-9 text-xs" : "h-10 w-10 text-sm"
        }`}
      >
        QG
      </div>
      {!collapsed ? (
        <div className="min-w-0">
          <p className="truncate font-headline text-base font-semibold text-on-surface">
            QGG Research Cockpit
          </p>
          <p className="truncate text-xs text-on-surface-variant">
            Hybrid QML-KG evidence workflow
          </p>
        </div>
      ) : null}
    </div>
  );
}

