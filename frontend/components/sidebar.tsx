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
    <nav className="flex flex-col gap-6" aria-label="Primary">
      {navSections.map((section) => (
        <div key={section.heading}>
          <p className="mb-2 font-label text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
            {section.heading}
          </p>
          <ul className="flex flex-col gap-0.5">
            {section.items.map((item) => {
              const active = isActive(pathname, item.href);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors ${
                      active
                        ? "bg-surface-container-highest text-primary"
                        : "text-on-surface hover:bg-surface-container/80"
                    }`}
                  >
                    <span
                      className="material-symbols-outlined text-lg opacity-80"
                      aria-hidden
                    >
                      {iconFor(item.href)}
                    </span>
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}

function iconFor(href: string): string {
  const map: Record<string, string> = {
    "/experiments": "monitoring",
    "/system": "dns",
    "/knowledge-graph": "hub",
    "/quantum": "blur_on",
    "/simulation": "tune",
    "/simulation/parameters": "settings",
    "/molecular-design": "science",
    "/hypotheses/new": "lightbulb",
    "/analysis/drug-delivery": "medication",
    "/analysis/next-steps": "arrow_forward",
    "/export": "download",
  };
  return map[href] ?? "chevron_right";
}
