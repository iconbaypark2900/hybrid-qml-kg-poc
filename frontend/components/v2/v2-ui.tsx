"use client";

import Link from "next/link";
import { type MouseEventHandler } from "react";
import type { V2Session } from "@/lib/v2-data";

export function PageHero({
  eyebrow,
  title,
  children,
  actions,
}: {
  eyebrow: string;
  title: string;
  children?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <header className="rounded-xl border border-outline-variant/25 bg-surface-container-lowest/80">
      <div className="flex flex-col gap-5 p-5 lg:flex-row lg:items-start lg:justify-between lg:p-6">
        <div className="max-w-5xl">
          <p className="font-label text-xs font-bold uppercase tracking-widest text-primary">
            {eyebrow}
          </p>
          <h1 className="mt-3 max-w-5xl font-headline text-3xl font-semibold leading-tight text-on-surface lg:text-4xl">
            {title}
          </h1>
          {children ? (
            <div className="mt-3 max-w-4xl text-sm leading-relaxed text-on-surface-variant">
              {children}
            </div>
          ) : null}
        </div>
        {actions ? <div className="flex shrink-0 flex-wrap gap-2">{actions}</div> : null}
      </div>
    </header>
  );
}

export function CurrentInvestigationStrip({ session }: { session: V2Session }) {
  return (
    <section className="grid gap-3 rounded-xl border border-primary/25 bg-primary/10 p-3 text-xs text-on-surface-variant md:grid-cols-[1.2fr_1fr_1fr_auto] md:items-center">
      <div>
        <p className="font-label font-bold uppercase tracking-widest text-primary">
          Current investigation
        </p>
        <p className="mt-1 text-sm font-semibold text-on-surface">
          {session.selectedCandidate.candidate} to {session.selectedCandidate.disease}
        </p>
      </div>
      <StripItem label="Starting point" value={session.selectedEntity.name} />
      <StripItem label="Run path" value={session.runMode} />
      <Link
        href={`/v2/experiment?entity=${encodeURIComponent(session.selectedEntity.name)}&runMode=${encodeURIComponent(session.runMode)}&candidate=${encodeURIComponent(session.selectedCandidate.disease)}`}
        className="rounded-lg border border-primary/50 bg-primary/15 px-3 py-2 text-center text-xs font-bold text-on-surface hover:bg-primary/25"
      >
        Continue
      </Link>
    </section>
  );
}

function StripItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="font-label text-[0.65rem] font-bold uppercase tracking-widest text-on-surface-variant">
        {label}
      </p>
      <p className="mt-1 truncate font-mono text-xs text-on-surface">{value}</p>
    </div>
  );
}

export function Card({
  title,
  kicker,
  help,
  children,
  className = "",
}: {
  title: string;
  kicker?: string;
  help?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-outline-variant/25 bg-surface-container-low/80 p-4 ${className}`}
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <h2 className="flex min-w-0 items-center gap-2 font-headline text-xl font-semibold text-on-surface">
          <span>{title}</span>
          {help ? <HelpTooltip text={help} /> : null}
        </h2>
        {kicker ? (
          <p className="shrink-0 text-xs text-on-surface-variant">{kicker}</p>
        ) : null}
      </div>
      {children}
    </section>
  );
}

export function HelpTooltip({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-primary/40 bg-primary/15 text-xs font-bold text-primary"
        aria-label={text}
      >
        ?
      </button>
      <span className="pointer-events-none absolute left-0 top-7 z-50 hidden w-72 rounded-lg border border-primary/25 bg-surface-container-highest p-3 text-left text-xs font-normal leading-relaxed text-on-surface shadow-2xl group-hover:block group-focus-within:block">
        {text}
      </span>
    </span>
  );
}

export function Chip({
  children,
  active = false,
  help,
  onClick,
}: {
  children: React.ReactNode;
  active?: boolean;
  help?: string;
  onClick?: MouseEventHandler<HTMLButtonElement>;
}) {
  const interactive = typeof onClick === "function";

  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        onClick={onClick}
        aria-pressed={interactive ? active : undefined}
        className={`rounded-full border px-3 py-2 text-xs font-semibold transition-colors ${
          active
            ? "border-tertiary/60 bg-tertiary/15 text-tertiary"
            : "border-outline-variant/60 bg-surface-container-high text-on-surface-variant hover:border-primary/40 hover:text-on-surface"
        }`}
      >
        {children}
      </button>
      {help ? (
        <span className="pointer-events-none absolute left-0 top-11 z-50 hidden w-72 rounded-lg border border-primary/25 bg-surface-container-highest p-3 text-xs font-normal leading-relaxed text-on-surface shadow-2xl group-hover:block group-focus-within:block">
          {help}
        </span>
      ) : null}
    </span>
  );
}

export function ActionLink({
  href,
  children,
  variant = "primary",
  className = "",
}: {
  href: string;
  children: React.ReactNode;
  variant?: "primary" | "secondary";
  className?: string;
}) {
  return (
    <Link
      href={href}
      className={`rounded-lg border px-4 py-2 text-xs font-semibold transition-colors ${
        variant === "primary"
          ? "border-primary/60 bg-primary/20 text-on-surface hover:bg-primary/30"
          : "border-outline-variant/60 bg-surface-container-high text-on-surface hover:border-primary/40"
      } ${className}`}
    >
      {children}
    </Link>
  );
}

export function Metric({
  label,
  value,
  detail,
  help,
}: {
  label: string;
  value: string;
  detail: string;
  help?: string;
}) {
  return (
    <div className="rounded-xl border border-outline-variant/25 bg-surface-container-high/60 p-4">
      <dt className="flex items-center gap-2 font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">
        {label}
        {help ? <HelpTooltip text={help} /> : null}
      </dt>
      <dd className="mt-3 font-headline text-2xl font-semibold text-on-surface">
        {value}
      </dd>
      <p className="mt-1 text-xs text-on-surface-variant">{detail}</p>
    </div>
  );
}

export function MetricStrip({ children }: { children: React.ReactNode }) {
  return <dl className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">{children}</dl>;
}

export function EvidenceCard({
  title,
  value,
  detail,
  tone = "success",
}: {
  title: string;
  value: string;
  detail: string;
  tone?: "success" | "warning" | "danger" | "quantum";
}) {
  return (
    <div className="rounded-xl border border-outline-variant/25 bg-surface-container-high/60 p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="font-headline text-base font-semibold text-on-surface">{title}</p>
        <StatusBadge tone={tone}>{value}</StatusBadge>
      </div>
      <p className="mt-3 text-xs leading-relaxed text-on-surface-variant">{detail}</p>
    </div>
  );
}

export function RunSummaryCard({
  model,
  relation,
  backend,
  artifact,
}: {
  model: string;
  relation: string;
  backend: string;
  artifact: string;
}) {
  return (
    <Card title="Run summary" kicker="Provenance">
      <dl className="grid gap-3 text-xs">
        <StripItem label="Model" value={model} />
        <StripItem label="Relation" value={relation} />
        <StripItem label="Backend" value={backend} />
        <StripItem label="Artifact" value={artifact} />
      </dl>
    </Card>
  );
}

export function ActionRail({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap gap-2 rounded-xl border border-outline-variant/25 bg-surface-container-low/70 p-3">
      {children}
    </div>
  );
}

export function StatusBadge({
  children,
  tone = "success",
}: {
  children: React.ReactNode;
  tone?: "success" | "warning" | "danger" | "quantum";
}) {
  const tones = {
    success: "bg-tertiary/15 text-tertiary",
    warning: "bg-[#f8c64f]/15 text-[#f8c64f]",
    danger: "bg-error/15 text-error",
    quantum: "bg-secondary/15 text-secondary",
  };
  return (
    <span className={`rounded-full px-2 py-1 text-xs font-bold ${tones[tone]}`}>
      {children}
    </span>
  );
}
