import Link from "next/link";
import type { ReactNode } from "react";

export type JourneySetup = {
  topic: string;
  kind: string;
  mode: "classical" | "hybrid" | "quantum";
  score: string;
  mechanism: string;
  targets: boolean;
  pathways: boolean;
  weakLinks: boolean;
  drug: string;
  disease: string;
};

export const defaultSetup: JourneySetup = {
  topic: "Parkinson disease",
  kind: "Disease",
  mode: "hybrid",
  score: "0.65",
  mechanism: "high",
  targets: true,
  pathways: true,
  weakLinks: false,
  drug: "Nilotinib",
  disease: "Parkinson disease",
};

export function readJourneySetup(params: URLSearchParams): JourneySetup {
  const topic = params.get("topic") || defaultSetup.topic;
  const modeParam = params.get("mode");
  const mode =
    modeParam === "classical" || modeParam === "quantum" ? modeParam : "hybrid";
  return {
    topic,
    kind: params.get("kind") || defaultSetup.kind,
    mode,
    score: params.get("score") || defaultSetup.score,
    mechanism: params.get("mechanism") || defaultSetup.mechanism,
    targets: params.get("targets") !== "0" && params.get("targets") !== "false",
    pathways: params.get("pathways") !== "0" && params.get("pathways") !== "false",
    weakLinks: params.get("weak_links") === "1" || params.get("weak_links") === "true",
    drug: params.get("drug") || defaultSetup.drug,
    disease: params.get("disease") || topic,
  };
}

export function journeyQuery(setup: JourneySetup): string {
  const params = new URLSearchParams({
    topic: setup.topic,
    kind: setup.kind,
    mode: setup.mode,
    score: setup.score,
    mechanism: setup.mechanism,
    targets: setup.targets ? "1" : "0",
    pathways: setup.pathways ? "1" : "0",
    weak_links: setup.weakLinks ? "1" : "0",
    drug: setup.drug,
    disease: setup.disease,
  });
  return params.toString();
}

export function titleCase(value: string): string {
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function HelpTip({ text }: { text: string }) {
  return (
    <span className="journey-help" tabIndex={0} aria-label={text}>
      ?
      <span role="tooltip">{text}</span>
    </span>
  );
}

export function PageHero({
  eyebrow,
  title,
  body,
  actions,
}: {
  eyebrow: string;
  title: string;
  body?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="journey-hero">
      <div className="max-w-5xl">
        <p className="journey-eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        {body ? <p>{body}</p> : null}
      </div>
      {actions ? <div className="journey-hero-actions">{actions}</div> : null}
    </header>
  );
}

export function JourneyButton({
  href,
  children,
  tone = "default",
}: {
  href: string;
  children: ReactNode;
  tone?: "default" | "primary";
}) {
  return (
    <Link className={`journey-button ${tone === "primary" ? "primary" : ""}`} href={href}>
      {children}
    </Link>
  );
}

export function JourneyCard({
  title,
  kicker,
  help,
  children,
  className = "",
}: {
  title: string;
  kicker?: string;
  help?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`journey-card ${className}`}>
      <div className="journey-card-head">
        <div className="flex min-w-0 items-center gap-2">
          <h2>{title}</h2>
          {help ? <HelpTip text={help} /> : null}
        </div>
        {kicker ? <span>{kicker}</span> : null}
      </div>
      {children}
    </section>
  );
}

export function MetricTile({
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
    <div className="metric-tile">
      <div className="flex items-center gap-2">
        <p>{label}</p>
        {help ? <HelpTip text={help} /> : null}
      </div>
      <strong>{value}</strong>
      <span>{detail}</span>
    </div>
  );
}

export function StageSteps() {
  const steps = [
    ["01", "Start", "Pick a topic, route, and parameters."],
    ["02", "Experiment", "Compare the result of that setup."],
    ["03", "Validation", "Decide what is worth believing."],
    ["04", "Visual", "Inspect the evidence from several angles."],
  ];

  return (
    <section className="journey-steps">
      {steps.map(([number, title, body]) => (
        <div key={number}>
          <span>{number}</span>
          <strong>{title}</strong>
          <p>{body}</p>
        </div>
      ))}
    </section>
  );
}

export function Pill({
  children,
  active,
  onClick,
  help,
}: {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
  help?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`journey-pill ${active ? "active" : ""}`}
      title={help}
    >
      {children}
    </button>
  );
}

export function TypeTag({ type }: { type: string }) {
  const lower = type.toLowerCase();
  const tone = lower.includes("quantum")
    ? "quantum"
    : lower.includes("hybrid") || lower.includes("ensemble")
      ? "hybrid"
      : "classical";
  return <span className={`type-tag ${tone}`}>{type}</span>;
}

export function MiniCircuit() {
  const wires = ["q0", "q1", "q2", "q3"];
  return (
    <div className="mini-circuit" aria-label="Quantum feature-map circuit preview">
      {wires.map((wire, index) => (
        <div className="circuit-row" key={wire}>
          <span>{wire}</span>
          <i />
          <b>H</b>
          <i />
          <b>RZ</b>
          <i />
          {index < wires.length - 1 ? <b className="wide">ZZ</b> : <b>RY</b>}
          <i />
          <b>M</b>
        </div>
      ))}
    </div>
  );
}
