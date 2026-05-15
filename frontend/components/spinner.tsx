export function Spinner({
  className = "h-4 w-4",
}: {
  className?: string;
}) {
  return (
    <svg
      className={`animate-spin ${className}`}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  );
}

export function LoadingBlock({
  text = "Loading…",
  className = "",
}: {
  text?: string;
  className?: string;
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={`inline-flex items-center gap-2 text-sm text-on-surface-variant ${className}`}
    >
      <Spinner className="h-4 w-4 text-primary" />
      <span>{text}</span>
    </div>
  );
}

export function SkeletonCards({
  count = 4,
  className = "",
}: {
  count?: number;
  className?: string;
}) {
  return (
    <div
      aria-hidden="true"
      className={`grid gap-4 sm:grid-cols-2 lg:grid-cols-4 ${className}`}
    >
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="h-16 animate-pulse rounded-lg bg-surface-container-high/50"
        />
      ))}
    </div>
  );
}
