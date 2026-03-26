export function PagePlaceholder({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          {title}
        </h1>
        {description ? (
          <p className="mt-1 text-sm text-on-surface-variant">{description}</p>
        ) : null}
      </header>
      {children ? (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5">
          {children}
        </div>
      ) : (
        <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5">
          <p className="text-sm text-on-surface-variant">
            This page is under construction.
          </p>
        </div>
      )}
    </div>
  );
}
